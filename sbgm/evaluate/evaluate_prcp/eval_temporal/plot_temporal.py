from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sbgm.evaluate.evaluate_prcp.plot_utils import _nice, _ensure_dir, _savefig
from sbgm.variable_utils import get_color_for_model

col_hr = get_color_for_model("HR")
col_gen = get_color_for_model("GEN")
col_pmm = get_color_for_model("PMM")
col_lr = get_color_for_model("LR")
_COL = {"HR": col_hr, "PMM": col_pmm, "GEN": col_gen, "LR": col_lr}

SET_DPI = 300

def plot_timeseries(figdir: Path, group: str, dates: np.ndarray, series_dict: Dict[str, np.ndarray]) -> None:
    _nice()
    # parse dates to datetime64 for cleaner locators
    try:
        if dates.dtype.kind in {"U", "S", "O"}:
            # accept YYYYMMDD or YYYY-MM-DD
            dconv = []
            for s in dates:
                s = str(s)
                if len(s) == 8 and s.isdigit():
                    dconv.append(np.datetime64(f"{s[:4]}-{s[4:6]}-{s[6:8]}"))
                else:
                    dconv.append(np.datetime64(s))
            dts = np.array(dconv, dtype="datetime64[D]")
        else:
            dts = dates.astype("datetime64[D]")
    except Exception:
        dts = np.arange(len(dates))
    fig, ax = plt.subplots(1,1, figsize=(10,4.2))
    for label, arr in series_dict.items():
        ax.plot(dts, arr, label=label, color=_COL.get(label, None), linewidth=1.2)
    ax.set_title(f"{group}: domain-mean precipitation")
    ax.set_ylabel("mm/day"); ax.set_xlabel("date")
    # Tidy x ticks
    try:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        fig.autofmt_xdate()
    except Exception:
        pass
    leg = ax.legend(ncol=3, frameon=True)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)



    plt.close(fig)

def plot_autocorr(figdir: Path, group: str, ac_dict: Dict[str, np.ndarray]) -> None:
    _nice()
    fig, ax = plt.subplots(1,1, figsize=(6.5,3.2))
    for label, ac in ac_dict.items():
        ax.plot(np.arange(1, len(ac)+1), ac, marker="o", markersize=3.0, linewidth=1.2,
                label=label, color=_COL.get(label, None))
    ax.set_title(f"{group}: lag-k autocorrelation")
    ax.set_xlabel("lag (days)"); ax.set_ylabel("autocorr")
    ax.set_ylim(-0.2, 1.0)
    ax.set_ylim(-0.15, 1.0)
    leg = ax.legend(ncol=3, frameon=True)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)
    _savefig(fig, figdir / f"temporal_{group}_autocorr.png", dpi=SET_DPI)
    plt.close(fig)

def plot_spell_pmf(figdir: Path, group: str, metrics: Dict[str, dict], pair_metrics: Optional[Dict[str, Dict[str, float]]] = None) -> None:
    _nice()
    # determine safe maxima for wet/dry bins so `bins` is not used uninitialized
    if metrics:
        # handle possibly different bin arrays and ensure non-empty arrays
        wet_max_candidates = [int(np.max(d["wet_bins"])) for d in metrics.values() if len(d.get("wet_bins", [])) > 0]
        dry_max_candidates = [int(np.max(d["dry_bins"])) for d in metrics.values() if len(d.get("dry_bins", [])) > 0]
        max_wet = max(wet_max_candidates) if wet_max_candidates else 1
        max_dry = max(dry_max_candidates) if dry_max_candidates else 1
    else:
        max_wet = 1; max_dry = 1
    fig, axes = plt.subplots(1,2, figsize=(11,3.3), sharey=False)
    # --- Wet spells ---
    ax = axes[0]
    for label, d in metrics.items():
        bins = d["wet_bins"]; pmf = d["wet_pmf"]; p = d["wet_geom_p"]
        col = _COL.get(label, None)
        ax.plot(bins, pmf, marker="o", markersize=3.0, linewidth=1.2, label=f"{label} (emp)", color=col)
        if np.isfinite(p):
            ax.plot(bins, (p * (1-p)**(bins-1)), linestyle="--", linewidth=1.2, label=f"{label} geom fit", color=col)
    ax.set_title(f"{group}: wet-spell length PMF")
    ax.set_xlim(1, max_wet); ax.set_ylim(-0.005, None)
    leg = ax.legend(ncol=2, frameon=True)
    leg.get_frame().set_edgecolor("black"); leg.get_frame().set_linewidth(0.8)
    if pair_metrics and "wet" in pair_metrics:
        txt = []
        if "JSD_GEN_HR" in pair_metrics["wet"]:
            txt.append(f"JSD(PMM,HR)={pair_metrics['wet']['JSD_GEN_HR']:.3f}")
        if "JSD_LR_HR" in pair_metrics["wet"]:
            txt.append(f"JSD(LR,HR)={pair_metrics['wet']['JSD_LR_HR']:.3f}")
        if "KS_GEN_HR" in pair_metrics["wet"]:
            txt.append(f"KS(PMM,HR)={pair_metrics['wet']['KS_GEN_HR']:.3f}")
        if "KS_LR_HR" in pair_metrics["wet"]:
            txt.append(f"KS(LR,HR)={pair_metrics['wet']['KS_LR_HR']:.3f}")
        if txt:
            ax.text(0.98, 0.5, "\n".join(txt), transform=ax.transAxes, ha="right", va="top",
                    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"), fontsize=9)
    # --- Dry spells ---
    ax = axes[1]
    for label, d in metrics.items():
        bins = d["dry_bins"]; pmf = d["dry_pmf"]; p = d["dry_geom_p"]
        col = _COL.get(label, None)
        ax.plot(bins, pmf, marker="o", markersize=3.0, linewidth=1.2, label=f"{label} (emp)", color=col)
        if np.isfinite(p):
            ax.plot(bins, (p * (1-p)**(bins-1)), linestyle="--", linewidth=1.2, label=f"{label} geom fit", color=col)
    ax.set_title(f"{group}: dry-spell length PMF")
    ax.set_xlabel("length (days)"); ax.set_ylabel("probability")
    ax.set_xlim(1, max_dry); ax.set_ylim(-0.005, None)
    leg = ax.legend(ncol=2, frameon=True)
    leg.get_frame().set_edgecolor("black"); leg.get_frame().set_linewidth(0.8)
    if pair_metrics and "dry" in pair_metrics:
        txt = []
        if "JSD_GEN_HR" in pair_metrics["dry"]:
            txt.append(f"JSD(PMM,HR)={pair_metrics['dry']['JSD_GEN_HR']:.3f}")
        if "JSD_LR_HR" in pair_metrics["dry"]:
            txt.append(f"JSD(LR,HR)={pair_metrics['dry']['JSD_LR_HR']:.3f}")
        if "KS_GEN_HR" in pair_metrics["dry"]:
            txt.append(f"KS(PMM,HR)={pair_metrics['dry']['KS_GEN_HR']:.3f}")
        if "KS_LR_HR" in pair_metrics["dry"]:
            txt.append(f"KS(LR,HR)={pair_metrics['dry']['KS_LR_HR']:.3f}")
        if txt:
            ax.text(0.98, 0.5, "\n".join(txt), transform=ax.transAxes, ha="right", va="top",
                    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"), fontsize=9)
    
    _savefig(fig, figdir / f"temporal_{group}_spell_pmf.png", dpi=SET_DPI)
    plt.close(fig)

def plot_temporal(figdir: Path, group: str, dates: np.ndarray, series_dict: Dict[str, np.ndarray], metrics: Dict[str, dict], pair_metrics: Optional[Dict[str, Dict[str, float]]] = None) -> None:
    """
    Master plotting entry point for the temporal block: time series, autocorrelation, and spell PMFs.
    """
    plot_timeseries(figdir, group, dates, series_dict)
    ac_dict = {k: d["autocorr"] for k, d in metrics.items() if "autocorr" in d}
    if ac_dict:
        plot_autocorr(figdir, group, ac_dict)
    plot_spell_pmf(figdir, group, metrics, pair_metrics=pair_metrics)
