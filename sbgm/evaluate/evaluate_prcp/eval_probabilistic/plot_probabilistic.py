"""
    Plotting utilities for the precipitation probabilistic evaluation block.

    Reads the artefacts that "sbgm.evaluate.evaluate_prcp.eval_probabilistic" writes to:
        <eval_root>/tables/
            - prob_crps_daily.csv
            - prob_reliability_{thr}.csv
            - prob_spread_skill.csv
            - prob_rank_histogram.npz
            - prob_pit_values.npz

    and creates plots in:
        <eval_root>/figures/
            - prob_crps_summary.png
            - prob_pit_hist.png
            - prob_rank_hist.png
            - prob_reliability_{thr}.png
            - prob_spread_skill.png
"""


from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
import logging

from mpl_toolkits.axes_grid1 import make_axes_locatable

from sbgm.evaluate.evaluate_prcp.plot_utils import _ensure_dir, _savefig, _nice, _to_date_safe, _season_from_month
from sbgm.variable_utils import get_cmap_for_variable

logger = logging.getLogger(__name__)

# ================================================================================
# 1. PIT
# ================================================================================

def plot_pit(eval_root: str | Path, *, bins: int = 20):
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")

    pit_file = tables_dir / "prob_pit_values.npz"
    if not pit_file.exists():
        # nothing to do
        return

    arr = np.load(pit_file)
    pit = arr["pit"]

    _nice()
    fig, ax = plt.subplots()
    ax.hist(pit, bins=bins, range=(0, 1), density=True)
    ax.hlines(1.0, 0, 1, linestyles="dashed", colors=["0.4"], linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("PIT")
    ax.set_ylabel("Density")
    ax.set_title("PIT histogram")
    _savefig(fig, figs_dir / "prob_pit_hist.png")



# ================================================================================
# 2. Rank histogram
# ================================================================================

def plot_rank(eval_root: str | Path):
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")

    rh_file = tables_dir / "prob_rank_histogram.npz"
    if not rh_file.exists():
        return

    arr = np.load(rh_file)
    counts = arr["rank_hist"]

    _nice()
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(counts)), counts, width=0.8)
    ax.set_xlabel("Rank (0..M)")
    ax.set_ylabel("Count")
    ax.set_title("Rank histogram")
    _savefig(fig, figs_dir / "prob_rank_hist.png")




# ================================================================================
# 3. Reliability diagrams
# ================================================================================

def plot_reliability(
    eval_root: str | Path,
    *,
    thresholds: Sequence[float] = (1.0, 5.0, 10.0),
    min_count_to_show: int = 150,
):
    """
    Expects files like:
        <tables>/prob_reliability_1.0mm.csv
        <tables>/prob_reliability_5.0mm.csv
    written by evaluate_probabilistic.py.
    """
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")

    for thr in thresholds:
        fpath = tables_dir / f"prob_reliability_{thr:.1f}mm.csv"
        if not fpath.exists():
            continue

        lines = fpath.read_text().strip().splitlines()
        if len(lines) <= 1:
            continue

        bc, pf, po, cnt = [], [], [], []
        for ln in lines[1:]:
            s = ln.split(",")
            if len(s) < 4:
                continue
            bc.append(float(s[0]))
            pf.append(float(s[1]))
            po.append(float(s[2]))
            cnt.append(int(float(s[3])))

        bc = np.array(bc, dtype=float)
        pf = np.array(pf, dtype=float)
        po = np.array(po, dtype=float)
        cnt = np.array(cnt, dtype=int)

        # sort by forecast prob so lines don't zig-zag
        order = np.argsort(pf)
        bc = bc[order]
        pf = pf[order]
        po = po[order]
        cnt = cnt[order]

        _nice()
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Perfect")

        # draw high-count bins as line+markers, low-count as faint dots
        high = cnt >= min_count_to_show
        low = ~high

        if np.any(high):
            ax.plot(pf[high], po[high], "o-", label=f"≥ {thr:.1f} mm", linewidth=1.0)

        if np.any(low):
            ax.scatter(pf[low], po[low], s=24, alpha=0.35, edgecolors="none")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Forecast probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"Reliability – {thr:.1f} mm/day")
        ax.legend(loc="lower right")

        # add count axis (optional)
        ax2 = ax.twinx()
        ax2.bar(pf, cnt, width=0.025, alpha=0.18)
        ax2.set_ylabel("Bin count")

        _savefig(fig, figs_dir / f"prob_reliability_{thr:.1f}mm.png")



# ================================================================================
# 4. Spread-skill
# ================================================================================


def plot_spread_skill(
    eval_root: str | Path,
    *,
    min_count_to_show: int = 500,
):
    """
    Reads:
        <tables>/prob_spread_skill.csv
    with rows:
        date,spread_mean,skill_mean
    and plots:
      1) time series with sparse date labels and mean lines
      2) spread vs skill scatter
    """
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")

    csv_path = tables_dir / "prob_spread_skill.csv"
    if not csv_path.exists():
        return

    lines = csv_path.read_text().strip().splitlines()
    if len(lines) <= 1:
        return

    dates, spreads, skills = [], [], []
    for ln in lines[1:]:
        s = ln.split(",")
        if len(s) < 3:
            continue
        dates.append(s[0].strip())
        try:
            spreads.append(float(s[1]))
        except Exception:
            spreads.append(np.nan)
        try:
            skills.append(float(s[2]))
        except Exception:
            skills.append(np.nan)

    x = np.arange(len(dates))

    # --- time series ---
    _nice()
    fig, ax = plt.subplots()
    ax.plot(x, spreads, label="spread (mean)")
    ax.plot(x, skills, label="skill (mean)")

    # mean lines
    if len(spreads):
        m_spread = float(np.nanmean(spreads))
        ax.axhline(m_spread, color="0.5", linestyle="--", linewidth=1.0,
                   label=f"spread mean={m_spread:.2f}")
    if len(skills):
        m_skill = float(np.nanmean(skills))
        ax.axhline(m_skill, color="0.7", linestyle=":", linewidth=1.0,
                   label=f"skill mean={m_skill:.2f}")

    # sparse date labels
    n_labels = min(10, len(dates))
    if n_labels > 0:
        idxs = np.linspace(0, len(dates) - 1, n_labels, dtype=int)
        ax.set_xticks(idxs)
        ax.set_xticklabels([dates[i] for i in idxs], rotation=30, ha="right")

    ax.set_xlabel("sample index (date order)")
    ax.set_ylabel("mm/day")
    ax.set_title("Spread–skill diagnostics (per date)")
    ax.legend()
    _savefig(fig, figs_dir / "prob_spread_skill.png")

    # --- spread vs skill scatter ---
    _nice()
    fig, ax = plt.subplots()
    ax.scatter(spreads, skills, s=14, alpha=0.6, edgecolors="none")
    maxv = float(max(
        np.nanmax(spreads) if len(spreads) else 0.0,
        np.nanmax(skills) if len(skills) else 0.0,
        1.0
    ))
    ax.plot([0, maxv], [0, maxv], "k--", lw=1.0, label="spread = skill")
    ax.set_xlim(0, maxv)
    ax.set_ylim(0, maxv)
    ax.set_xlabel("Spread (mm/day)")
    ax.set_ylabel("Skill (mm/day)")
    ax.set_title("Spread vs skill")
    ax.legend()
    _savefig(fig, figs_dir / "prob_spread_vs_skill.png")


# ================================================================================
# 4b. Energy score & Variogram score
# ================================================================================

def plot_energy_variogram(eval_root: str | Path):
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")

    es_path = tables_dir / "prob_energy_daily.csv"
    vs_path = tables_dir / "prob_variogram_daily.csv"

    if (not es_path.exists()) and (not vs_path.exists()):
        return

    dates_es, es_vals = [], []
    if es_path.exists():
        lines = es_path.read_text().strip().splitlines()
        for ln in lines[1:]:
            s = ln.split(",")
            if len(s) < 2:
                continue
            dates_es.append(s[0].strip())
            try:
                es_vals.append(float(s[1]))
            except Exception:
                es_vals.append(np.nan)

    dates_vs, vs_vals = [], []
    if vs_path.exists():
        lines = vs_path.read_text().strip().splitlines()
        for ln in lines[1:]:
            s = ln.split(",")
            if len(s) < 2:
                continue
            dates_vs.append(s[0].strip())
            try:
                vs_vals.append(float(s[1]))
            except Exception:
                vs_vals.append(np.nan)

    _nice()
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.0))
    ax1, ax2 = axs

    # --- ES ---
    if es_vals:
        x = np.arange(len(es_vals))
        ax1.plot(x, es_vals, lw=1.0, label="ES")
        m_es = float(np.nanmean(es_vals))
        ax1.axhline(m_es, color="0.5", ls="--", lw=0.8, label=f"mean={m_es:.3f}")
        ax1.set_title("Energy score (daily)")
        ax1.set_xlabel("sample index")
        ax1.set_ylabel("ES")
        ax1.legend()
        ax1.grid(True, ls=":", alpha=0.4)

    # --- VS ---
    if vs_vals:
        x = np.arange(len(vs_vals))
        ax2.plot(x, vs_vals, lw=1.0, label="VS")
        m_vs = float(np.nanmean(vs_vals))
        ax2.axhline(m_vs, color="0.5", ls="--", lw=0.8, label=f"mean={m_vs:.3f}")
        ax2.set_title("Variogram score (daily)")
        ax2.set_xlabel("sample index")
        ax2.set_ylabel("VS")
        ax2.legend()
        ax2.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    _savefig(fig, figs_dir / "prob_energy_variogram.png")


# ================================================================================
# 5. CRPS, examples and timeseries
# ================================================================================

def plot_crps_examples(
        eval_root: str | Path,
        *,
        gen_root: Optional[str | Path] = None,
        n_examples: int = 6,  # 3 best + 3 worst
):
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")

    crps_path = tables_dir / "prob_crps_daily.csv"
    if not crps_path.exists():
        return

    lines = crps_path.read_text().strip().splitlines()
    if len(lines) <= 1:
        return

    rows = []
    for ln in lines[1:]:
        s = ln.split(",")
        if len(s) < 2:
            continue
        date_s = s[0].strip()
        try:
            crps_v = float(s[1])
        except Exception:
            continue
        rows.append((date_s, crps_v))

    if not rows:
        return

    # pick 3 best + 3 worst, deprioritizing zero-CRPS days (no-rain)
    rows_sorted = sorted(rows, key=lambda x: x[1])
    eps = 0.01
    nonzero_rows = [r for r in rows_sorted if r[1] > eps]
    zero_rows = [r for r in rows_sorted if r[1] <= eps]
    n_half = max(1, n_examples // 2)
    if len(nonzero_rows) >= n_half:
        best = nonzero_rows[:n_half]
    else:
        # Take all nonzero, fill up with zero-CRPS days
        best = nonzero_rows + zero_rows[:(n_half - len(nonzero_rows))]
    worst = rows_sorted[-n_half:]
    selected = best + worst

    # --------------------------
    # get generation root
    # --------------------------
    if gen_root is not None:
        gen_root = Path(gen_root)
        if not gen_root.exists():
            logger.warning(f"[plot_crps_examples] Provided gen_root={gen_root} does not exist – falling back to inference.")
            gen_root = None

    if gen_root is None:
        # automatic inference as fallback
        model_name = eval_root.parents[2].name  # folder right above 'prcp'
        for p in eval_root.parents:
            if p.name == "generated_samples":
                cand = p / "generation" / model_name
                if cand.exists():
                    gen_root = cand
                    break
                cand2 = p / "generation"
                if cand2.exists():
                    gen_root = cand2
                    break

    # --------------------------
    # if still None -> fallback bar plot
    # --------------------------
    if gen_root is None or not gen_root.exists():
        logger.info(f"[plot_crps_examples] Could not determine generation root from {eval_root}; falling back to bar plot.")
        _nice()
        fig, ax = plt.subplots()
        labs = [d for d, _ in selected]
        vals = [v for _, v in selected]
        x = np.arange(len(vals))
        ax.bar(x, vals)
        ax.set_xticks(x, labs, rotation=30, ha="right")
        ax.set_ylabel("CRPS")
        ax.set_title("CRPS – selected days")
        _savefig(fig, figs_dir / "prob_crps_examples.png")
        return

    # --------------------------
    # helpers to load HR / PMM
    # --------------------------
    def _load_np(p: Path, keys: list[str]):
        if not p.exists():
            return None
        d = np.load(p, allow_pickle=True)
        for k in keys:
            if k in d:
                arr = np.asarray(d[k])
                # squeeze singleton leading dims
                while arr.ndim > 2 and 1 in arr.shape:
                    arr = np.squeeze(arr)
                return arr
        return None

    panels = []
    # use project-wide colormap for precipitation
    cmap = get_cmap_for_variable("prcp")
    for date_s, crps_v in selected:
        hr = _load_np(gen_root / "lr_hr" / f"{date_s}.npz",
                      ["hr", "hr_phys", "target", "truth", "obs", "y"])
        if hr is None:
            hr = _load_np(gen_root / "lr_hr_phys" / f"{date_s}.npz",
                          ["hr", "hr_phys", "target", "truth", "obs", "y"])
        pmm = _load_np(gen_root / "pmm_phys" / f"{date_s}.npz",
                       ["pmm", "x", "y_pred"])
        if pmm is None:
            pmm = _load_np(gen_root / "pmm" / f"{date_s}.npz",
                           ["pmm", "x", "y_pred"])
        panels.append((date_s, crps_v, hr, pmm))

    _nice()
    ncols = len(panels)
    # IMPORTANT: no constrained_layout here, since _savefig() does tight_layout()
    fig, axs = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.0))

    last_im = None
    for j, (date_s, crps_v, hr, pmm) in enumerate(panels):
        ax_hr = axs[0, j] if ncols > 1 else axs[0]
        ax_pm = axs[1, j] if ncols > 1 else axs[1]

        # per-day color range
        this_vals = []

        if hr is not None:
            this_vals.append(hr.ravel())
        if pmm is not None:
            this_vals.append(pmm.ravel())
        if this_vals:
            this_all = np.concatenate(this_vals)
            vmax_j = float(np.percentile(this_all, 99.5))
            if vmax_j < 1.0:
                vmax_j = 1.0
        else:
            vmax_j = 1.0
        vmin_j = 0.0

        hr_or_zeros = hr if hr is not None else np.zeros((2, 2))
        pmm_or_zeros = pmm if pmm is not None else np.zeros((2, 2))

        im = ax_hr.imshow(hr_or_zeros, origin="lower", vmin=vmin_j, vmax=vmax_j, cmap=cmap)
        ax_hr.set_title(f"{date_s}\nCRPS={crps_v:.3f}")
        ax_hr.set_xticks([]); ax_hr.set_yticks([])

        ax_pm.imshow(pmm_or_zeros, origin="lower", vmin=vmin_j, vmax=vmax_j, cmap=cmap)
        ax_pm.set_xticks([]); ax_pm.set_yticks([])

        if j == 0:
            ax_hr.set_ylabel("HR")
            ax_pm.set_ylabel("PMM")

        # Individual colorbar for each column (day pair), attached to PMM axis
        divider = make_axes_locatable(ax_pm)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        fig.colorbar(im, cax=cax, label="mm/day")

    _savefig(fig, figs_dir / "prob_crps_examples.png")


def plot_crps_timeseries(
        eval_root: str | Path,
):
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")

    crps_path = tables_dir / "prob_crps_daily.csv"
    if not crps_path.exists():
        return

    lines = crps_path.read_text().strip().splitlines()
    if len(lines) <= 1:
        return

    dates_dt: list[datetime] = []
    dates_raw: list[str] = []
    crps_vals: list[float] = []
    seasons: list[str] = []

    for ln in lines[1:]:
        s = ln.split(",")
        if len(s) < 2:
            continue
        d_raw = s[0].strip()
        d_dt = _to_date_safe(d_raw)
        try:
            v = float(s[1])
        except Exception:
            continue
        if d_dt is None:
            continue
        dates_dt.append(d_dt)
        dates_raw.append(d_raw)
        crps_vals.append(v)
        seasons.append(_season_from_month(d_dt.month))

    if not crps_vals:
        return

    # sort by date
    order = np.argsort(np.array(dates_dt, dtype="datetime64[s]"))
    dates_dt = [dates_dt[i] for i in order]
    dates_raw = [dates_raw[i] for i in order]
    crps_vals = [crps_vals[i] for i in order]
    seasons = [seasons[i] for i in order]

    # --- figure 1: daily timeseries ---
    _nice()
    fig, ax = plt.subplots()
    x = np.arange(len(crps_vals))
    ax.plot(x, crps_vals, "-", lw=0.9, label="daily CRPS")

    # 7d smoother
    if len(crps_vals) >= 7:
        window = 7
        sm = np.convolve(crps_vals, np.ones(window)/window, mode="valid")
        ax.plot(np.arange(window-1, window-1+len(sm)), sm, lw=1.5, label="7d mean")

    # horizontal mean line
    mean_crps = float(np.mean(crps_vals))
    ax.axhline(mean_crps, color="0.5", linestyle="--", linewidth=1.0,
               label=f"mean={mean_crps:.2f}")

    # sparse date labels (max 10)
    n_labels = min(10, len(dates_raw))
    idxs = np.linspace(0, len(dates_raw) - 1, n_labels, dtype=int)
    ax.set_xticks(idxs)
    ax.set_xticklabels([dates_raw[i] for i in idxs], rotation=30, ha="right")

    ax.set_xlabel("time (sorted by date)")
    ax.set_ylabel("CRPS")
    ax.set_title("Daily CRPS")
    ax.legend()
    _savefig(fig, figs_dir / "prob_crps_timeseries.png")

    # --- figure 2: seasonal + ALL boxplot ---
    by_season: Dict[str, list[float]] = {"DJF": [], "MAM": [], "JJA": [], "SON": []}
    for v, s in zip(crps_vals, seasons):
        by_season[s].append(v)

    _nice()
    fig, ax = plt.subplots()
    data_all = crps_vals
    data = [data_all,
            by_season["DJF"],
            by_season["MAM"],
            by_season["JJA"],
            by_season["SON"]]
    labels = ["ALL", "DJF", "MAM", "JJA", "SON"]
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("CRPS")
    ax.set_title("Seasonal CRPS distribution")

    # # tiny legend explaining the green mean symbol
    # legend_elems = [
    #     Patch(facecolor="white", edgecolor="black", label="IQR + whiskers"),
    #     Patch(facecolor="white", edgecolor="none", label="● mean (green)")
    # ]
    # ax.legend(handles=legend_elems, loc="upper right")

    _savefig(fig, figs_dir / "prob_crps_seasonal.png")

def plot_crps_spatial(eval_root: str | Path):
    eval_root = Path(eval_root)
    tables_dir = eval_root / "tables"
    figs_dir = _ensure_dir(eval_root / "figures")
    fpath = tables_dir / "prob_crps_mean_map.npz"
    if not fpath.exists():
        return
    data = np.load(fpath)
    crps_map = data["crps_mean_map"]
    mean_crps = np.mean(crps_map)

    _nice()
    cmap = get_cmap_for_variable("prcp")
    fig, ax = plt.subplots()
    im = ax.imshow(crps_map, origin="lower", cmap=cmap)
    ax.set_title(f"Mean CRPS (per pixel, over time)\nOverall mean: {mean_crps:.2f}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="CRPS")
    _savefig(fig, figs_dir / "prob_crps_mean_map.png")



# ================================================================================
# Master plotting function
# ================================================================================

def plot_probabilistic(
        eval_root: str | Path,
        *,
        gen_root: Optional[str | Path] = None,
        thresholds: Sequence[float] = (1.0, 5.0, 10.0),
        pit_bins: int = 20,
):
    """
        One master function to plot all probabilistic evaluation plots.
        To be called from evaluate_probabilistic.py after all tables have been written.
    """
    plot_pit(eval_root, bins=pit_bins)
    plot_rank(eval_root)
    plot_reliability(eval_root, thresholds=thresholds)
    plot_spread_skill(eval_root)
    plot_energy_variogram(eval_root)
    plot_crps_examples(eval_root, gen_root=gen_root)
    plot_crps_timeseries(eval_root)
    plot_crps_spatial(eval_root)