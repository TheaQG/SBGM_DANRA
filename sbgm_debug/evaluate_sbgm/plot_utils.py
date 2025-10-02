"""
    Publication plotting + table utilities for EDM evaluation.

    Reads CSV/JSON artifacts from <eval_root>/{tables,figures} and writes:
        - reliability_{thr}.png
        - spread_skill.png
        - fss_curves.png            (FSS vs scale_km per threshold)
        - psd_slope_bar.png
        - pit_hist.png
        - rank_hist.png
        - gev_rx{1,5}_return_levels.png
        - pot_diagnostics_{hr,pmm}.png

    Tables:
        - summary_metrics.csv           (CRPS mean, FSS@10km for {1, 5, 10 mm}, tails, wet-day freq)
        - summary_metrics.tex           (LaTeX version of above)
"""

from __future__ import annotations
import json
import csv
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sbgm.evaluate_sbgm.metrics_univariate import compute_isotropic_psd

# ---------- helpers ----------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _nice():
    plt.rcParams.update({
        "figure.figsize": (6,4),
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    })


# ---------- reliability ----------

def plot_reliability(eval_root: str, thr_mm_list=(1,5,10)):
    tdir = Path(eval_root) / "tables"
    fdir = _ensure_dir(Path(eval_root) / "figures")
    # read CSV without pandas
    rows = []
    with open(tdir / "reliability_bins.csv", 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "date": r.get("date", ""),
                    "thr": float(r.get("thr", 0.0)),
                    "bin_center": float(r.get("bin_center", r.get("prob_pred", 0.0))),
                    "prob_pred": float(r.get("prob_pred", 0.0)),
                    "freq_obs": float(r.get("freq_obs", 0.0)),
                    "count": int(float(r.get("count", 0))),
                })
            except Exception:
                continue

    for thr in thr_mm_list:
        # aggregate by bin_center
        bins = {}
        for r in rows:
            if abs(r["thr"] - float(thr)) > 1e-9:
                continue
            bc = r["bin_center"]
            if bc not in bins:
                bins[bc] = {"prob_sum":0.0, "freq_sum":0.0, "cnt_sum":0, "n":0}
            bins[bc]["prob_sum"] += r["prob_pred"]
            bins[bc]["freq_sum"] += r["freq_obs"]
            bins[bc]["cnt_sum"]  += r["count"]
            bins[bc]["n"]        += 1
        gb = []
        for bc, dct in bins.items():
            n = max(dct["n"], 1)
            gb.append({
                "bin_center": bc,
                "prob_pred": dct["prob_sum"] / n,
                "freq_obs":  dct["freq_sum"] / n,
                "count":     dct["cnt_sum"],
            })
        gb.sort(key=lambda x: x["bin_center"])  # <- list of dicts

        _nice()
        fig, ax = plt.subplots()
        ax.plot([0,1], [0,1], "--", lw=1, label="Perfect")
        xs = [row["prob_pred"] for row in gb]
        ys = [row["freq_obs"] for row in gb]
        cs = [row["count"] for row in gb]
        ax.plot(xs, ys, marker="o", lw=1.5, label=f"Thr ≥ {thr} mm")
        ax2 = ax.twinx()
        ax2.bar(xs, cs, width=0.07, alpha=0.25)
        ax2.set_ylim(0, max(max(cs)*1.2 if cs else 1, 1))
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xlabel("Forecast probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"Reliability diagram (≥ {thr} mm/day)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(str(fdir / f"reliability_{int(thr)}mm.png"), dpi=200)
        plt.close(fig)


# ---------- spread–skill ----------

def plot_spread_skill(eval_root: str):
    tdir = Path(eval_root) / "tables"
    fdir = _ensure_dir(Path(eval_root) / "figures")
    rows = []
    with open(tdir / "spread_skill.csv", 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "bin_center": float(r.get("bin_center", 0.0)),
                    "spread": float(r.get("spread", 0.0)),
                    "skill": float(r.get("skill", 0.0)),
                    "count": int(float(r.get("count", 0))),
                })
            except Exception:
                continue

    bins = {}
    for r in rows:
        bc = r["bin_center"]
        if bc not in bins:
            bins[bc] = {"spread_sum":0.0, "skill_sum":0.0, "count_sum":0, "n":0}
        bins[bc]["spread_sum"] += r["spread"]
        bins[bc]["skill_sum"]  += r["skill"]
        bins[bc]["count_sum"]  += r["count"]
        bins[bc]["n"]          += 1

    gb = []
    for bc, d in bins.items():
        n = max(d["n"], 1)
        gb.append({
            "bin_center": bc,
            "spread": d["spread_sum"]/n,
            "skill":  d["skill_sum"]/n,
            "count":  d["count_sum"],
        })
    gb.sort(key=lambda x: x["bin_center"])

    _nice()
    fig, ax = plt.subplots()
    xs  = [r["bin_center"] for r in gb]
    ys1 = [r["spread"] for r in gb]
    ys2 = [r["skill"]  for r in gb]
    ax.plot(xs, ys1, marker="o", label="Spread")
    ax.plot(xs, ys2, marker="s", label="Skill (MAE vs obs)")
    ax.set_xlabel("Ensemble mean (bin) • or another binning variable")
    ax.set_ylabel("Value (units of mm/day)")
    ax.set_title("Spread–skill")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(fdir / "spread_skill.png"), dpi=200)
    plt.close(fig)

# ---------- FSS curves ----------

def plot_fss_curves(eval_root: str, thr_mm_list=(1,5,10)):
    tdir = Path(eval_root) / "tables"
    fdir = _ensure_dir(Path(eval_root) / "figures")
    with open(tdir / "fss_summary.csv", 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        fss_cols = [c for c in fieldnames if c.lower().startswith("fss_")]
        items = []
        for row in reader:
            try:
                thr = float(row.get("thr", 0.0))
                for c in fss_cols:
                    try:
                        sc = float(c.split("_")[1].replace("km",""))
                    except Exception:
                        continue
                    val = float(row.get(c, 0.0))
                    items.append((thr, sc, val))
            except Exception:
                continue

    if not items:
        raise RuntimeError("No FSS_* columns found in fss_summary.csv")

    from collections import defaultdict
    by_thr = defaultdict(list)
    for thr, sc, val in items:
        by_thr[thr].append((sc, val))

    _nice()
    fig, ax = plt.subplots()
    for thr in thr_mm_list:
        pairs = sorted(by_thr.get(float(thr), []), key=lambda t: t[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.plot(xs, ys, marker="o", label=f"≥ {int(thr)} mm")
    ax.set_xlabel("Neighborhood scale (km)")
    ax.set_ylabel("FSS")
    ax.set_ylim(0,1)
    ax.set_title("FSS vs scale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(fdir / "fss_curves.png"), dpi=200)
    plt.close(fig)

# ---------- PSD slope ----------

def plot_psd_slope_bar(eval_root: str):
    tdir = Path(eval_root) / "tables"
    fdir = _ensure_dir(Path(eval_root) / "figures")
    with open(tdir / "psd_slope_summary.json", "r") as f:
        summ = json.load(f)

    labs, genv, obsv = [], [], []
    # Accept dict-of-dicts or list-of-dicts
    if isinstance(summ, dict):
        items = summ.items()
    elif isinstance(summ, list):
        items = [(str(i), itm) for i, itm in enumerate(summ)]
    else:
        items = []

    for k, v in items:
        if isinstance(v, dict) and ("gen_slope" in v) and ("hr_slope" in v):
            try:
                labs.append(k)
                genv.append(float(v["gen_slope"]))
                obsv.append(float(v["hr_slope"]))
            except Exception:
                continue

    if not labs:
        return
    
    _nice()
    x = np.arange(len(labs))
    w = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - w/2, obsv, width=w, label="Obs")
    ax.bar(x + w/2, genv, width=w, label="PMM")
    ax.set_xticks(x, labs)
    ax.set_ylabel("PSD slope (dimensionless)")
    ax.set_title("PSD slopes by season")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(fdir / "psd_slope_bar.png"), dpi=200)
    plt.close(fig)

# ---------- PIT & Rank ----------

def plot_pit_and_rank(eval_root: str, pit_bins=20):
    fdir = _ensure_dir(Path(eval_root) / "figures")
    # PIT
    pit_npz = fdir / "pit_values_all.npz"
    if pit_npz.exists():
        pits = np.load(pit_npz)["pits"]
        _nice()
        fig, ax = plt.subplots()
        ax.hist(pits, bins=pit_bins, range=(0,1), density=True)
        ax.hlines(1.0, 0, 1, linestyles="dashed")
        ax.set_xlim(0,1); ax.set_ylim(0, max(1.5, ax.get_ylim()[1]))
        ax.set_xlabel("PIT")
        ax.set_ylabel("Density")
        ax.set_title("PIT histogram")
        fig.tight_layout()
        fig.savefig(str(fdir / "pit_hist.png"), dpi=200)
        plt.close(fig)
    # Rank
    rank_npz = fdir / "rank_hist_counts.npz"
    if rank_npz.exists():
        counts = np.load(rank_npz)["counts"]
        _nice()
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(counts)), counts)
        ax.set_xlabel("Rank (0..M)")
        ax.set_ylabel("Count")
        ax.set_title("Rank histogram")
        fig.tight_layout()
        fig.savefig(str(fdir / "rank_hist.png"), dpi=200)
        plt.close(fig)

    

# ---------- Extremes plots (optional; robust to missing keys) ----------

def plot_return_levels(eval_root: str):
    tdir = Path(eval_root) / "tables"
    fdir = _ensure_dir(Path(eval_root) / "figures")
    def _safe(path):
        return json.load(open(path,"r")) if path.exists() else None
    rx1_hr  = _safe(tdir / "gev_rx1_hr.json")
    rx1_pmm = _safe(tdir / "gev_rx1_pmm.json")
    rx5_hr  = _safe(tdir / "gev_rx5_hr.json")
    rx5_pmm = _safe(tdir / "gev_rx5_pmm.json")
    def _plot_one(name, hr, pm):
        if not hr or not pm: return
        # expect {"return_periods":[...], "return_levels":[...], "ci_low":[...], "ci_high":[...]}
        rp  = np.array(hr.get("return_periods", []))
        rlo = np.array(hr.get("ci_low", [])); rhi = np.array(hr.get("ci_high", []))
        gp  = np.array(pm.get("return_periods", []))
        plo = np.array(pm.get("ci_low", [])); phi = np.array(pm.get("ci_high", []))
        if rp.size==0 or gp.size==0: return
        _nice()
        fig, ax = plt.subplots()
        ax.plot(rp, (rlo+rhi)/2, "-o", label="Obs")
        ax.fill_between(rp, rlo, rhi, alpha=0.2) # type: ignore
        ax.plot(gp, (plo+phi)/2, "-s", label="PMM")
        ax.fill_between(gp, plo, phi, alpha=0.2) # type: ignore
        ax.set_xscale("log")
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("Return level (mm/day)")
        ax.set_title(name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(fdir / f"{name}_return_levels.png"), dpi=200)
        plt.close(fig)
    _plot_one("rx1", rx1_hr, rx1_pmm)
    _plot_one("rx5", rx5_hr, rx5_pmm)

# ---------- Summary table (CSV + LaTeX) ----------

def write_summary_table(eval_root: str):
    tdir = Path(eval_root) / "tables"

    # CRPS mean
    crps_vals = []
    p = tdir / "crps_daily.csv"
    if p.exists():
        with open(p, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    crps_vals.append(float(r.get("crps", 0.0)))
                except Exception:
                    pass
    crps_mean = (sum(crps_vals) / len(crps_vals)) if crps_vals else float('nan')

    # FSS@10km
    fss10 = {}
    p = tdir / "fss_summary.csv"
    if p.exists():
        with open(p, 'r') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            cand = None
            for key in ["FSS_10km","fss_10km","FSS_10"]:
                if key in fields:
                    cand = key; break
            for r in reader:
                try:
                    thr = int(float(r.get("thr", 0)))
                    if cand is not None:
                        fss10[thr] = float(r.get(cand, 0.0))
                except Exception:
                    pass

    # tails summary
    tails_path = tdir / "tails_summary.json"
    if tails_path.exists():
        tails = json.load(open(tails_path, 'r'))
        wet_freq = tails.get("wet_day_frequency", float('nan'))
        p95 = tails.get("p95", float('nan'))
        p99 = tails.get("p99", float('nan'))
    else:
        wet_freq = p95 = p99 = float('nan')

    # write CSV
    out_csv = tdir / "summary_metrics.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["CRPS_mean","FSS10km_1mm","FSS10km_5mm","FSS10km_10mm","WetDayFreq","P95","P99"])
        w.writerow([
            crps_mean,
            fss10.get(1, float('nan')),
            fss10.get(5, float('nan')),
            fss10.get(10, float('nan')),
            wet_freq, p95, p99,
        ])

    # minimal LaTeX table
    out_tex = tdir / "summary_metrics.tex"
    vals = [
        ("CRPS_mean", crps_mean),
        ("FSS10km_1mm", fss10.get(1, float('nan'))),
        ("FSS10km_5mm", fss10.get(5, float('nan'))),
        ("FSS10km_10mm", fss10.get(10, float('nan'))),
        ("WetDayFreq", wet_freq),
        ("P95", p95),
        ("P99", p99),
    ]
    with open(out_tex, 'w') as f:
        f.write("\\begin{tabular}{lr}\n\\toprule\n")
        for k, v in vals:
            try:
                f.write(f"{k} & {float(v):.3f}\\\\\n")
            except Exception:
                f.write(f"{k} & {v}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def plot_psd_curves(
    gen_bt: torch.Tensor,  # [B,1,H,W] PMM (or forecast)
    hr_bt:  torch.Tensor,  # [B,1,H,W] observations
    mask: torch.Tensor | None = None,
    dx_km: float | None = None,
    out_dir: str | None = None,
    seasons: tuple = ("ALL",),  # kept for API compatibility; not used to split here
    fname: str = "psd_curves.png",
):
    """
    Plot isotropic PSD curves averaged over the batch for GEN vs HR.
    If dx_km is None, defaults to 1.0 (relative wavenumber).
    """
    if dx_km is None: dx_km = 1.0
    out_path = Path(out_dir) if out_dir is not None else Path(".")
    out_path.mkdir(parents=True, exist_ok=True)

    psd_gen = compute_isotropic_psd(gen_bt, dx_km=dx_km, mask=mask)
    psd_hr  = compute_isotropic_psd(hr_bt,  dx_km=dx_km, mask=mask)

    k  = psd_gen["k"].detach().cpu().numpy()
    Pg = psd_gen["psd"].detach().cpu().numpy()
    Ph = psd_hr["psd"].detach().cpu().numpy()

    # drop k=0 for log plots
    k_plot  = k[1:]
    Pg_plot = Pg[1:]
    Ph_plot = Ph[1:]

    plt.figure(figsize=(6,4))
    plt.loglog(k_plot, Pg_plot, label="PMM", linewidth=1.8)
    plt.loglog(k_plot, Ph_plot, label="Obs", linewidth=1.8)
    plt.xlabel("Wavenumber k (1/km)")
    plt.ylabel("Power (arb.)")
    plt.title("Isotropic PSD (batch mean)")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / fname, dpi=200)
    plt.close()










# ---------- Driver ----------

def make_publication_outputs(cfg):
    cfg_full_gen_eval = cfg.get("full_gen_eval", {})
    eval_root = cfg_full_gen_eval.get("eval_dir", None)  # If None, use default
    if eval_root is None:
        from sbgm.evaluate_sbgm.evaluation_main import _default_eval_dir
        eval_root = _default_eval_dir(cfg)


    thresholds = cfg.evaluation.get("thresholds_mm", [1,5,10])
    pit_bins = cfg.evaluation.get("pit_bins", 20)
    
    eval_root_str = str(eval_root)
    plot_reliability(eval_root_str, thr_mm_list=thresholds)
    plot_spread_skill(eval_root_str)
    plot_fss_curves(eval_root_str, thr_mm_list=thresholds)
    plot_psd_slope_bar(eval_root_str)
    plot_pit_and_rank(eval_root_str, pit_bins=pit_bins)
    plot_return_levels(eval_root_str)
    write_summary_table(eval_root_str)

