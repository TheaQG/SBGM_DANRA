# sbgm/evaluate/evaluate_prcp/eval_distributional/plot_distributional.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import logging

from sbgm.evaluate.evaluate_prcp.plot_utils import _ensure_dir, _nice, _savefig
from sbgm.variable_utils import get_color_for_model

logger = logging.getLogger(__name__)

SET_DPI = 300


def plot_distributional(dist_root: str | Path) -> None:
    dist_root = Path(dist_root)
    tables = dist_root / "tables"
    figs = _ensure_dir(dist_root / "figures")

    # Set colors
    col_hr = get_color_for_model("hr")
    col_pmm = get_color_for_model("pmm")
    col_ens = get_color_for_model("ensemble")
    col_lr  = get_color_for_model("lr")

    bins_path = tables / "dist_bins.csv"
    if not bins_path.exists():
        logger.warning("[plot_distributional] No dist_bins.csv – skipping.")
        return
    bins = np.loadtxt(bins_path, delimiter=",", skiprows=1) if bins_path.read_text().startswith("bin_edge") else np.loadtxt(bins_path, delimiter=",")
    # If 1D
    if bins.ndim > 1:
        bins = bins[:, 0]

    def _read_hist(name: str) -> Optional[np.ndarray]:
        p = tables / f"dist_{name}.csv"
        if not p.exists():
            return None
        xs, cs = [], []
        with open(p, "r") as f:
            next(f)  # header
            for ln in f:
                s = ln.strip().split(",")
                if len(s) != 2:
                    continue
                xs.append(int(s[0])); cs.append(int(float(s[1])))
        return np.array(cs, dtype=float)

    hr = _read_hist("hr")
    gen = _read_hist("gen")
    lr  = _read_hist("lr")

    # Try to read ensemble artifacts
    ens_mode = None
    gen_ens_pool = None
    gen_ens_mean = None
    ens_npz = tables / "dist_member_histograms.npz"
    if (tables / "dist_gen_ens_pool.csv").exists():
        xs, cs = [], []
        with open(tables / "dist_gen_ens_pool.csv", "r") as f:
            next(f)
            for ln in f:
                s = ln.strip().split(",")
                if len(s) == 2:
                    xs.append(int(s[0])); cs.append(float(s[1]))
        gen_ens_pool = np.array(cs, dtype=float)
        ens_mode = "pool"
    if (tables / "dist_gen_ens_mean.csv").exists():
        xs, ps = [], []
        with open(tables / "dist_gen_ens_mean.csv", "r") as f:
            next(f)
            for ln in f:
                s = ln.strip().split(",")
                if len(s) == 2:
                    xs.append(int(s[0])); ps.append(float(s[1]))
        gen_ens_mean = np.array(ps, dtype=float)
        ens_mode = "member_mean"
    q10 = q50 = q90 = None
    if ens_npz.exists():
        try:
            d = np.load(ens_npz)
            q10 = d.get("pdf_q10", None)
            q50 = d.get("pdf_q50", None)
            q90 = d.get("pdf_q90", None)
            if ens_mode is None and "mode" in d:
                try:
                    ens_mode = str(d["mode"])  # may be 0-d array
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[plot_distributional] Could not load ensemble NPZ: {e}")

    metrics_path = tables / "dist_metrics.csv"
    gen_text = None
    lr_text = None
    if metrics_path.exists():
        lines = metrics_path.read_text().strip().splitlines()
        # header: ref,comp,wasserstein,ks_stat,ks_p,kl_hr_to_x
        rows = []
        for ln in lines[1:]:
            ref, comp, w1, ks_s, ks_p, kl = ln.split(",")
            rows.append((ref, comp, float(w1), float(ks_s), float(ks_p), float(kl)))
        # Separate GEN and LR metrics
        gen_parts = []
        lr_parts = []
        for (ref, comp, w1, kss, ksp, kl) in rows:
            if comp.lower() in ("gen_ens_pool", "gen_ens_mean", "gen_pmm"):
                txt = (
                    f"{comp.upper()} vs {ref.upper()}:\n"
                    f"  W1  = {w1:.3f}\n"
                    f"  KS  = {kss:.3f} (p={ksp:.2f})\n"
                    f"  KL  = {kl:.3f}"
                )
                gen_parts.append(txt)
        gen_text = "\n".join(gen_parts).strip() if gen_parts else None

        for (ref, comp, w1, kss, ksp, kl) in rows:
            if comp.lower() == "lr":
                txt = (
                    f"{comp.upper()} vs {ref.upper()}:\n"
                    f"  W1  = {w1:.3f}\n"
                    f"  KS  = {kss:.3f} (p={ksp:.2f})\n"
                    f"  KL  = {kl:.3f}"
                )
                lr_parts.append(txt)
        lr_text = "\n".join(lr_parts).strip() if lr_parts else None

    # Plot
    _nice()
    fig, ax = plt.subplots(figsize=(6,4))

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    eps = 1e-12
    # normalize to PDF shape (so the area is comparable)
    def _norm(h: np.ndarray | None) -> np.ndarray | None:
        if h is None:
            return None
        s = h.sum()
        if s <= 0:
            return h
        return h / s

    hr_n = _norm(hr)
    gen_n = _norm(gen)
    lr_n  = _norm(lr)

    # # Optional daily uncertainty bands
    # daily_npz = tables / "dist_daily.npz"
    # ci = {}
    # if daily_npz.exists():
    #     try:
    #         d = np.load(daily_npz)
    #         def _series_ci(counts_key: str, n_key: str):
    #             if counts_key not in d or n_key not in d:
    #                 return None
    #             C = d[counts_key]      # [D,B]
    #             n = d[n_key]           # [D]
    #             if C.size == 0 or n.size == 0:
    #                 return None
    #             # avoid division by zero
    #             n = np.maximum(n.astype(float), 1.0)
    #             pdf = (C.astype(float).T / n).T  # [D,B]
    #             lo = np.percentile(pdf, 5, axis=0)
    #             hi = np.percentile(pdf, 95, axis=0)
    #             med = np.percentile(pdf, 50, axis=0)
    #             return lo, hi, med
    #         ci["hr"]  = _series_ci("counts_hr",  "n_hr")
    #         ci["gen"] = _series_ci("counts_gen", "n_gen")
    #         if "counts_lr" in d and "n_lr" in d:
    #             ci["lr"] = _series_ci("counts_lr", "n_lr")
    #     except Exception as e:
    #         logger.warning(f"[plot_distributional] Failed to parse dist_daily.npz for CI shading: {e}")

    # # Plot CI bands (shading) before lines
    # val = ci.get("hr")
    # if val is not None:
    #     lo, hi, _ = val
    #     ax.fill_between(bin_centers, np.maximum(lo, eps), np.maximum(hi, eps), color="black", alpha=0.12, linewidth=0)
    # val = ci.get("gen")
    # if val is not None:
    #     lo, hi, _ = val
    #     ax.fill_between(bin_centers, np.maximum(lo, eps), np.maximum(hi, eps), color="royalblue", alpha=0.10, linewidth=0)
    # val = ci.get("lr")
    # if val is not None and lr_n is not None:
    #     lo, hi, _ = val
    #     ax.fill_between(bin_centers, np.maximum(lo, eps), np.maximum(hi, eps), color="deeppink", alpha=0.07, linewidth=0)

    # # Then plot the pooled lines on top

    if hr_n is not None:
        ax.plot(bin_centers, hr_n, color=col_hr, lw=1.5, label="HR")
    if gen_n is not None:
        ax.plot(bin_centers, gen_n, color=col_pmm, lw=1.2, label="PMM")
    if lr_n is not None:
        ax.plot(bin_centers, lr_n, color=col_lr, lw=1.0, ls="--", label="LR")

    # Plot ensemble curve(s)
    if gen_ens_pool is not None:
        s = np.sum(gen_ens_pool)
        if s > 0:
            y = gen_ens_pool / s
            ax.plot(bin_centers, np.maximum(y, eps), lw=1.6, label="GEN (ensemble)", color=col_ens)
    if gen_ens_mean is not None:
        ax.plot(bin_centers, np.maximum(gen_ens_mean, eps), lw=1.6, ls=":", label="GEN (ens mean)", color=col_ens)
    if q10 is not None and q90 is not None:
        ax.fill_between(bin_centers, np.maximum(q10, eps), np.maximum(q90, eps), alpha=0.10, linewidth=0, label="Ens spread (10–90%)")

    ax.set_xlabel("Precipitation (mm/day)")
    ax.set_yscale("log")
    ax.set_ylabel("Probability")
    ax.set_title("Pooled pixel distributions" + (" (ensemble)" if (gen_ens_pool is not None or gen_ens_mean is not None) else ""))

    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()

    # Place GEN vs HR metrics (top-left) and LR vs HR (bottom-right) to avoid overlap
    boxprops = dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85)
    if gen_text:
        ax.text(
            0.02, 0.6, gen_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=boxprops,
        )

    if lr_text:
        ax.text(
            0.02, 0.02, lr_text, transform=ax.transAxes,
            va="bottom", ha="left", fontsize=8,
            bbox=boxprops,
        )
    fig.tight_layout()
    _savefig(fig, figs / "dist_pooled.png", dpi=SET_DPI)