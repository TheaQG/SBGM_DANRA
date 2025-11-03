from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Sequence, Any, List

import numpy as np
import matplotlib.pyplot as plt
import logging

from sbgm.evaluate.evaluate_prcp.plot_utils import _ensure_dir, _savefig, _nice, _to_date_safe, _season_from_month

logger = logging.getLogger(__name__)


# ================================================================================
# 1. PSD curves
# ================================================================================

def plot_scale_psd(scale_root: Path) -> None:
    """
    Read scale_psd_curves.npz and make a single log-log PSD plot:
      - HR: thick, solid
      - GEN: solid
      - LR: full curve, but 'ghosted' (low alpha) for k < lr_nyquist and solid for k >= lr_nyquist
      - x-axis in wavelength lambda (km), log-scaled, inverted (large -> small)
      - HR: thick, black, with +/- 1 sigma shading
      - GEN/PMM: blue, with +/- 1 sigma shading
      - LR: pink
          - solid for λ >= λ_nyq   (i.e. k <= k_nyq)
          - faint/dashed for λ < λ_nyq (i.e. k > k_nyq)
      - vertical line at LR Nyquist
      - optional amplitude alignment for LR so it sits in the right ballpark
        compared to HR on the trusted (k <= k_nyq) range
    """
    tables = scale_root / "tables"
    figs = _ensure_dir(scale_root / "figures")

    npz_path = tables / "scale_psd_curves.npz"
    if not npz_path.exists():
        logger.warning(f"[plot_scale_psd] Did not find {npz_path} – skipping PSD plot.")
        return

    data = np.load(npz_path)
    k = data["k"]               # [K]
    psd_hr = data["psd_hr"]     # [N, K]
    psd_gen = data["psd_gen"]   # [N, K]
    psd_lr = data["psd_lr"]     # [N, K]
    psd_lr_hr = data["psd_lr_hr"]  # [N, K] – LR evaluated on HR grid
    dates = data["dates"]          # [N]
    psd_hr_ci_lo = data.get("psd_hr_ci_lo", None)
    psd_hr_ci_hi = data.get("psd_hr_ci_hi", None)
    psd_gen_ci_lo = data.get("psd_gen_ci_lo", None)
    psd_gen_ci_hi = data.get("psd_gen_ci_hi", None)
    lr_nyquist = float(data.get("lr_nyquist", np.array(0.0)))

    # mean over dates
    eps = 1e-12
    hr_mean = psd_hr.mean(axis=0)
    hr_std = psd_hr.std(axis=0)
    gen_mean = psd_gen.mean(axis=0)
    gen_std = psd_gen.std(axis=0)
    lr_mean = psd_lr.mean(axis=0)

    lr_hr_mean = None
    if psd_lr_hr is not None:
        lr_hr_mean = psd_lr_hr.mean(axis=0)

    hr_mean = np.maximum(hr_mean, eps)
    gen_mean = np.maximum(gen_mean, eps)
    lr_mean = np.maximum(lr_mean, eps)

    if lr_nyquist > 0.0:
        lr_mask_lo = k <= lr_nyquist * 1.0001
        lr_mask_hi = k > lr_nyquist * 1.0001
    else:
        # no LR Nyquist info → just plot as one line
        lr_mask_lo = np.ones_like(k, dtype=bool)
        lr_mask_hi = np.zeros_like(k, dtype=bool)

    # Convert to wavelength (km)
    mask_pos = k > 0.0
    k_pos = k[mask_pos]
    lam = 1.0 / k_pos
    # Sort from large to small wavelength so line is monotonic on x
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    hr_mean = hr_mean[mask_pos][order]
    gen_mean = gen_mean[mask_pos][order]
    hr_std = hr_std[mask_pos][order]
    gen_std = gen_std[mask_pos][order]
    lr_mean = lr_mean[mask_pos][order]
    if lr_hr_mean is not None:
        lr_hr_mean = np.maximum(lr_hr_mean, eps)
        lr_hr_mean = lr_hr_mean[mask_pos][order]

    # --- Compute band powers and ratios from mean PSDs (as plotted) ---
    # 1. k-array in plotted order
    k_plot = k_pos[order]
    # 2. Band definitions (keep same defaults as evaluate_scale.py)
    low_k_max = 1.0 / 200.0  # k <= 5.000e-03 -> lambda >= 200 km
    high_k_min = 1.0 / 20.0  # k >= 5.000e-02 -> lambda <= 20 km
    # 3. Helper for band integration
    def _band_int(k_arr: np.ndarray, p_arr: np.ndarray, kmin: float, kmax: float) -> float:
        m = (k_arr >= kmin) & (k_arr <= kmax)
        if np.any(m):
            return float(np.trapz(p_arr[m], k_arr[m]))

        # Fallback 1: requested band is entirely *below* available k -> use lowest bin
        if kmax < k_arr.min():  # e.g. low-k band but k_arr starts higher
            m = (k_arr <= k_arr.min() * 1.01)
            return float(np.trapz(p_arr[m], k_arr[m]))

        # Fallback 2: requested band is entirely *above* available k -> use highest bin
        if kmin > k_arr.max():  # e.g. very-high-k band
            m = (k_arr >= k_arr.max() * 0.99)
            return float(np.trapz(p_arr[m], k_arr[m]))

        # Final fallback – should rarely happen
        return float(np.trapz(p_arr, k_arr))

    # 4. Compute band powers for HR, GEN, LR-on-HR-grid (or native LR)
    hr_low  = _band_int(k_plot, hr_mean, 0.0, low_k_max)
    hr_high = _band_int(k_plot, hr_mean, high_k_min, k_plot.max())
    gen_low  = _band_int(k_plot, gen_mean, 0.0, low_k_max)
    gen_high = _band_int(k_plot, gen_mean, high_k_min, k_plot.max())
    if lr_hr_mean is not None:
        lr_used = lr_hr_mean
    else:
        lr_used = lr_mean
    lr_low  = _band_int(k_plot, lr_used, 0.0, low_k_max)
    lr_high = _band_int(k_plot, lr_used, high_k_min, k_plot.max())
    # 5. Compute ratios (guard against zero/NaN)
    def _safe_ratio(num: float, den: float) -> float:
        if den is None or not np.isfinite(den) or den <= 0.0:
            return float("nan")
        if num is None or not np.isfinite(num):
            return float("nan")
        return float(num / den)
    gen_hr_low_ratio  = _safe_ratio(gen_low, hr_low)
    gen_hr_high_ratio = _safe_ratio(gen_high, hr_high)
    lr_hr_low_ratio   = _safe_ratio(lr_low, hr_low)
    lr_hr_high_ratio  = _safe_ratio(lr_high, hr_high)
    # 6. Write to CSV
    ratios_path = tables / "scale_psd_band_ratios_avg.csv"
    with open(ratios_path, "w") as f:
        f.write("series,band,power,ratio_to_hr\n")
        f.write(f"HR,low-k,{hr_low:.6e},1.0\n")
        f.write(f"HR,high-k,{hr_high:.6e},1.0\n")
        f.write(f"GEN,low-k,{gen_low:.6e},{gen_hr_low_ratio:.4f}\n")
        f.write(f"GEN,high-k,{gen_high:.6e},{gen_hr_high_ratio:.4f}\n")
        if np.isfinite(lr_low) or np.isfinite(lr_high):
            f.write(f"LR,low-k,{lr_low:.6e},{lr_hr_low_ratio:.4f}\n")
            f.write(f"LR,high-k,{lr_high:.6e},{lr_hr_high_ratio:.4f}\n")


    # --- Find HR–LR intersection in log space ---
    cross_info = None
    if lr_hr_mean is not None:
        lr_for_x = lr_hr_mean.copy()
    else:
        lr_for_x = lr_mean.copy()
    try:
        diff = np.abs(np.log10(hr_mean) - np.log10(lr_for_x))
        ix = int(np.argmin(diff))
        lam_cross = float(lam[ix])
        k_cross = float(k_pos[order][ix])
        cross_info = (lam_cross, k_cross)
    except Exception:
        cross_info = None

    # Nyquist as wavelength
    lam_nyq = None
    if lr_nyquist > 0.0:
        lam_nyq = 1.0 / lr_nyquist

    _nice()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # HR
    ax.plot(lam, hr_mean, color="black", lw=1.6, label="HR (DANRA)")
    if psd_hr_ci_lo is not None and psd_hr_ci_hi is not None:
        ci_lo = np.asarray(psd_hr_ci_lo)[mask_pos][order]
        ci_hi = np.asarray(psd_hr_ci_hi)[mask_pos][order]
        ax.fill_between(lam, np.maximum(ci_lo, eps), np.maximum(ci_hi, eps), # type: ignore
                        color="black", alpha=0.15)
    else:
        ax.fill_between(lam,
                        np.maximum(hr_mean - hr_std, eps),
                        hr_mean + hr_std,
                        color="black", alpha=0.15)

    # GEN / PMM
    ax.plot(lam, gen_mean, color="royalblue", lw=1.4, label="PMM (gen)")
    if psd_gen_ci_lo is not None and psd_gen_ci_hi is not None:
        ci_lo = np.asarray(psd_gen_ci_lo)[mask_pos][order]
        ci_hi = np.asarray(psd_gen_ci_hi)[mask_pos][order]
        ax.fill_between(lam, np.maximum(ci_lo, eps), np.maximum(ci_hi, eps), # type: ignore
                        color="royalblue", alpha=0.12)
    else:
        ax.fill_between(lam,
                        np.maximum(gen_mean - gen_std, eps),
                        gen_mean + gen_std,
                        color="royalblue", alpha=0.12)

    # LR
    if lr_nyquist > 0.0 and lam_nyq is not None:
        # trusted part: λ >= λ_nyq
        trusted = lam >= (lam_nyq * 0.999)
        ghost   = lam <  (lam_nyq * 0.999)

        # "Ghost" part: Use LR-on-HR-grid if we have it, otherwise fall back to native LR (plot all of it, just faint)
        if lr_hr_mean is not None:
            ax.plot(lam, lr_hr_mean,
                    color="deeppink", lw=0.9, linestyle="--", alpha=0.35,
                    label="LR (ERA5, > Nyq, HR grid)")
        else:
            ax.plot(lam, lr_mean,
                    color="deeppink", lw=0.9, linestyle="--", alpha=0.35,
                    label="LR (ERA5, > Nyq)")
        ax.axvline(x=lam_nyq, color="black", lw=0.6, linestyle="--", label="LR Nyq")

        # Solid, trusted LR (native spacing) plotted on top
        if np.any(trusted):
            if lr_hr_mean is not None:
                ax.plot(lam[trusted], lr_hr_mean[trusted],
                        color="deeppink", lw=1.2, label="LR (ERA5 <= Nyq, HR grid)")
            else:
                ax.plot(lam[trusted], lr_mean[trusted],
                        color="deeppink", lw=1.2, label="LR (ERA5 <= Nyq)")

    else:
        # no Nyquist info → single line
        ax.plot(lam, lr_mean, color="deeppink", lw=1.2, label="LR (ERA5)")

    # --- Add HR–LR crossing line and annotation ---
    if cross_info is not None:
        lam_cross, k_cross = cross_info
        ax.axvline(x=lam_cross, color="magenta", lw=0.7, ls=":", label="HR-LR intersextion")
        y_max = ax.get_ylim()[1]
        ax.text(
            lam_cross,
            y_max * 0.45,
            f"k={k_cross:.3e}\nλ={lam_cross:.0f} km",
            color="magenta",
            ha="right",
            va="center",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.55),
            rotation=90,
        )

    # --- mark low-k and high-k limits ---
    ax.axvline(1.0 / low_k_max, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    # move slightly to the left to avoid overlap with high-k line
    x1 = 1.0 / low_k_max * 1.08
    # Place y1 at 20% of the current y-axis limit (remember it's log-scaled!)
    y1 = np.log10(ax.get_ylim()[1] * 0.0001)
    ax.text(x1, y1, f"low-k λ={1.0/low_k_max:.0f} km",
        rotation=90, color="gray", fontsize=6.5, ha="center", va="bottom",)

    ax.axvline(1.0 / high_k_min, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,)
    x2 = 1.0 / high_k_min * 1.08
    y2 = np.log10(ax.get_ylim()[1] * 0.0001)
    ax.text(x2, y2, f"high-k λ={1.0/high_k_min:.0f} km",
        rotation=90, color="gray", fontsize=6.5, ha="center", va="bottom",)


    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Wavelength λ (km)")
    ax.set_ylabel("Spectral power")
    ax.set_title("Isotropic Power Spectral Density (PSD)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8) # a bit smaller text

    # --- annotate band ratios ---
    lines = [
        f"low\u2011k (k ≤ {low_k_max:.3e}, λ ≥ {1.0/low_k_max:.0f} km)",
        f"  GEN / HR = {gen_hr_low_ratio:.2f}",
    ]
    if np.isfinite(lr_hr_low_ratio):
        lines.append(f"  LR  / HR = {lr_hr_low_ratio:.2f}")
    lines += [
        "",  # blank line
        f"high\u2011k (k ≥ {high_k_min:.3e}, λ ≤ {1.0/high_k_min:.0f} km)",
        f"  GEN / HR = {gen_hr_high_ratio:.2f}",
    ]
    if np.isfinite(lr_hr_high_ratio):
        lines.append(f"  LR  / HR = {lr_hr_high_ratio:.2f}")

    ax.text(
        0.01,
        0.01,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7),
    )

    fig.tight_layout()
    fig.savefig(str(figs / "scale_psd.png"), dpi=200)
    plt.close(fig)





# ================================================================================
# 2. PSD low/high band ratio diag plot
# ================================================================================

# in sbgm/evaluate/evaluate_prcp/eval_scale/plot_scale.py

def plot_psd_lowhigh_diag(scale_root: Path) -> None:
    """
    Summarize PSD low/high band ratios across *all* days.

    We read scale_psd_summary.csv (written by evaluate_scale.py) and build boxplots for:
      - GEN / HR (low-k)
      - GEN / LR (low-k)  [only if present]
      - GEN / HR (high-k)

    This is much more stable than plotting vs date index, because a single day with
    ~zero HR high-k power can blow up the ratio.
    """
    tables = scale_root / "tables"
    figs = _ensure_dir(scale_root / "figures")
    csv_path = tables / "scale_psd_summary.csv"
    if not csv_path.exists():
        logger.warning(f"[plot_psd_lowhigh_diag] Did not find {csv_path} – skipping.")
        return

    # read rows (no pandas)
    gen_low_hr: list[float] = []
    gen_low_lr: list[float] = []
    gen_high_hr: list[float] = []

    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        # expected (from evaluate_scale.py):
        # date,hr_lowk,gen_lowk,gen_lowk_vs_hr,gen_lowk_vs_lr,hr_highk,gen_highk,gen_highk_vs_hr
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue

            def _to_float(s: str) -> float | None:
                s = s.strip()
                if s == "":
                    return None
                try:
                    return float(s)
                except Exception:
                    return None

            hr_lowk = _to_float(parts[1])
            gen_lowk = _to_float(parts[2])
            hr_highk = _to_float(parts[5])
            gen_highk = _to_float(parts[6])
            gl_hr_csv = _to_float(parts[3])   # gen_lowk_vs_hr
            gl_lr_csv = _to_float(parts[4])   # gen_lowk_vs_lr (can be empty)
            gh_hr_csv = _to_float(parts[7])   # gen_highk_vs_hr

            # New logic: recompute ratios from powers for low/high bands
            def _ok_low(x):
                return (x is not None) and np.isfinite(x) and (x > 0.0) and (x < 5.0)
            def _ok_high(x):
                return (x is not None) and np.isfinite(x) and (x > 0.0)

            # GEN/HR low-k ratio: always try to fill if powers are finite and HR > 1e-7
            gl_hr = None
            if (hr_lowk is not None and gen_lowk is not None
                    and np.isfinite(hr_lowk) and np.isfinite(gen_lowk)
                    and hr_lowk > 1e-7 and gen_lowk > 0.0):
                gl_hr = float(gen_lowk / hr_lowk)
            # If not, fallback to CSV column (legacy), but only if valid
            if gl_hr is None and gl_hr_csv is not None and np.isfinite(gl_hr_csv) and gl_hr_csv > 0.0:
                gl_hr = float(gl_hr_csv)
            if _ok_low(gl_hr):
                assert gl_hr is not None
                gen_low_hr.append(float(gl_hr))

            gl_lr = None
            if len(parts) > 4:
                gl_lr = gl_lr_csv
            if _ok_low(gl_lr):
                assert gl_lr is not None
                gen_low_lr.append(float(gl_lr))
                gen_low_lr.append(float(gl_lr))
            gh_hr = None
            if (hr_highk is not None and gen_highk is not None
                    and np.isfinite(hr_highk) and np.isfinite(gen_highk)
                    and hr_highk > 1e-8 and gen_highk > 0.0):
                gh_hr = float(gen_highk / hr_highk)
            # fallback to CSV column if not available
            if gh_hr is None and gh_hr_csv is not None and np.isfinite(gh_hr_csv) and gh_hr_csv > 0.0:
                gh_hr = float(gh_hr_csv)
            if _ok_high(gh_hr) and hr_highk is not None and hr_highk > 1e-8:
                assert gh_hr is not None
                gen_high_hr.append(float(gh_hr))

    if not gen_low_hr and not gen_low_lr and not gen_high_hr:
        logger.warning("[plot_psd_lowhigh_diag] No valid ratios found – skipping plot.")
        return

    # build the boxplot data
    labels = []
    data = []

    if gen_low_hr:
        labels.append("GEN / HR (low-k)")
        data.append(gen_low_hr)
    if gen_low_lr:
        labels.append("GEN / LR (low-k)")
        data.append(gen_low_lr)
    if gen_high_hr:
        labels.append("GEN / HR (high-k)")
        data.append(gen_high_hr)

    _nice()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bp = ax.boxplot(
        data,
        vert=False,
        labels=labels,
        showfliers=True,
        patch_artist=True,
    )

    # light fill like you used for PSD plot
    for patch in bp["boxes"]:
        p: Any = patch
        try:
            p.set_facecolor("0.9")
        except Exception:
            # Some backends/types may return Line2D objects without set_facecolor;
            # try a more generic set_color fallback and ignore failures.
            try:
                p.set_color("0.9")
            except Exception:
                pass

    # annotate means
    for y, arr in enumerate(data, start=1):
        arr_np = np.asarray(arr, dtype=float)
        mean_val = float(arr_np.mean())
        ax.text(
            mean_val,
            y,
            f"  μ={mean_val:.2f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    # --- set informative x-limits and note if clipped ---
    xmax = max([max(a) for a in data]) if data else 2.0
    if xmax > 5.0:
        ax.set_xlim(0, 5.0)
        ax.text(0.99, 0.02, "(values >5 clipped)", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, alpha=0.5)
    else:
        ax.set_xlim(0, max(2.0, xmax * 1.05))

    # --- subtitle with bands in k and λ ---
    low_k_max = 1.0 / 200.0  # cycles/km → λ = 200 km
    high_k_min = 1.0 / 20.0  # cycles/km → λ = 20 km

    ax.set_xlabel("Ratio")
    ax.set_title("PSD band ratios (all days)")
    ax.text(
        0.01,
        1.02,
        f"low-k: k ≤ {low_k_max:.3e} (λ ≥ {1.0/low_k_max:.0f} km)\n"
        f"high-k: k ≥ {high_k_min:.3e} (λ ≤ {1.0/high_k_min:.0f} km)",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    ax.grid(True, axis="x", ls=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(str(figs / "scale_psd_lowhigh.png"), dpi=200)
    plt.close(fig)





# ================================================================================
# 3. FSS vs scale curves
# ================================================================================

def plot_fss_curves(scale_root: Path) -> None:
    """
    Read FSS outputs and make **one** multi-panel figure:
      - 1 subplot per base threshold (gen + LR if present)
      - 1 final subplot with all thresholds together
    Layout: 2 rows x 3 columns (covers up to 5 thresholds + 1 overview).
    If there are more than 5 thresholds, extra ones are added to the last row.
    
    Primary source:  <scale_root>/tables/scale_fss_summary.csv
    Optional source: <scale_root>/tables/scale_fss_daily.csv  (used ONLY to recover
    LR baselines, because the summary file currently does not contain LR rows.)

    """
    tables = scale_root / "tables"
    figs = _ensure_dir(scale_root / "figures")
    summary_path = tables / "scale_fss_summary.csv"
    if not summary_path.exists():
        logger.warning(f"[plot_fss_curves] Did not find {summary_path} – skipping FSS plot.")
        return

    # ------------------------------------------------------------
    # 1) Read summary (always available) – gives us GEN per threshold
    # ------------------------------------------------------------
    with open(summary_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        logger.warning("[plot_fss_curves] scale_fss_summary.csv is empty.")
        return

    header = lines[0].split(",")
    rows = [l.split(",") for l in lines[1:]]

    # find FSS columns
    fss_cols = [(i, col) for i, col in enumerate(header) if col.lower().startswith("fss_")]
    if not fss_cols:
        logger.warning("[plot_fss_curves] No FSS_* columns found.")
        return

    # by_thr["1.00"] = {"gen": [(5,0.5),...], "lr": [(5,0.4), ...]}
    by_thr: dict[str, dict[str, list[tuple[float, float]]]] = {}

    for r in rows:
        base_thr = r[0].strip()  # e.g. "1.00"
        if base_thr == "":
            continue
        
        if base_thr not in by_thr:
            by_thr[base_thr] = {"gen": [], "lr": []}

        for idx, col in fss_cols:
            # col looks like "fss_5km"
            try:
                scale_km = float(col.split("_")[1].replace("km", ""))
            except Exception:
                continue
            val = r[idx].strip()
            if val == "":
                continue
            by_thr[base_thr]["gen"].append((scale_km, float(val)))

    # ------------------------------------------------------------
    # 2) Try to recover LR baselines from the *daily* CSV
    #    (evaluate_scale.py currently only writes LR lines there)
    # ------------------------------------------------------------
    daily_path = tables / "scale_fss_daily.csv"
    if daily_path.exists():
        with open(daily_path, "r") as f:
            d_lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(d_lines) > 1:
            d_header = d_lines[0].split(",")  # ["date", "thr_mm", "fss_5km", ...]
            d_rows = [l.split(",") for l in d_lines[1:]]
            d_fss_cols = [(i, col) for i, col in enumerate(d_header) if col.lower().startswith("fss_")]
            # tmp: (base_thr -> scale_km -> list of values)
            lr_acc: dict[str, dict[float, list[float]]] = {}
            for r in d_rows:
                thr_str = r[1].strip()  # e.g. "1.00" or "1.00_LR"
                if not thr_str.endswith("_LR"):
                    continue  # only interested in LR here
                base_thr = thr_str.replace("_LR", "")
                for idx, col in d_fss_cols:
                    try:
                        scale_km = float(col.split("_")[1].replace("km", ""))
                    except Exception:
                        continue
                    val = r[idx].strip()
                    if val == "":
                        continue
                    v = float(val)
                    lr_acc.setdefault(base_thr, {}).setdefault(scale_km, []).append(v)

            # turn accumulators into means, append to by_thr
            for base_thr, per_scale in lr_acc.items():
                if base_thr not in by_thr:
                    by_thr[base_thr] = {"gen": [], "lr": []}
                for scale_km, vals in per_scale.items():
                    m = float(np.mean(vals))
                    by_thr[base_thr]["lr"].append((scale_km, m))
    else:
        logger.debug("[plot_fss_curves] No scale_fss_daily.csv – LR baselines will not be shown.")

    # ------------------------------------------------------------
    # 3) Build figure with subplots
    # ------------------------------------------------------------
    # Sort thresholds numerically if possible
    def _thr_key(x: str) -> float:
        try:
            return float(x)
        except Exception:
            return 9e9

    thr_list = sorted(by_thr.keys(), key=_thr_key)
    n_thr = len(thr_list)
    n_panels = n_thr + 1  # last panel = all thresholds

    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    _nice()
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols * 0.6, 4 * nrows * 0.65), sharex=True, sharey=True)
    axs = np.atleast_2d(axs)

    # color cycle for per-threshold lines (will re-use in the last panel)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])

    # -- 3a) individual panels --
    for i, thr in enumerate(thr_list):
        row = i // ncols
        col = i % ncols
        ax = axs[row, col]
        data = by_thr[thr]
        gen_pairs = sorted(data["gen"], key=lambda t: t[0])
        lr_pairs = sorted(data["lr"], key=lambda t: t[0]) if data["lr"] else []

        x_gen = [p[0] for p in gen_pairs]
        y_gen = [p[1] for p in gen_pairs]
        ax.plot(x_gen, y_gen, marker="o", linewidth=1.4, color=colors[i % len(colors)],
                label=f"gen (≥ {float(thr):.0f} mm)")

        if lr_pairs:
            x_lr = [p[0] for p in lr_pairs]
            y_lr = [p[1] for p in lr_pairs]
            ax.plot(x_lr, y_lr, marker="x", linestyle="--", linewidth=1.0,
                    color="0.25", label=f"LR (≥ {float(thr):.0f} mm)")

        # Only set labels on leftmost and bottom plots
        if col == 0:
            ax.set_ylabel("FSS")
        if row == nrows - 1:
            ax.set_xlabel("Neighborhood scale (km)")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"FSS vs scale. Thr ≥ {float(thr):.0f} mm/day")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, ls=":", alpha=0.4)

    # -- 3b) combined panel (all thresholds) --
    last_idx = n_panels - 1
    row = last_idx // ncols
    col = last_idx % ncols
    ax_all = axs[row, col]
    for i, thr in enumerate(thr_list):
        data = by_thr[thr]
        gen_pairs = sorted(data["gen"], key=lambda t: t[0])
        if not gen_pairs:
            continue
        x_gen = [p[0] for p in gen_pairs]
        y_gen = [p[1] for p in gen_pairs]
        colr = colors[i % len(colors)]
        line, = ax_all.plot(x_gen, y_gen, marker="o", linewidth=1.2, color=colr)
        # annotate with tiny text near last point
        ax_all.text(x_gen[-1] * 1.01, y_gen[-1], f"≥ {float(thr):.0f} mm",
                    color=colr, fontsize=7, va="center")

        # if LR exists for this thr, plot as faint grey
        lr_pairs = sorted(data["lr"], key=lambda t: t[0]) if data["lr"] else []
        if lr_pairs:
            x_lr = [p[0] for p in lr_pairs]
            y_lr = [p[1] for p in lr_pairs]
            ax_all.plot(x_lr, y_lr, marker="x", linestyle="--", linewidth=0.8,
                        color="0.5", alpha=0.7)

    ax_all.set_xlabel("Neighborhood scale (km)")
    ax_all.set_ylabel("FSS")
    ax_all.set_ylim(0.0, 1.0)
    ax_all.set_title("FSS vs scale (all thresholds)")
    ax_all.grid(True, ls=":", alpha=0.4)

    # turn off any unused subplots (if n_panels < nrows*ncols)
    for j in range(n_panels, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axs[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(str(figs / "scale_fss_multi.png"), dpi=200)
    plt.close(fig)




# ================================================================================
# 4. ISS at scales
# ================================================================================

def plot_iss_curves(scale_root: Path) -> None:
    """
    ISS vs scale, same layout as FSS:
      - 1 panel per threshold (GEN + LR if present)
      - 1 final panel with all thresholds together
    Reads:
      - <scale_root>/tables/scale_iss_summary.csv  (always)
      - <scale_root>/tables/scale_iss_daily.csv    (to recover LR baselines)
    """
    tables = scale_root / "tables"
    figs = _ensure_dir(scale_root / "figures")
    summary_path = tables / "scale_iss_summary.csv"
    if not summary_path.exists():
        logger.warning(f"[plot_iss_curves] Did not find {summary_path} – skipping ISS plot.")
        return

    with open(summary_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        logger.warning("[plot_iss_curves] scale_iss_summary.csv is empty – skipping.")
        return

    header = lines[0].split(",")
    rows = [l.split(",") for l in lines[1:]]

    iss_cols = [(i, col) for i, col in enumerate(header) if col.lower().startswith("iss_")]
    if not iss_cols:
        logger.warning("[plot_iss_curves] No ISS_* columns found – skipping.")
        return

    # by_thr["1.00"] = {"gen": [(5,0.7), ...], "lr": [(5,0.8), ...]}
    by_thr: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for r in rows:
        thr_id = r[0].strip()
        if not thr_id:
            continue
        by_thr.setdefault(thr_id, {"gen": [], "lr": []})
        for idx, col in iss_cols:
            try:
                scale_km = float(col.split("_")[1].replace("km", ""))
            except Exception:
                continue
            val = r[idx].strip()
            if val == "":
                continue
            by_thr[thr_id]["gen"].append((scale_km, float(val)))

    # try to get LR baselines from daily CSV
    daily_path = tables / "scale_iss_daily.csv"
    if daily_path.exists():
        with open(daily_path, "r") as f:
            d_lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(d_lines) > 1:
            d_header = d_lines[0].split(",")
            d_rows = [l.split(",") for l in d_lines[1:]]
            d_iss_cols = [(i, col) for i, col in enumerate(d_header) if col.lower().startswith("iss_")]
            lr_acc: dict[str, dict[float, list[float]]] = {}
            for r in d_rows:
                thr_raw = r[1].strip()  # e.g. "1.00" or "1.00_LR"
                is_lr = thr_raw.endswith("_LR")
                base_thr = thr_raw.replace("_LR", "")
                if base_thr not in by_thr:
                    continue
                for idx, col in d_iss_cols:
                    try:
                        scale_km = float(col.split("_")[1].replace("km", ""))
                    except Exception:
                        continue
                    v = r[idx].strip()
                    if v == "":
                        continue
                    if is_lr:
                        lr_acc.setdefault(base_thr, {}).setdefault(scale_km, []).append(float(v))
            # push averaged LR back into by_thr
            for thr, scales in lr_acc.items():
                for scale_km, arr in scales.items():
                    mean_lr = float(np.mean(arr))
                    by_thr[thr]["lr"].append((scale_km, mean_lr))

    thrs_sorted = sorted(by_thr.keys(), key=lambda s: float(s))
    n_thr = len(thrs_sorted)
    n_panels = n_thr + 1  # overview
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))

    _nice()
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols * 0.6, 4 * nrows * 0.65), sharex=True, sharey=True)
    axs = np.atleast_2d(axs)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])

    for i, thr in enumerate(thrs_sorted):
        row = i // ncols
        col = i % ncols
        # leave the very last cell for the overview
        ax = axs[row, col]
        gen_pts = sorted(by_thr[thr]["gen"], key=lambda p: p[0])
        x = [p[0] for p in gen_pts]
        y = [p[1] for p in gen_pts]
        colr = colors[i % len(colors)]
        ax.plot(x, y, marker="o", linewidth=1.4, color=colr, label=f"gen (≥ {float(thr):.0f} mm)")

        lr_pts = sorted(by_thr[thr]["lr"], key=lambda p: p[0]) if by_thr[thr]["lr"] else []
        if lr_pts:
            ax.plot([p[0] for p in lr_pts], [p[1] for p in lr_pts],
                    linestyle="--", marker="x", linewidth=1.0, color="0.35",
                    label=f"LR (≥ {float(thr):.2f} mm)")

        ax.set_ylim(0.0, 1.05)
        
        # Only plot xlabel on bottom row
        if row == nrows - 1:
            ax.set_xlabel("Neighborhood scale (km)")
        # only plot ylabel on leftmost column
        if col == 0:
            ax.set_ylabel(f"ISS vs scale.")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=9, loc="lower right")

    # overview panel in the last slot
    last_idx = n_panels - 1
    ov_row = last_idx // ncols
    ov_col = last_idx % ncols
    ax_all = axs[ov_row, ov_col]
    for i, thr in enumerate(thrs_sorted):
        gen_pts = sorted(by_thr[thr]["gen"], key=lambda p: p[0])
        if not gen_pts:
            continue
        x = [p[0] for p in gen_pts]
        y = [p[1] for p in gen_pts]
        colr = colors[i % len(colors)]
        ax_all.plot(x, y, marker="o", linewidth=1.2, color=colr)
        ax_all.text(x[-1] * 1.01, y[-1], f"≥ {float(thr):.0f} mm", color=colr, fontsize=4, va="center")

        lr_pts = sorted(by_thr[thr]["lr"], key=lambda p: p[0]) if by_thr[thr]["lr"] else []
        if lr_pts:
            ax_all.plot([p[0] for p in lr_pts], [p[1] for p in lr_pts],
                        linestyle="--", linewidth=0.8, color="0.5", alpha=0.7)

    ax_all.set_ylim(0.0, 1.05)
    ax_all.set_xlabel("Neighborhood scale (km)")
    ax_all.set_ylabel("ISS")
    ax_all.set_title("ISS vs scale (all thresholds)")
    ax_all.grid(True, ls=":", alpha=0.5)

    # turn off empty axes (if any)
    for j in range(n_panels, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axs[r, c].axis("off")

    fig.tight_layout()
    _savefig(fig, figs / "scale_iss_curves.png")



# ================================================================================
# Master entry point
# ================================================================================

def plot_scale(eval_root: str | Path, baseline_eval_dirs: Optional[Dict[str, str]] = None) -> None:
    """
    Master entry point – call this from evaluate_scale.py

    baseline_eval_dirs is kept here for symmetry with your old plotting module,
    but we don’t actually use it yet (easy to add later).
    """
    scale_root = Path(eval_root)
    if not scale_root.exists():
        logger.warning(f"[plot_scale] {scale_root} does not exist.")
        return

    plot_scale_psd(scale_root)
    plot_fss_curves(scale_root)
    plot_iss_curves(scale_root)
    plot_psd_lowhigh_diag(scale_root)
    logger.info(f"[plot_scale] Plots written to {scale_root / 'figures'}")