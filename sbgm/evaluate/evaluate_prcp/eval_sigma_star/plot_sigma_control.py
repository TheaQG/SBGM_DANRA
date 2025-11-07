"""
Plot sigma*-dependent evaluation metrics: correlation, PSD slope, and CRPS.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Additional imports for color and data utilities
from sbgm.variable_utils import get_color_for_model, get_cmap_for_variable
from sbgm.evaluate.data_resolver import EvalDataResolver

# New imports for plotting utilities and DK outline
from sbgm.evaluate.evaluate_prcp.plot_utils import _ensure_dir, _nice, _savefig, get_dk_lsm_outline, overlay_outline
from sbgm.plotting_utils import _add_colorbar_and_boxplot

SET_DPI = 300

def plot_sigma_control(summary_csv, figures_dir, combined: bool = False):
    """
    Render sigma*-dependent summary plots.

    Parameters
    ----------
    summary_csv : str or Path
        Path to tables/agg_summary.csv
    figures_dir : str or Path
        Directory to write figures (created if missing).
    combined : bool
        If True, draw all three panels side-by-side in a single figure and also
        save individual panels. If False, save only individual panels.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = np.genfromtxt(summary_csv, delimiter=',', names=True, dtype=None, encoding='utf-8')

    # Handle empty summary gracefully
    if isinstance(data, np.ndarray) and data.size == 0:
        return {}

    sigma = np.asarray(data['sigma_star'], dtype=float)
    r_lp_mean = np.asarray(data['r_lp_mean'], dtype=float)
    r_lp_std = np.asarray(data['r_lp_std'], dtype=float)
    slope_gen_mean = np.asarray(data['slope_gen_mean'], dtype=float)
    slope_gen_std = np.asarray(data['slope_gen_std'], dtype=float)
    slope_hr_mean = np.asarray(data['slope_hr_mean'], dtype=float)
    slope_hr_std = np.asarray(data['slope_hr_std'], dtype=float)
    crps_mean = np.asarray(data['crps_mean'], dtype=float)
    crps_std = np.asarray(data['crps_std'], dtype=float)

    # Optional: high-k gain (may be absent in old runs)
    hk_gain_mean = np.asarray(data['hk_gain_mean'], dtype=float) if (data.dtype.names is not None and 'hk_gain_mean' in data.dtype.names) else None
    hk_gain_std  = np.asarray(data['hk_gain_std'], dtype=float)  if (data.dtype.names is not None and 'hk_gain_std' in data.dtype.names) else None

    # Colors and styling
    color_gen = get_color_for_model("pmm")
    color_hr = get_color_for_model("hr")
    color_ens = get_color_for_model("ens")

    # Style
    _nice()
    marker_style = dict(marker="o", markersize=5, lw=1.8, capsize=3)

    # Utility for x padding
    def _xpad(x):
        if x.size == 0:
            return (0, 1)
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        pad = 0.03 * max(1e-6, hi - lo)
        return lo - pad, hi + pad

    figpaths = {}

    # --- Combined figure (optional) ---
    if combined:
        import matplotlib.pyplot as plt
        ncols = 4 if hk_gain_mean is not None else 3
        fig, axes = plt.subplots(1, ncols, figsize=(4.6*ncols, 4.5), sharex=False)
        # 1) Correlation
        ax = axes[0]
        ax.errorbar(sigma, r_lp_mean, yerr=r_lp_std, color=color_gen, **marker_style)
        ax.set_xlabel(r"$\sigma^*$")
        ax.set_ylabel("LR-GEN correlation\n(LP ≤ LR Nyquist)")
        ax.set_xlim(*_xpad(sigma))
        ax.set_ylim(0.0, min(1.0, max(0.02, float(np.nanmax(r_lp_mean + r_lp_std)) + 0.02)))
        ax.set_title("Scale-aware correlation")
        # 2) PSD slope
        ax = axes[1]
        ax.errorbar(sigma, slope_gen_mean, yerr=slope_gen_std, color=color_gen, label="Generated", **marker_style)
        if slope_hr_mean.size > 0 and np.isfinite(slope_hr_mean).any():
            hr_ref = float(np.nanmean(slope_hr_mean))
            ax.axhline(hr_ref, color=color_hr, ls="--", lw=1.5, label="DANRA")
        ax.set_xlabel(r"$\sigma^*$")
        ax.set_ylabel("PSD slope (5-20 km)")
        ax.set_xlim(*_xpad(sigma))
        ax.legend(frameon=False)
        ax.set_title("Mesoscale PSD slope")
        # 3) CRPS
        ax = axes[2]
        ax.errorbar(sigma, crps_mean, yerr=crps_std, color=color_ens, **marker_style)
        ax.set_xlabel(r"$\sigma^*$")
        ax.set_ylabel("CRPS")
        ax.set_xlim(*_xpad(sigma))
        ax.set_title(r"Probabilistic skill vs $\sigma^*$")
        # 4) High-k gain panel if available
        if hk_gain_mean is not None:
            ax = axes[3]
            ax.errorbar(sigma, hk_gain_mean, yerr=hk_gain_std, color=color_gen, **marker_style)
            ax.set_xlabel(r"$\sigma^*$")
            ax.set_ylabel(r"$G_\mathrm{high}$  (P_GEN / P_HR, k > k_\mathrm{Nyq}^{LR})")
            ax.set_xlim(*_xpad(sigma))
            ax.set_title("High‑k power gain")
        fig.tight_layout()
        out_all = figures_dir / "sigma_control_overview.png"
        _savefig(fig, out_all, dpi=300)
        figpaths["overview"] = str(out_all)

    # --- Individual panels ---
    import matplotlib.pyplot as plt
    # 1) Correlation
    fig = plt.figure()
    ax = plt.gca()
    ax.errorbar(sigma, r_lp_mean, yerr=r_lp_std, color=color_gen, **marker_style)
    ax.set_xlabel(r"$\sigma^*$")
    ax.set_ylabel("LR-GEN correlation (LP ≤ LR Nyquist)")
    ax.set_xlim(*_xpad(sigma))
    ax.set_ylim(0.0, min(1.0, max(0.02, float(np.nanmax(r_lp_mean + r_lp_std)) + 0.02)))
    ax.set_title(r"Scale-aware correlation vs $\sigma^*$")
    figpaths["corr"] = str(figures_dir / "hr_lr_corr_vs_sigma.png")
    _savefig(fig, Path(figpaths["corr"]), dpi=SET_DPI)

    # 2) PSD slope
    fig = plt.figure()
    ax = plt.gca()
    ax.errorbar(sigma, slope_gen_mean, yerr=slope_gen_std, color=color_gen, label="Generated", **marker_style)
    if slope_hr_mean.size > 0 and np.isfinite(slope_hr_mean).any():
        hr_ref = float(np.nanmean(slope_hr_mean))
        ax.axhline(hr_ref, color=color_hr, ls="--", lw=1.5, label="DANRA")
    ax.set_xlabel(r"$\sigma^*$")
    ax.set_ylabel("PSD slope (5-20 km)")
    ax.set_xlim(*_xpad(sigma))
    ax.set_title(r"PSD slope vs $\sigma^*$")
    ax.legend(frameon=False)
    figpaths["slope"] = str(figures_dir / "psd_slope_vs_sigma.png")
    _savefig(fig, Path(figpaths["slope"]), dpi=SET_DPI)

    # 3) CRPS
    fig = plt.figure()
    ax = plt.gca()
    ax.errorbar(sigma, crps_mean, yerr=crps_std, color=color_ens, **marker_style)
    ax.set_xlabel(r"$\sigma^*$")
    ax.set_ylabel("CRPS")
    ax.set_xlim(*_xpad(sigma))
    ax.set_title(r"Probabilistic skill vs $\sigma^*$")
    figpaths["crps"] = str(figures_dir / "crps_vs_sigma.png")
    _savefig(fig, Path(figpaths["crps"]), dpi=SET_DPI)

    # 4) High-k gain (if present)
    if hk_gain_mean is not None:
        fig = plt.figure()
        ax = plt.gca()
        ax.errorbar(sigma, hk_gain_mean, yerr=hk_gain_std, color=color_gen, **marker_style)
        ax.set_xlabel(r"$\sigma^*$")
        ax.set_ylabel(r"$G_\mathrm{high}$  (P_GEN / P_HR, k > k_\mathrm{Nyq}^{LR})")
        ax.set_xlim(*_xpad(sigma))
        ax.axhline(1.0, color="0.3", ls="--", lw=1.2)  # reference where GEN matches HR power
        ax.set_title(r"High‑k power gain vs $\sigma^*$")
        figpaths["hk_gain"] = str(figures_dir / "high_k_gain_vs_sigma.png")
        _savefig(fig, Path(figpaths["hk_gain"]), dpi=SET_DPI)

    return figpaths




# --------------------------------------------------------------------------
# Qualitative example montage function for sigma* control
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Helper functions for squeezing and masking arrays
# --------------------------------------------------------------------------
def _squeeze2d_arr(a):
    import numpy as _np
    import torch as _torch
    if a is None:
        return None
    if isinstance(a, _torch.Tensor):
        a = a.detach().cpu().numpy()
    x = _np.asarray(a)
    while x.ndim > 2 and 1 in x.shape:
        x = _np.squeeze(x)
    if x.ndim == 3:
        if x.shape[0] <= 8:
            x = x[0]
        else:
            x = x[..., 0]
    return x if x.ndim == 2 else None

def _apply_mask_arr(x, mask):
    import numpy as _np
    if x is None:
        return None
    if mask is None:
        return x
    out = x.copy()
    m = mask
    if m.shape != out.shape:
        m = _np.broadcast_to(m, out.shape)
    out[~m] = _np.nan
    return out

# --------------------------------------------------------------------------
# New function: plot_sigma_control_examples_grid
# --------------------------------------------------------------------------
def plot_sigma_control_examples_grid(
    cfg,
    sigma_star_grid,
    gen_base_dir,
    out_dir,
    *,
    n_members=3,
    date: str | None = None,
    land_only: bool = True,
    percentile: float = 99.5,
    fname: str = "examples_sigma_grid.png",
):
    gen_base_dir = Path(gen_base_dir)
    figs_root = Path(out_dir) / "figures" / "examples"
    _ensure_dir(figs_root)

    cmap = get_cmap_for_variable("prcp")
    dk_outline = get_dk_lsm_outline()
    if dk_outline is not None:
        dk_outline = np.flipud(dk_outline)

    resolvers = []
    for sstar in sigma_star_grid:
        subdir = gen_base_dir / f"sigma_star={float(sstar):.2f}"
        if not subdir.exists():
            continue
        resolvers.append((sstar, EvalDataResolver(gen_root=subdir, eval_land_only=land_only, lr_phys_key="lr")))
    if not resolvers:
        return []

    if date is None:
        common = None
        for _, R in resolvers:
            ds = set(R.list_dates())
            common = ds if common is None else (common & ds)
        if common and len(common) > 0:
            date = sorted(list(common))[0]
        else:
            date = resolvers[0][1].list_dates()[0]

    rows = []
    max_members = 0
    for sstar, R in resolvers:
        mask = None
        try:
            if land_only:
                m = _squeeze2d_arr(R.load_mask(date))
                if m is not None:
                    mask = (m > 0.5)
        except Exception:
            mask = None

        hr = _apply_mask_arr(_squeeze2d_arr(R.load_obs(date)), mask)
        lr = _apply_mask_arr(_squeeze2d_arr(R.load_lr(date)), mask)
        pmm = _apply_mask_arr(_squeeze2d_arr(R.load_pmm(date)), mask)
        ens = R.load_ens(date)

        members = []
        if ens is not None:
            A = ens.detach().cpu().numpy() if hasattr(ens, "detach") else np.asarray(ens)
            if A.ndim == 4 and A.shape[1] == 1:
                A = A[:,0]
            if A.ndim == 3 and A.shape[0] > 0:
                m = min(n_members, A.shape[0])
                for i in range(m):
                    members.append(_apply_mask_arr(_squeeze2d_arr(A[i]), mask))
        max_members = max(max_members, len(members))
        
        rows.append({"sigma": sstar, "hr": hr, "lr": lr, "pmm": pmm, "members": members})

    Rn = len(rows)
    Cn = 2 + max_members + 1        

    _nice()
    fig_w = 3.2 * Cn
    fig_h = 3.2 * Rn
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(Rn, Cn, figsize=(fig_w, fig_h))
    if Rn == 1: axs = axs[np.newaxis, :]
    if Cn == 1: axs = axs[:, np.newaxis]

    titles = ["HR (DANRA)", "LR (ERA5)"] + [f"Ens-{i+1}" for i in range(max_members)] + ["PMM (gen)"]

    for r, row in enumerate(rows):
        hr = row["hr"]; lr = row["lr"]; pmm = row["pmm"]; members = row["members"]
        pool = [x for x in [hr, lr, pmm, *members] if x is not None]
        if pool:
            flat = np.concatenate([v[np.isfinite(v)] for v in pool]) if all([v is not None for v in pool]) else np.array([0.0])
            vmin = 0.0
            vmax = float(np.nanpercentile(flat, percentile)) if flat.size > 0 else 1.0
            vmax = max(vmin + 1e-6, vmax)
        else:
            vmin, vmax = 0.0, 1.0

        c = 0
        def draw(ax, img, title):
            if img is None:
                ax.axis("off"); return
            im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
            overlay_outline(ax, dk_outline)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
            _add_colorbar_and_boxplot(fig, ax, im, img, boxplot=True, ylim=(vmin, vmax))

        draw(axs[r, c], hr, titles[c]); c += 1
        draw(axs[r, c], lr, titles[c]); c += 1
        for j in range(max_members):
            ax = axs[r, c + j]
            if j < len(members):
                draw(ax, members[j], titles[c + j])
            else:
                ax.axis("off")
        c += max_members
        draw(axs[r, c], pmm, titles[-1])
        axs[r, 0].set_ylabel(f"σ*={row['sigma']:.2f}\n{date}", fontsize=9)

    fig.text(0.5, 0.02, "Precipitation [mm/day]", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    out_path = figs_root / fname
    _savefig(fig, out_path, dpi=300)
    return [str(out_path)]

def plot_sigma_control_psd_curves(out_dir: str | Path) -> str | None:
    """
    Read <out_dir>/tables/sigma_psd_curves.npz and make a PSD vs wavelength plot
    showing HR, LR, and one curve per sigma* (mean +/- std shading), with:
      - green color scale across sigma*
      - shaded band for the slope window (e.g., 5-20 km)
      - textbox listing slopes per sigma* computed from the mean GEN PSD
    Saves to <out_dir>/figures/sigma_psd_curves.png
    """
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    npz_path = tables / "sigma_psd_curves.npz"
    if not npz_path.exists():
        return None

    with np.load(npz_path) as d:
        k = d["k"]                       # [K]
        sigma_vals = d["sigma_vals"]     # [S]
        psd_hr_mean = d["psd_hr_mean"]   # [K]
        psd_hr_std  = d["psd_hr_std"]    # [K]
        psd_lr_mean = d["psd_lr_mean"]   # [K]
        psd_lr_std  = d["psd_lr_std"]    # [K]
        psd_gen_mean = d["psd_gen_mean"] # [S,K]
        psd_gen_std  = d["psd_gen_std"]  # [S,K]
        lr_nyq = float(d["lr_nyquist"]) if "lr_nyquist" in d.files else 0.0
        psd_band = tuple(d["psd_band_km"]) if "psd_band_km" in d.files else (5.0, 20.0)

    # Optional: read ramp/meta info
    ramp_info = None
    meta_path = (Path(out_dir) / "sigma_control_meta.json")
    if meta_path.exists():
        try:
            import json
            with open(meta_path, "r") as f:
                meta = json.load(f)
                ramp_info = meta.get("ramp", None)
        except Exception:
            ramp_info = None

    # Convert to wavelength (km), positive k only
    mpos = k > 0
    k_pos = k[mpos]
    lam = 1.0 / k_pos
    order = np.argsort(lam)[::-1] # Descending wavelength (reads large-scale to small-scale)
    lam = lam[order]

    # Prepare style
    _nice()
    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    # HR and LR references
    hr = np.maximum(psd_hr_mean[mpos][order], 1e-12)
    lr = np.maximum(psd_lr_mean[mpos][order], 1e-12)
    ax.plot(lam, hr, color=get_color_for_model("hr"), lw=2.0, label="HR (DANRA)")
    ax.plot(lam, lr, color=get_color_for_model("lr"), lw=1.6, ls="--", label="LR (ERA5↑)")

    # Shaded slope band (in λ)
    lam_lo, lam_hi = float(psd_band[0]), float(psd_band[1])  # e.g., 5–20 km
    left, right = min(lam_lo, lam_hi), max(lam_lo, lam_hi)
    ax.axvspan(left, right, color="0.85", alpha=0.35, zorder=0)
    # Optional: visualize the "controlled" high-k region (λ <= right) lightly
    ax.axvspan(0.0, right, color="0.9", alpha=0.15, hatch='///', linewidth=0, label="σ* control (late)") 

    # σ* curves with shaded ±1σ and green colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap("Greens")
    S = len(sigma_vals)
    slopes_txt = []
    
    for i, s in enumerate(sigma_vals):
        c = cmap(0.15 + 0.75 * (1 - i / max(1, S-1)))  # darkest for smallest σ*
        mean_i = np.maximum(psd_gen_mean[i][mpos][order], 1e-12)
        std_i  = np.maximum(psd_gen_std[i][mpos][order], 0.0)
        ax.plot(lam, mean_i, lw=1.8, color=c, label=fr"GEN (σ*={s:.2f})")
        # CI shading (±1σ in PSD domain)
        upper = mean_i + std_i
        lower = np.clip(mean_i - std_i, 1e-14, None)
        ax.fill_between(lam, lower, upper, color=c, alpha=0.25, linewidth=0)

        # Compute slope in the band using log10 fit
        k_lo = 1.0 / right
        k_hi = 1.0 / left
        mband = (k_pos > k_lo) & (k_pos < k_hi)
        if mband.any():
            x = np.log10(k_pos[mband])
            y = np.log10(psd_gen_mean[i][mpos][mband])
            coef = np.polyfit(x, y, 1)
            slopes_txt.append(fr"σ*={s:.2f}: {coef[0]:.2f}")
        else:
            slopes_txt.append(fr"σ*={s:.2f}: n/a")

    # Compute HR and LR slopes in the same mesoscale band
    k_lo = 1.0 / right
    k_hi = 1.0 / left
    mband_hr = (k_pos > k_lo) & (k_pos < k_hi)
    mband_lr = (k_pos > k_lo) & (k_pos < k_hi)
    hr_slope = np.nan
    lr_slope = np.nan
    if mband_hr.any():
        xh = np.log10(k_pos[mband_hr])
        yh = np.log10(hr[mband_hr])
        hr_slope = np.polyfit(xh, yh, 1)[0]
    if mband_lr.any():
        xl = np.log10(k_pos[mband_lr])
        yl = np.log10(lr[mband_lr])
        lr_slope = np.polyfit(xl, yl, 1)[0]
    slopes_txt.append(fr"HR: {hr_slope:.2f}")
    slopes_txt.append(fr"LR: {lr_slope:.2f}")

    # LR Nyquist as vertical line
    if lr_nyq > 0.0:
        lam_nyq = 1.0 / lr_nyq
        ax.axvline(lam_nyq, color="0.2", lw=1.0, ls="--", label="LR Nyq")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Wavelength λ (km)")
    ax.set_ylabel("Spectral power")
    ax.set_title(r"Mean ensemble PSDs vs wavelength across $\sigma^*$")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=9, frameon=False)

    # Add slope textbox
    text_lines = ["PSD slope in band:"]
    text_lines.extend(slopes_txt)
    if ramp_info is not None:
        mode = str(ramp_info.get("mode", "global"))
        sf = ramp_info.get("start_frac", None)
        ef = ramp_info.get("end_frac", None)
        ss = ramp_info.get("start_sigma", None)
        es = ramp_info.get("end_sigma", None)
        text_lines.append("")
        text_lines.append("Ramp:")
        text_lines.append(f"mode: {mode}")
        if (sf is not None) and (ef is not None):
            text_lines.append(f"frac: {float(sf):.2f}→{float(ef):.2f}")
        if (ss is not None) and (es is not None):
            text_lines.append(f"σ gate: ≤{ss} → ≤{es}")
    textstr = "\n".join(text_lines)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="0.6"))

    out_path = figs / "sigma_psd_curves.png"
    _savefig(fig, out_path, dpi=SET_DPI)
    return str(out_path)