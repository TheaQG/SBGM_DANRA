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
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=False)
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
    showing HR reference and one curve per sigma* (mean +/- std shading optional).
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
        k = d["k"]                 # [K]
        sigma_vals = d["sigma_vals"]  # [S]
        psd_hr_mean = d["psd_hr_mean"]  # [K]
        psd_hr_std  = d["psd_hr_std"]   # [K]
        psd_gen_mean = d["psd_gen_mean"]  # [S,K]
        psd_gen_std  = d["psd_gen_std"]   # [S,K]
        lr_nyq = float(d["lr_nyquist"]) if "lr_nyquist" in d.files else 0.0

    # Convert to wavelength (km), positive k only
    mpos = k > 0
    k_pos = k[mpos]
    lam = 1.0 / k_pos
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    hr_mean = np.maximum(psd_hr_mean[mpos][order], 1e-12)

    _nice()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # HR reference with shading
    ax.plot(lam, hr_mean, color=get_color_for_model("hr"), lw=1.8, label="HR (DANRA)")

    # Color cycle across sigma*
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
    if colors is None:
        colors = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

    for i, s in enumerate(sigma_vals):
        gen_mean = np.maximum(psd_gen_mean[i][mpos][order], 1e-12)
        ax.plot(lam, gen_mean, lw=1.4, color=colors[i % len(colors)], label=fr"GEN (σ*={s:.2f})")

    # LR Nyquist as vertical line
    if lr_nyq > 0.0:
        lam_nyq = 1.0 / lr_nyq
        ax.axvline(lam_nyq, color="0.2", lw=0.8, ls="--", label="LR Nyq")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Wavelength λ (km)")
    ax.set_ylabel("Spectral power")
    ax.set_title("Averaged PSD vs wavelength across σ*")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)

    out_path = figs / "sigma_psd_curves.png"
    _savefig(fig, out_path, dpi=SET_DPI)
    return str(out_path)