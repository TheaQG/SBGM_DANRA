from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List
import logging
import numpy as np
import matplotlib.pyplot as plt

from sbgm.variable_utils import get_cmap_for_variable, get_unit_for_variable, get_units

logger = logging.getLogger(__name__)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _nice():
    plt.rcParams.update({
        "figure.figsize": (7.5, 5.0),
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })

# Try to borrow your preferred colormap limits if available
def _get_var_style(var: str):
    """
    Hook to your variable_utils. If not available, use sensible defaults.
    Returns (cmap, vmin, vmax, cbar_label)
    """
    if var in {"mean", "p95", "p99"}:
        cmap = "cividis"
        vmin, vmax = 0.0, None
        clabel = "mm/day"
        return (cmap, vmin, vmax, clabel)
    if var in {"sum", "rx1", "rx5"}:
        cmap = "magma"
        vmin, vmax = 0.0, None
        clabel = "mm"
        return (cmap, vmin, vmax, clabel)
    if var in {"wetfreq"}:
        cmap = "Blues"
        vmin, vmax = 0.0, 1.0
        clabel = "fraction of days"
        return (cmap, vmin, vmax, clabel)
    else:
        # generic
        cmap = "viridis"
        vmin, vmax = None, None
        clabel = "Unknown units"
        return (cmap, vmin, vmax, clabel)

def _load_npz(tables_dir: Path, tag: str):
    p = tables_dir / f"{tag}.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)

def _draw_single(ax, data, title: str, cmap="viridis", vmin=None, vmax=None, cbar_label=""):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if cbar_label:
        cb.set_label(cbar_label)

def plot_spatial_maps(eval_root: str | Path) -> None:
    """
    Compose a small set of multi-panel figures:
      1) For each group (year/season/all), plot HR vs GEN (and LR if present) grids for:
         mean, sum, rx1, rx5, p95, p99, wetfreq
      2) Also add GEN/HR ratio heatmaps for a few key fields (mean, sum, rx1)
    """
    eval_root = Path(eval_root)
    tables = eval_root / "tables"
    figs   = _ensure_dir(eval_root / "figures")

    # Discover groups by NPZ names
    tags = [p.stem for p in tables.glob("spatial_*.npz")]
    # tags look like: spatial_hr_2019, spatial_pmm_ALL, spatial_lr_DJF, etc.
    # Build mapping group -> dict(source->npz)
    buckets: Dict[str, Dict[str, Path]] = {}
    for stem in tags:
        parts = stem.split("_", 2)  # ["spatial", src, group]
        if len(parts) != 3: 
            continue
        _, src, group = parts
        buckets.setdefault(group, {})[src] = tables / f"{stem}.npz"

    if not buckets:
        logger.warning("[plot_spatial] No spatial_* NPZ files found under %s", str(tables))
        return

    variables = ["mean","sum","rx1","rx5","p95","p99","wetfreq"]

    for group, src_map in sorted(buckets.items()):
        # Load available sources
        npz_hr  = _load_npz(tables, f"spatial_hr_{group}")   if "hr"  in src_map else None
        npz_gen = _load_npz(tables, f"spatial_pmm_{group}")  if "pmm" in src_map else None
        npz_lr  = _load_npz(tables, f"spatial_lr_{group}")   if "lr"  in src_map else None

        # --- Figure 1: side-by-side maps for core variables (rows = vars, cols = HR/GEN/(LR)) ---
        for var in variables:
            arrs: List[np.ndarray] = []
            titles: List[str] = []
            cmap, vmin, vmax, clabel = _get_var_style(var)

            if npz_hr is not None and var in npz_hr:
                arrs.append(npz_hr[var]); titles.append(f"HR • {var}")
            if npz_gen is not None and var in npz_gen:
                arrs.append(npz_gen[var]); titles.append(f"Generated • {var}")
            if npz_lr is not None and var in npz_lr:
                arrs.append(npz_lr[var]); titles.append(f"LR • {var}")

            if not arrs:
                continue

            # compute common vmin/vmax across all available arrays for this var (robust percentiles)
            orig_vmin, orig_vmax = vmin, vmax
            if arrs:
                stack_vals = np.concatenate([a.reshape(-1) for a in arrs])
                # ignore NaNs
                stack_vals = stack_vals[np.isfinite(stack_vals)]
                if stack_vals.size > 0:
                    if vmin is None:
                        vmin = float(np.nanpercentile(stack_vals, 1.0))
                    if vmax is None:
                        vmax = float(np.nanpercentile(stack_vals, 99.0))
                    if np.isfinite(vmin) and np.isfinite(vmax) and vmin >= vmax:
                        # fallback if degenerate
                        vmin, vmax = float(np.nanmin(stack_vals)), float(np.nanmax(stack_vals))

            _nice()
            ncol = len(arrs)
            fig, axs = plt.subplots(1, ncol, figsize=(4.0*ncol, 3.6), squeeze=False)
            for j, a in enumerate(arrs):
                _draw_single(axs[0, j], a, titles[j], cmap=cmap, vmin=vmin, vmax=vmax, cbar_label=clabel)
            fig.suptitle(f"{group}: {var}")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(str(figs / f"spatial_{group}_{var}.png"), dpi=200)
            plt.close(fig)

        # --- Figure 2: GEN/HR ratio heatmaps for mean, sum, rx1 (optional LR/H R too) ---
        def _safe_ratio(num: Optional[np.ndarray], den: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if num is None or den is None:
                return None
            with np.errstate(divide="ignore", invalid="ignore"):
                r = num / den
            r[~np.isfinite(r)] = np.nan
            return r

        core = ["mean", "sum", "rx1"]
        for var in core:
            hr   = npz_hr[var]  if (npz_hr  is not None and var in npz_hr)  else None
            gen  = npz_gen[var] if (npz_gen is not None and var in npz_gen) else None
            lr   = npz_lr[var]  if (npz_lr  is not None and var in npz_lr)  else None

            rat_gen = _safe_ratio(gen, hr)
            rat_lr  = _safe_ratio(lr, hr)

            if rat_gen is None and rat_lr is None:
                continue

            _nice()
            cols = (1 if rat_gen is not None else 0) + (1 if rat_lr is not None else 0)
            fig, axs = plt.subplots(1, cols, figsize=(4.0*cols, 3.6), squeeze=False)
            j = 0
            if rat_gen is not None:
                _draw_single(axs[0, j], rat_gen, f"Generated/HR • {var}", cmap="PuOr", vmin=0.5, vmax=1.5, cbar_label="ratio")
                j += 1
            if rat_lr is not None:
                _draw_single(axs[0, j], rat_lr, f"LR/HR • {var}", cmap="PuOr", vmin=0.5, vmax=1.5, cbar_label="ratio")
            fig.suptitle(f"{group}: ratios ({var})")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(str(figs / f"spatial_{group}_{var}_ratios.png"), dpi=200)
            plt.close(fig)