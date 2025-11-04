# sbgm/evaluate/evaluate_prcp/eval_dates/plot_dates.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Tuple, Callable, List
import numpy as np
import matplotlib.pyplot as plt

from sbgm.evaluate.evaluate_prcp.plot_utils import _ensure_dir, _nice, _savefig
from sbgm.variable_utils import get_cmap_for_variable

SET_DPI = 300

def _squeeze2d(a: object | None) -> np.ndarray | None:
    if a is None:
        return None
    x = np.asarray(a)
    while x.ndim > 2 and 1 in x.shape:
        x = np.squeeze(x)
    # tolerate channel-first arrays: pick the first channel if looks like [C,H,W]
    if x.ndim == 3 and x.shape[0] <= 8:
        x = x[0]
    return x if x.ndim == 2 else None


def _apply_mask(x: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray | None:
    if x is None:
        return None
    if mask is None:
        return x
    out = x.copy()
    m = mask
    if m.shape != out.shape:
        m = np.broadcast_to(m, out.shape)
    out[~m] = np.nan
    return out


def _collect_panels_for_date(
    resolver,
    date: str,
    *,
    include_lr: bool = True,
    include_members: bool = True,
    n_members: int = 3,
    land_only: bool = True,
) -> list[tuple[str, np.ndarray]]:
    """Load HR, PMM, optional LR and up to n ensemble members for one date."""
    # mask
    mask = None
    try:
        if land_only:
            m = resolver.load_mask(date)  # may be tensor/array/None
            if m is not None:
                m = np.asarray(m)
                while m.ndim > 2 and 1 in m.shape:
                    m = np.squeeze(m)
                if m.ndim == 3 and m.shape[0] <= 8:
                    m = m[0]
                mask = (m > 0.5)
    except Exception:
        mask = None

    def _safe(load_fn: Callable[[str], object] | None) -> np.ndarray | None:
        if load_fn is None:
            return None
        try:
            x = load_fn(date)
        except Exception:
            return None
        x = _squeeze2d(x)
        x = _apply_mask(x, mask)
        return x

    hr  = _safe(getattr(resolver, "load_obs", None))
    pmm = _safe(getattr(resolver, "load_pmm", None))
    lr  = _safe(getattr(resolver, "load_lr", None)) if include_lr else None

    panels: list[tuple[str, np.ndarray]] = []
    if hr is not None:  panels.append(("HR (DANRA)", hr))
    if pmm is not None: panels.append(("PMM (gen)", pmm))
    if include_lr and lr is not None:
        panels.append(("ERA5 (LR→HR)", lr))

    if include_members and hasattr(resolver, "load_ensembles"):
        try:
            ens = resolver.load_ensembles(date)  # expected [M,H,W] or [M,1,H,W]
            A = np.asarray(ens)
            if A.ndim == 4 and A.shape[1] == 1:
                A = A[:, 0, :, :]
            if A.ndim == 3 and A.shape[0] > 0:
                m = min(n_members, A.shape[0])
                for i in range(m):
                    im = _squeeze2d(A[i])
                    im = _apply_mask(im, mask)
                    if im is not None:
                        panels.append((f"M{i+1}", im))
        except Exception:
            pass

    return panels


def plot_dates_montages(
    resolver,
    out_root: str | Path,
    dates: Sequence[str],
    *,
    include_lr: bool = True,
    include_members: bool = True,
    n_members: int = 3,
    cmap: str = "Blues",
    percentile: float = 99.5,
    land_only: bool = True,
    fname_prefix: str = "montage_",
) -> None:
    """
    For each date, create a panel comparing:
      HR | PMM | (LR) | M1..Mk
    Uses a **shared** vmin/vmax per-date based on the chosen percentile.
    """
    out_root = Path(out_root)
    figs_dir = _ensure_dir(out_root / "figures")

    # Set cmap based on variable_utils, if "auto" - else, use given string
    if cmap == "auto":
        variable = resolver.get_variable_name()
        cmap = get_cmap_for_variable(variable)
    else:
        cmap = str(cmap)

    for d in dates:
        panels = _collect_panels_for_date(
            resolver, d,
            include_lr=include_lr,
            include_members=include_members,
            n_members=n_members,
            land_only=land_only,
        )
        if not panels:
            continue

        # shared normalization across panels for that date
        vals = [p[1] for p in panels if p[1] is not None]
        pool = np.concatenate([v[np.isfinite(v)].ravel() for v in vals if np.isfinite(v).any()])
        if pool.size == 0:
            continue
        vmin = 0.0
        vmax = float(np.nanpercentile(pool, percentile))
        vmax = max(vmin + 1e-6, vmax)

        _nice()
        C = len(panels)
        fig, axs = plt.subplots(1, C, figsize=(3.5 * C, 3.2), constrained_layout=True)
        if isinstance(axs, np.ndarray):
            axes = axs.ravel().tolist()
        else:
            axes = [axs]

        ims = []
        for ax, (lab, img) in zip(axes, panels):
            im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
            ims.append(im)
            ax.set_title(lab)
            ax.set_xticks([]); ax.set_yticks([])

        # single colorbar on right
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cb = fig.colorbar(ims[0], cax=cax)
        cb.set_label("mm/day")

        fig.suptitle(d)
        _savefig(fig, figs_dir / f"{fname_prefix}{d}.png", dpi=SET_DPI)
        plt.close(fig)