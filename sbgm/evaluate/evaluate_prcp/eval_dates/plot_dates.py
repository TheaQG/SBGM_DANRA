# sbgm/evaluate/evaluate_prcp/eval_dates/plot_dates.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, Tuple, Callable, List
import torch
import numpy as np
import matplotlib.pyplot as plt

from sbgm.evaluate.evaluate_prcp.plot_utils import _ensure_dir, _nice, _savefig, get_dk_lsm_outline, overlay_outline
from sbgm.variable_utils import get_cmap_for_variable

from sbgm.plotting_utils import _add_colorbar_and_boxplot

from typing import List
import numpy as np

def _season_of(yyyymmdd: str) -> str:
    s = yyyymmdd.strip()
    m = int(s[4:6]) if len(s) >= 6 and s[:6].isdigit() else int(s.split("-")[1])
    return "DJF" if m in (12, 1, 2) else ("MAM" if m in (3, 4, 5) else ("JJA" if m in (6, 7, 8) else "SON"))

def _select_representative_dates(resolver, all_dates: List[str], k: int) -> List[str]:
    if not all_dates:
        return []
    # domain means with HR→PMM→LR fallback
    means = []
    for d in all_dates:
        arr = resolver.load_obs(d) or resolver.load_pmm(d) or resolver.load_lr(d)
        try:
            v = np.asarray(arr.detach().cpu().numpy() if hasattr(arr, "detach") else arr)
            means.append(np.nanmean(v))
        except Exception:
            means.append(np.nan)
    means = np.asarray(means, dtype=float)
    finite = np.isfinite(means)
    if not np.any(finite):
        return all_dates[:k]

    chosen = []
    # always include driest and wettest if available
    try:
        chosen.extend([all_dates[int(np.nanargmin(means))], all_dates[int(np.nanargmax(means))]])
    except Exception:
        pass

    # add one per season (median-intensity date in each season)
    for s in ("DJF", "MAM", "JJA", "SON"):
        cand = [d for d in all_dates if _season_of(d) == s]
        if not cand:
            continue
        vals = np.array([means[all_dates.index(d)] for d in cand], dtype=float)
        if np.any(np.isfinite(vals)):
            j = int(np.nanargmin(np.abs(vals - np.nanmedian(vals))))
            chosen.append(cand[j])

    # fill remaining slots by stratifying quantiles of means
    if len(chosen) < k:
        qgrid = np.linspace(0.1, 0.9, max(1, k - len(chosen)))
        quant_vals = np.nanquantile(means[finite], qgrid)
        for qv in quant_vals:
            idx = int(np.nanargmin(np.abs(means - qv)))
            chosen.append(all_dates[idx])
            if len(chosen) >= k:
                break

    # keep order stable and unique
    seen = set()
    final = []
    for d in chosen:
        if d not in seen and d in all_dates:
            final.append(d); seen.add(d)
    # pad if still short
    for d in all_dates:
        if len(final) >= k:
            break
        if d not in seen:
            final.append(d); seen.add(d)
    return final[:k]

SET_DPI = 300

def _squeeze2d(a: object | None) -> np.ndarray | None:
    if a is None:
        return None
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    x = np.asarray(a)
    # drop singleton dims repeatedly
    while x.ndim > 2 and 1 in x.shape:
        x = np.squeeze(x)
    if x.ndim == 3:
        # prefer [C,H,W] → first channel; else [H,W,C] → first channel
        if x.shape[0] <= 8:
            x = x[0]
        else:
            x = x[..., 0]
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
    """
        Load HR, PMM, optional LR and up to n ensemble members for one date.
        Labels: HR (Obs), PMM ("mean"), LR (ERA5), Ens-1, Ens-2, ...
    """
    # mask
    mask = None
    try:
        if land_only:
            m = resolver.load_mask(date)
            m = _squeeze2d(m)
            if m is not None:
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
        return _apply_mask(x, mask)

    lr  = _safe(getattr(resolver, "load_lr",  None)) if include_lr else None
    hr  = _safe(getattr(resolver, "load_obs", None))
    pmm = _safe(getattr(resolver, "load_pmm", None))

    panels: list[tuple[str, np.ndarray]] = []

    # 1) LR (optional)
    if include_lr and lr is not None:
        panels.append(("LR (ERA5)", lr))

    # 2) HR
    if hr is not None:
        panels.append(("HR (DANRA)", hr))

    # 3) Ens-# from resolver.load_ens
    if include_members and hasattr(resolver, "load_ens"):
        try:
            ens = resolver.load_ens(date)
            A = ens.detach().cpu().numpy() if isinstance(ens, torch.Tensor) else np.asarray(ens) if ens is not None else None
            if A is not None:
                if A.ndim == 4 and A.shape[1] == 1:   # [M,1,H,W] -> [M,H,W]
                    A = A[:, 0, :, :]
                if A.ndim == 3 and A.shape[0] > 0:
                    m = min(n_members, A.shape[0])
                    for i in range(m):
                        im = _apply_mask(_squeeze2d(A[i]), mask)
                        if im is not None:
                            panels.append((f"Ens-{i+1}", im))
        except Exception:
            pass

    # 4) PMM (last)
    if pmm is not None:
        panels.append(("PMM (gen)", pmm))

    return panels


def plot_dates_montages(
    resolver,
    out_root: str | Path,
    dates: Sequence[str],
    *,
    include_lr: bool = True,
    include_members: bool = True,
    n_members: int = 3,
    cmap: str = "auto",
    percentile: float = 99.5,
    land_only: bool = True,
    fname_prefix: str = "montage_",
) -> None:
    out_root = Path(out_root)
    figs_dir = _ensure_dir(out_root / "figures")

    # choose cmap
    try:
        base_cmap = get_cmap_for_variable("prcp") if cmap == "auto" else cmap
    except Exception:
        base_cmap = "Blues"
    hr_cmap = base_cmap
    lr_cmap = base_cmap

    dk_outline = get_dk_lsm_outline()

    # Collect rows and learn max #members to size the grid
    rows: list[list[tuple[str, np.ndarray]]] = []
    max_members = 0
    for d in dates:
        panels = _collect_panels_for_date(
            resolver, d,
            include_lr=include_lr,
            include_members=include_members,
            n_members=n_members,
            land_only=land_only,
        )
        max_members = max(max_members, sum(1 for lab, _ in panels if lab.startswith("Ens-")))
        rows.append(panels)

    R = len(rows)
    C = (1 if include_lr else 0) + 1 + max_members + 1  # LR | HR | Ens.. | PMM

    _nice()
    fig, axs = plt.subplots(R, C, figsize=(3.2 * C, 3.2 * R))
    if R == 1: axs = axs[np.newaxis, :]
    if C == 1: axs = axs[:, np.newaxis]

    for r, (d, panels) in enumerate(zip(dates, rows)):
        # slot-by-slot content for this row
        lr = hr = pmm = None
        members: list[np.ndarray] = []
        for lab, img in panels:
            if lab.startswith("LR"): lr = img
            elif lab.startswith("HR"): hr = img
            elif lab.startswith("PMM"): pmm = img
            elif lab.startswith("Ens-"): members.append(img)

        # per-row robust vlims
        pool = [x for x in [lr, hr, pmm, *members] if x is not None]
        if pool:
            flat = np.concatenate([v[np.isfinite(v)] for v in pool])
            vmin = 0.0
            vmax = float(np.nanpercentile(flat, percentile))
            vmax = max(vmin + 1e-6, vmax)
        else:
            vmin, vmax = 0.0, 1.0

        c = 0
        def draw(ax, img, title, is_lr=False):
            if img is None:
                ax.axis("off"); return
            im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap=(lr_cmap if is_lr else hr_cmap))
            overlay_outline(ax, dk_outline)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
            _add_colorbar_and_boxplot(fig, ax, im, img, boxplot=True, ylim=(vmin, vmax))

        if include_lr:
            draw(axs[r, c], lr, "LR (ERA5)", is_lr=True); c += 1
        draw(axs[r, c], hr, "HR (DANRA)"); c += 1
        for j in range(max_members):
            ax = axs[r, c + j]
            if j < len(members):
                draw(ax, members[j], f"Ens-{j+1}")
            else:
                ax.axis("off")
        c += max_members
        draw(axs[r, c], pmm, "PMM (gen)")

        # date label on the left
        axs[r, 0].set_ylabel(str(d), fontsize=9)

    fig.text(0.5, 0.02, "Precipitation [mm/day]", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    _savefig(fig, figs_dir / f"{fname_prefix}{R}dates_{int(n_members)}m.png", dpi=SET_DPI)
    plt.close(fig)


