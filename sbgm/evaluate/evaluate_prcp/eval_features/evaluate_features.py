# sbgm/evaluate/evaluate_prcp/eval_features/evaluate_features.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Dict, List
import numpy as np
import logging
import torch

from sbgm.evaluate.evaluate_prcp.eval_features.metrics_features import compute_sal
from sbgm.evaluate.evaluate_prcp.eval_features.plot_features import plot_features_all

logger = logging.getLogger(__name__)


def _year_of(d: str) -> int:
    s = d.strip()
    if len(s) == 8 and s.isdigit():
        return int(s[:4])
    return int(s[:4])


def _season_of(d: str) -> str:
    s = d.strip()
    if len(s) == 8 and s.isdigit():
        m = int(s[4:6])
    else:
        m = int(s[5:7])
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"


def _group_dates(dates: List[str], group_by: str, seasons: Sequence[str]) -> Dict[str, List[str]]:
    if group_by == "all":
        return {"ALL": dates}
    if group_by == "season":
        out: Dict[str, List[str]] = {s: [] for s in seasons}
        for d in dates:
            s = _season_of(d)
            if s in out:
                out[s].append(d)
        return {s: out.get(s, []) for s in seasons if s in out}
    # default: year
    buckets: Dict[str, List[str]] = {}
    for d in dates:
        y = _year_of(d)
        buckets.setdefault(str(y), []).append(d)
    return dict(sorted(buckets.items()))


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_hw_np(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        arr = x.detach().cpu().float().squeeze().numpy()
    else:
        arr = np.asarray(x).squeeze()
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    return arr


def _apply_mask(arr: np.ndarray | None, mask) -> np.ndarray | None:
    if arr is None:
        return None
    if mask is None:
        return arr
    m = _to_hw_np(mask)
    if m is None:
        return arr
    m = m > 0.5
    out = arr.copy()
    out[~m] = np.nan
    return out


def run_features(
    resolver,
    eval_cfg,
    out_root: str | Path,
    *,
    group_by: str = "year",
    seasons: Sequence[str] = ("ALL", "DJF", "MAM", "JJA", "SON"),
    make_plots: bool = True,
    plot_only: bool = False,
) -> None:
    """
    Main entry point for feature/object-based evaluation (SAL).
    """
    out_root = Path(out_root)
    tables_dir = _ensure_dir(out_root / "tables")
    figs_dir = _ensure_dir(out_root / "figures")

    include_lr = bool(getattr(eval_cfg, "feat_include_lr", True))

    # All available dates and grouping
    all_dates: List[str] = list(resolver.list_dates())
    if not all_dates:
        logger.warning("[evaluate_features] No dates found — aborting.")
        return
    groups = _group_dates(all_dates, group_by, seasons)

    # Plot-only mode
    if plot_only:
        sal_files = list(tables_dir.glob("sal_*.npz"))
        if not sal_files:
            logger.warning("[evaluate_features] No existing SAL tables found for plot_only=True.")
            return
        for f in sal_files:
            gname = f.stem.replace("sal_", "")
            data = dict(np.load(f, allow_pickle=True))
            plot_features_all(figs_dir, gname, data)
        return

    # Helper for computing mean maps
    def _mean_map(resolver, dates, src: str, use_mask: bool = True):
        if src == "HR":
            loader = resolver.load_obs
        elif src in ("GEN", "PMM"):
            loader = resolver.load_pmm
        elif src == "LR":
            loader = resolver.load_lr
        else:
            raise ValueError(f"Unknown source: {src}")
        acc = []
        for d in dates:
            x = loader(d)
            if x is None:
                continue
            x = _to_hw_np(x)
            m = resolver.load_mask(d) if use_mask else None
            x = _apply_mask(x, m)
            acc.append(x)
        if not acc:
            return None
        with np.errstate(all="ignore"):
            return np.nanmean(np.stack(acc, axis=0), axis=0)

    land_only = bool(getattr(eval_cfg, "eval_land_only", True))
    for gname, gdates in groups.items():
        if not gdates:
            continue
        hr_map = _mean_map(resolver, gdates, "HR", use_mask=land_only)
        gen_map = _mean_map(resolver, gdates, "GEN", use_mask=land_only)
        lr_map = (
            _mean_map(resolver, gdates, "LR", use_mask=land_only)
            if include_lr
            else None
        )

        if hr_map is None or gen_map is None:
            logger.warning("[evaluate_features] Group %s: missing HR/GEN data; skipping.", gname)
            continue

        sal_metrics = compute_sal(hr_map, gen_map, lr_map)
        np.savez_compressed(tables_dir / f"sal_{gname}.npz", **sal_metrics)

        if make_plots:
            plot_features_all(figs_dir, gname, sal_metrics)