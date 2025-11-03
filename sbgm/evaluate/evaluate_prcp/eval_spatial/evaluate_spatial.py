from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List
import logging
import numpy as np
import torch

from .metrics_spatial import (
    accumulate_daily_fields,
    compute_spatial_climatologies,
    save_maps_npz,
)

logger = logging.getLogger(__name__)

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def run_spatial(
    resolver,                 # EvalDataResolver-like (load_obs/load_pmm/load_lr/load_mask/list_dates)
    eval_cfg,                 # namespace-like; we read attributes with getattr
    out_root: str | Path,
    *,
    which_source: str = "pmm",   # "pmm" (generated), "hr" (truth), "lr" (upsampled ERA5)
    group_by: str = "year",      # "year" | "season" | "all"
    seasons: Sequence[str] = ("ALL","DJF","MAM","JJA","SON"),
    make_plots: bool = True,
) -> None:
    """
    Spatial maps evaluation:
      - Compute pixelwise climatologies (mean, sum, wetfreq, Rx1, Rx5, P95/P99)
      - For HR, GEN (PMM), and optional LR
      - Grouped by year, season, or whole period
    Writes per-group NPZ bundles to <out_root>/tables and then calls plotter.
    """
    out_root = Path(out_root)
    tables_dir = _ensure_dir(out_root / "tables")
    figs_dir = _ensure_dir(out_root / "figures")

    hr_first = bool(getattr(eval_cfg, "spatial_include_hr", True))
    gen_first = bool(getattr(eval_cfg, "spatial_include_gen", True))
    lr_first = bool(getattr(eval_cfg, "spatial_include_lr", True))

    wet_thr = float(getattr(eval_cfg, "spatial_wet_thr_mm", 1.0))
    rxk_days = tuple(getattr(eval_cfg, "spatial_rxk_days", (1,5)))
    pct_list = tuple(getattr(eval_cfg, "spatial_percentiles", (95.0, 99.0)))

    # pick loader by source label for convenience
    def loader(label: str):
        if label == "hr":
            return resolver.load_obs
        if label in {"pmm", "gen", "generated"}:
            return resolver.load_pmm
        if label == "lr":
            return resolver.load_lr
        raise ValueError(f"Unknown source: {label}")

    # dates universe
    all_dates: List[str] = list(resolver.list_dates())
    if not all_dates:
        logger.warning("[spatial] No dates found. Nothing to do.")
        return

    # Utility: split dates per group
    def split_dates(dates: List[str]) -> Dict[str, List[str]]:
        if group_by == "all":
            return {"ALL": dates}
        if group_by == "season":
            # mimic your other season splitter (DJF etc.); accept YYYYMMDD or YYYY-MM-DD stems
            out: Dict[str, List[str]] = {s: [] for s in seasons}
            for d in dates:
                s = _season_of(d)
                if s in out:
                    out[s].append(d)
            # Keep only requested seasons (preserve order)
            return {s: out.get(s, []) for s in seasons if s in out}
        # default year
        buckets: Dict[str, List[str]] = {}
        for d in dates:
            y = _year_of(d)
            buckets.setdefault(str(y), []).append(d)
        return dict(sorted(buckets.items()))
    
    def _year_of(d: str) -> int:
        s = d.strip()
        if len(s) == 8 and s.isdigit():
            return int(s[:4])
        # assume YYYY-MM-DD
        return int(s[:4])

    def _season_of(d: str) -> str:
        # returns DJF/MAM/JJA/SON from month
        s = d.strip()
        if len(s) == 8 and s.isdigit():
            m = int(s[4:6])
        else:
            m = int(s[5:7])
        if m in (12,1,2): return "DJF"
        if m in (3,4,5):  return "MAM"
        if m in (6,7,8):  return "JJA"
        return "SON"

    groups = split_dates(all_dates)
    logger.info("[spatial] Groups: %s", ", ".join(f"{k}({len(v)})" for k,v in groups.items()))

    # Which sources to compute
    sources: List[str] = []
    if hr_first: sources.append("hr")
    if gen_first: sources.append("pmm")
    if lr_first: sources.append("lr")
    if not sources:
        sources = ["pmm"]

    # Loop groups × sources
    for gname, gdates in groups.items():
        if not gdates:
            logger.warning("[spatial] Group %s is empty; skipping.", gname)
            continue

        # Mask: prefer resolver.load_mask (land & ROI handled there)
        def mask_fn(d: str):
            return resolver.load_mask(d)

        for src in sources:
            load_fn = loader(src)
            kept, daily_list, union_mask = accumulate_daily_fields(
                gdates, load_fn=load_fn, mask_fn=mask_fn,
                use_mask=bool(getattr(eval_cfg, "eval_land_only", True))
            )
            if not daily_list:
                logger.warning("[spatial] No %s fields in group %s", src, gname)
                continue

            maps = compute_spatial_climatologies(
                daily_list,
                wet_thr_mm=wet_thr,
                percentiles=pct_list,
                rxk_days=rxk_days,
            )
            # Save bundle NPZ
            npz_path = tables_dir / f"spatial_{src}_{gname}.npz"
            save_maps_npz(npz_path, **maps)
            # Also write a tiny sidecar meta with counts
            (tables_dir / f"spatial_{src}_{gname}.meta.txt").write_text(
                f"source={src}\n"
                f"group={gname}\n"
                f"n_days={len(kept)}\n"
                f"wet_thr_mm={wet_thr}\n"
                f"rxk_days={list(rxk_days)}\n"
                f"percentiles={list(pct_list)}\n"
            )

    # Plotting
    if make_plots:
        try:
            from .plot_spatial import plot_spatial_maps
            plot_spatial_maps(out_root)
            logger.info("[spatial] Plots saved to %s", str(out_root / "figures"))
        except Exception as e:
            logger.warning(f"[spatial] Could not produce plots: {e}")