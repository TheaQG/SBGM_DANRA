#!/usr/bin/env python3
"""
test_dataset_transforms.py

Sanity checks for:
- DANRA_Dataset_cutouts_ERA5_Zarr __getitem__
- forward transforms from stats (inside Dataset)
- back transforms from stats (special_transforms.build_back_transforms_from_stats)
- remap_between_scalings / lr_baseline_to_hr_zspace
- dual_lr channel semantics + potential mismatch causing low sums

Run:
  python scripts/test_dataset_transforms.py --cfg path/to/config.yaml --n 5 --idx 0 10 100
"""



from __future__ import annotations

import os
import json
import copy
import argparse
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import torch

from pathlib import Path
import matplotlib.pyplot as plt

import yaml

# Import datasets + transform helpers
from sbgm.data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
from sbgm.special_transforms import (
    build_back_transforms_from_stats,
    load_global_stats,
    lr_baseline_to_hr_zspace,
    get_transforms_from_stats,
    get_backtransforms_from_stats,
)

# Colormap helper for variables
from sbgm.variable_utils import get_cmap_for_variable

# ------------------------- helpers -------------------------

def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return None

def _stats(x: Any) -> str:
    a = _to_numpy(x)
    if a is None:
        return "None"
    a = np.asarray(a)
    if a.size == 0:
        return f"empty shape={a.shape}"
    return f"shape={a.shape} dtype={a.dtype} min={np.nanmin(a):.4g} mean={np.nanmean(a):.4g} max={np.nanmax(a):.4g}"

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.nanmean((a - b) ** 2))

def _mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.nanmean(np.abs(a - b)))

def _rel_sum_err(a: np.ndarray, b: np.ndarray) -> float:
    sa = float(np.nansum(a))
    sb = float(np.nansum(b))
    denom = max(abs(sa), 1e-12)
    return float((sb - sa) / denom)

def _print_header(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------- dataset tester helpers -------------------------

def _ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _tensor_info_simple(name, t):
    if t is None:
        return f"{name}: None"
    if torch.is_tensor(t):
        return (
            f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"min={float(t.min()):.4g} mean={float(t.mean()):.4g} max={float(t.max()):.4g}"
        )
    return f"{name}: type={type(t)}"

def _save_dataset_quicklook_plotting_utils(sample: Dict[str, Any], cfg: dict, outdir: Path, figsize: Tuple[int,int]) -> None:
    try:
        from sbgm.plotting_utils import plot_sample
    except Exception as e:
        print(f"[DATASET TESTER] Could not import plot_sample: {e}")
        return

    date_str = sample.get("date", "unknown")
    hr_pts = sample.get("hr_points", None)
    lr_pts = sample.get("lr_points", None)
    hr_tag = "hrpts_" + "_".join(map(str, hr_pts)) if isinstance(hr_pts, (list, tuple)) else "hrpts_na"
    lr_tag = "lrpts_" + "_".join(map(str, lr_pts)) if isinstance(lr_pts, (list, tuple)) else "lrpts_na"

    try:
        fig, _ = plot_sample(sample, cfg, figsize=tuple(figsize))
        fig.savefig(str(outdir / f"dataset_quicklook_{date_str}_{hr_tag}_{lr_tag}.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[DATASET TESTER] plot_sample failed: {e}")

def _save_dataset_quicklook(sample: Dict[str, Any], outdir: Path, max_lr_vars: int = 3):
    hr_keys = [k for k in sample.keys() if k.endswith("_hr")]
    if not hr_keys:
        return
    
    hr_key = hr_keys[0]
    hr = sample[hr_key]
    if not (torch.is_tensor(hr) and hr.ndim >= 2):
        return
    
    hr_img = hr[0] if hr.ndim == 3 else hr  # handle channel dim
    hr_img = hr_img.detach().cpu().numpy()

    lr_keys = [k for k in sample.keys() if k.endswith("_lr")][:max_lr_vars]

    ncols = 1 + len(lr_keys)
    fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    hr_var = hr_key.replace('_hr', '')
    im = axs[0].imshow(hr_img, cmap=get_cmap_for_variable(hr_var))
    axs[0].set_title(f"HR: {hr_key}")
    plt.colorbar(im, ax=axs[0])

    for i, k in enumerate(lr_keys, start=1):
        lr = sample[k]
        if torch.is_tensor(lr):
            img = lr[0] if lr.ndim == 3 else lr
            img = img.detach().cpu().numpy()
            lr_var = k.replace('_lr', '')
            im = axs[i].imshow(img, cmap=get_cmap_for_variable(lr_var))
            axs[i].set_title(f"LR: {k}")
            plt.colorbar(im, ax=axs[i])
    
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    # Build informative filename
    date_str = sample.get('date', 'unknown')
    hr_pts = sample.get('hr_points', None)
    lr_pts = sample.get('lr_points', None)
    hr_tag = 'hrpts_' + '_'.join(map(str, hr_pts)) if isinstance(hr_pts, (list, tuple)) else 'hrpts_na'
    lr_tag = 'lrpts_' + '_'.join(map(str, lr_pts)) if isinstance(lr_pts, (list, tuple)) else 'lrpts_na'
    fig.savefig(str(outdir / f"dataset_quicklook_{date_str}_{hr_tag}_{lr_tag}.png"), dpi=150)
    plt.close(fig)


# ------------------------- region stats helpers -------------------------

def _region_slices(h: int, w: int) -> Dict[str, Tuple[slice, slice]]:
    """ A few consistent subregions for quick sanity stats. """
    cy0, cy1 = int(h * 0.25), int(h * 0.75)
    cx0, cx1 = int(w * 0.25), int(w * 0.75)
    return {
        "full": (slice(0, h), slice(0, w)),
        "center": (slice(cy0, cy1), slice(cx0, cx1)),
        "nw": (slice(0, cy0), slice(0, cx0)),
        "ne": (slice(0, cy0), slice(cx1, w)),
        "sw": (slice(cy1, h), slice(0, cx0)),
        "se": (slice(cy1, h), slice(cx1, w)),
    }


def _stats_dict(x: Any) -> Dict[str, Any]:
    a = _to_numpy(x)
    if a is None:
        return {'shape': None}
    a = np.asarray(a)
    if a.size == 0:
        return {'shape': list(a.shape)}
    return {
        'shape': list(a.shape),
        'min': float(np.nanmin(a)),
        'mean': float(np.nanmean(a)),
        'max': float(np.nanmax(a)),
        'sum': float(np.nansum(a)),
        'p95': float(np.nanpercentile(a, 95)),
        'p99': float(np.nanpercentile(a, 99)),
    }


def _write_sample_stats(sample: Dict[str, Any], outdir: Path, keys: List[str]) -> None:
    """ Write per-sample stats JSON (full + a few subregions) for selected tensor keys"""
    date_str = str(sample.get('date', 'unknown'))
    hr_pts = sample.get('hr_points', None)
    lr_pts = sample.get('lr_points', None)
    hr_tag = 'hrpts_' + '_'.join(map(str, hr_pts)) if isinstance(hr_pts, (list, tuple)) else 'hrpts_na'
    lr_tag = 'lrpts_' + '_'.join(map(str, lr_pts)) if isinstance(lr_pts, (list, tuple)) else 'lrpts_na'

    payload: Dict[str, Any] = {
        'date': date_str,
        'hr_points': hr_pts,
        'lr_points': lr_pts,
        'stats': {}
    }

    for k in keys:
        if k not in sample:
            continue
        arr = _to_numpy(sample[k])
        if arr is None:
            continue
        arr = np.asarray(arr)
        # If channel-first 3D: take channel 0 for regional stats, but keep global stats on full tensor
        payload['stats'][k] = {'global': _stats_dict(arr)}
        if arr.ndim >= 2:
            # Choose 2D plane for regional stats
            if arr.ndim == 3:
                plane = arr[0]
            elif arr.ndim == 2:
                plane = arr
            elif arr.ndim == 4:
                plane = arr[0, 0]
            else:
                plane = arr.reshape(arr.shape[-2], arr.shape[-1])
            
            h, w = plane.shape[-2], plane.shape[-1]
            regs = _region_slices(h, w)
            payload['stats'][k]['regions'] = {}
            for rname, (ys, xs) in regs.items():
                payload['stats'][k]['regions'][rname] = _stats_dict(plane[ys, xs])

    (outdir / f"dataset_stats_{date_str}_{hr_tag}_{lr_tag}.json").write_text(json.dumps(payload, indent=2) + "\n")

# ------------------------- main checks -------------------------

# --- Land/sea mask helpers and stats
def _get_land_mask(sample: Dict[str, Any], target_2d_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """ Return boolean mask for land pixels in HR space if available

    Try a few common keys used in dataset.
    Mask convention: land ~1, sea ~0.  
    """
    cand_keys = ["lsm", "lsm_hr", "landmask", "land_sea_mask"]
    mask = None
    for k in cand_keys:
        if k in sample:
            mask = _to_numpy(sample[k])
            if mask is not None:
                break

    if mask is None:
        return None
    
    mask = np.asarray(mask)

    # Reduce to 2D if needed
    if mask.ndim == 3:
        mask2 = mask[0]
    elif mask.ndim == 2:
        mask2 = mask
    elif mask.ndim == 4:
        mask2 = mask[0, 0]
    else:
        return None
    
    if mask2.shape != target_2d_shape:
        # Cannot safely resample; skip rather than wrong mask
        return None
    return (mask2 >= 0.5)

def _print_land_sea_stats(name: str, field_2d: np.ndarray, land_mask: Optional[np.ndarray]) -> None:
    """Print mean/std over all pixels vs land-only"""
    arr = np.asarray(field_2d)
    all_flat = arr[np.isfinite(arr)]
    if all_flat.size == 0:
        print(f"[MASK STATS] {name}: no valid pixels/finite values.")
        return
    
    mu_all = float(np.nanmean(all_flat))
    std_all = float(np.nanstd(all_flat))

    if land_mask is None:
        print(f"[MASK STATS] {name}: all pixels: mean={mu_all:.4g} std={std_all:.4g} (no land mask in sample)")
        return
    
    land_flat = arr[land_mask]
    land_flat = land_flat[np.isfinite(land_flat)]
    if land_flat.size == 0:
        print(f"[MASK STATS] {name}: all={mu_all:.4g} +/- {std_all:.4g} | land=EMPTY")
        return
    
    mu_land = float(np.nanmean(land_flat))
    sd_land = float(np.nanstd(land_flat))
    frac_land = float(np.mean(land_mask))

    print(f"[MASK STATS] {name}: all={mu_all:.4g} +/- {std_all:.4g} | land={mu_land:.4g} +/- {sd_land:.4g} (frac land={frac_land:.3f})")

# Dataset test runner
def run_dataset_tester(cfg: dict, outdir: str, indices: List[int],
                       *, save_stats: bool = False,
                       plot_style: str = "plotting_utils",
                       figsize: Tuple[int,int] = (15,4),
                       context_demo: bool = False):
    """
    Lightweight dataset sanity checker
    Verifies that spatial/temporal context mechanisms are active and
    inspects concrete samples before training/generation.

    Note: 
        This tester is not meant to compute normalization statistics for a domain.
        For that, run the full data_analysis_pipeline/stats_analysis on the target domain
    """
    outp = _ensure_outdir(outdir)

    # Save resolved config for provenance
    (outp / 'cfg_used.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))


    from sbgm.utils import build_data_path

    def _run_once(cfg_local: dict, tag: str):
        """Run the dataset tester once for a given cfg variant (used for D0/D1 demo)."""
        print("\n" + "#" * 90)
        print(f"[DATASET TESTER] CONTEXT DEMO RUN: {tag}")
        print("#" * 90)
        full_domain_dims_hr = cfg_local["highres"].get("full_domain_dims", None)
        full_domain_dims_lr = cfg_local["lowres"].get("full_domain_dims", None)

        hr_zarr = build_data_path(
            cfg_local['paths']['data_dir'],
            cfg_local['highres']['model'],
            cfg_local['highres']['variable'],
            full_domain_dims_hr,
            'train'
        )

        lr_zarr_dict = {}
        for v in cfg_local["lowres"]["condition_variables"]:
            lr_zarr_dict[v] = build_data_path(
                cfg_local['paths']['data_dir'],
                cfg_local['lowres']['model'],
                v,
                full_domain_dims_lr,
                'train'
            )

        # Load lsm/topo if available
        lsm = None
        topo = None
        lsm_path = cfg_local.get("paths", {}).get("lsm_path", None)
        topo_path = cfg_local.get("paths", {}).get("topo_path", None)
        if lsm_path and os.path.exists(lsm_path):
            lsm = np.flipud(np.load(lsm_path)["data"]).copy()
        if topo_path and os.path.exists(topo_path):
            topo = np.flipud(np.load(topo_path)["data"]).copy()

        default_bounds = (200, 328, 380, 508)
        hr_bounds = cfg_local.get("highres", {}).get("stationary_cutout", {}).get("bounds", None) or list(default_bounds)
        lr_bounds = cfg_local.get("lowres", {}).get("stationary_cutout", {}).get("bounds", None) or list(default_bounds)

        # NOTE: YAML stationary_cutout.bounds are [y0, y1, x0, x1]. Dataset convention is [x1, x2, y1, y2].
        def _yx_to_xy(b):
            if b is None:
                return None
            b = [int(v) for v in b]
            if len(b) != 4:
                return b
            y0, y1, x0, x1 = b
            return [x0, x1, y0, y1]

        hr_bounds_xy = _yx_to_xy(hr_bounds)
        lr_bounds_xy = _yx_to_xy(lr_bounds)

        ds = DANRA_Dataset_cutouts_ERA5_Zarr(
            hr_variable_dir_zarr=hr_zarr,
            hr_data_size=tuple(cfg_local["highres"]["data_size"]) if cfg_local["highres"]["data_size"] is not None else (128, 128),
            n_samples=int(cfg_local.get("data_handling", {}).get("n_samples_debug", 20)),
            cache_size=int(cfg_local.get("data_handling", {}).get("cache_size", 0)),
            hr_variable=cfg_local["highres"]["variable"],
            hr_model=cfg_local["highres"]["model"],
            hr_scaling_method=cfg_local["highres"]["scaling_method"],
            lr_conditions=cfg_local["lowres"]["condition_variables"],
            lr_model=cfg_local["lowres"]["model"],
            lr_scaling_methods=cfg_local["lowres"]["scaling_methods"],
            lr_cond_dirs_zarr=lr_zarr_dict,
            geo_variables=cfg_local.get('stationary_conditions', {}).get('geographic_conditions', {}).get('geo_variables', None),
            lsm_full_domain=lsm,
            topo_full_domain=topo,
            cfg=cfg_local,
            scale=cfg_local.get("transforms", {}).get("scaling", False),
            split='train',
            shuffle=False,
            cutouts=bool(cfg_local.get("transforms", {}).get("sample_w_cutouts", False)),
            cutout_domains=cfg_local["highres"].get("cutout_domains", None),
            lr_data_size=tuple(cfg_local["lowres"]["data_size"]) if cfg_local["lowres"]["data_size"] is not None else None,
            lr_cutout_domains=cfg_local["lowres"].get("cutout_domains", None),
            fixed_cutout_hr=True,
            fixed_hr_bounds=hr_bounds_xy,
            fixed_cutout_lr=True,
            fixed_lr_bounds=lr_bounds_xy,
        )

        print(f"[DATASET TESTER] ({tag}) Dataset length: {len(ds)}")
        # Print key context settings
        lr_cfg = cfg_local.get("lowres", {})
        print(f"[DATASET TESTER] ({tag}) lowres.context_mode={lr_cfg.get('context_mode', None)}")
        print(f"[DATASET TESTER] ({tag}) lowres.context_data_size={lr_cfg.get('context_data_size', None)}")
        print(f"[DATASET TESTER] ({tag}) lowres.data_size={lr_cfg.get('data_size', None)}")

        for i in indices:
            sample = ds[i]
            print(f"\n[DATASET TESTER] ({tag}) Sample index: {i}")
            for k in sorted(sample.keys()):
                print(_tensor_info_simple(k, sample[k]))

            # Explicit sanity: LR context window should match cfg['lowres']['context_data_size'] when provided
            for v in cfg_local["lowres"]["condition_variables"]:
                k = f"{v}_lr"
                if k in sample and torch.is_tensor(sample[k]):
                    print(f"[DATASET TESTER] ({tag}) {k} context shape: {tuple(sample[k].shape)}")

            # Save quicklook
            if plot_style == "plotting_utils":
                _save_dataset_quicklook_plotting_utils(sample, cfg_local, outp, figsize)
            else:
                _save_dataset_quicklook(sample, outp)

            # Optional stats JSON (debug only)
            if save_stats:
                keys = [f"{cfg_local['highres']['variable']}_hr"] + [f"{v}_lr" for v in cfg_local["lowres"]["condition_variables"]]
                for g in ["lsm", "lsm_hr", "topo", "sdf"]:
                    if g in sample:
                        keys.append(g)
                _write_sample_stats(sample, outp, keys)


    # If requested, run both D0 and D1 variants back-to-back.
    if context_demo:
        # D0: co-located patch LR (same size as HR patch)
        cfg_d0 = copy.deepcopy(cfg)
        cfg_d0.setdefault("lowres", {})
        cfg_d0["lowres"]["context_mode"] = "centered"
        cfg_d0["lowres"]["context_data_size"] = list(cfg_d0["highres"].get("data_size", [128, 128]))

        # D1: full-domain LR context
        cfg_d1 = copy.deepcopy(cfg)
        cfg_d1.setdefault("lowres", {})
        cfg_d1["lowres"]["context_mode"] = "full"
        # keep context_data_size present for clarity/debug, but full overrides it
        if cfg_d1["lowres"].get("full_domain_dims", None) is not None:
            cfg_d1["lowres"]["context_data_size"] = list(cfg_d1["lowres"]["full_domain_dims"])

        _run_once(cfg_d0, tag="D0")
        _run_once(cfg_d1, tag="D1")
    else:
        _run_once(cfg, tag="single")


@torch.no_grad()
def test_one_sample(
    sample: Dict[str, Any],
    *,
    cfg: dict,
    back_transforms: Dict[str, Any],
    verbose: bool = True,
) -> None:
    hr_var = cfg["highres"]["variable"]
    lr_vars = cfg["lowres"]["condition_variables"]

    if verbose:
        _print_header(f"SAMPLE date={sample.get('date')} idx_points hr={sample.get('hr_points')} lr={sample.get('lr_points')}")
        print(f"Keys: {sorted(sample.keys())}")

    # --- basic prints
    hr_key = f"{hr_var}_hr"
    print(f"\n[HR scaled] {hr_key}: {_stats(sample.get(hr_key))}")
    if f"{hr_var}_hr_original" in sample:
        print(f"[HR orig ] {hr_var}_hr_original: {_stats(sample.get(f'{hr_var}_hr_original'))}")

    # Land/sea masking sanity (HR space)
    if hr_key in sample and torch.is_tensor(sample.get(hr_key)):
        hr = sample[hr_key]
        hr2d = _to_numpy(hr[0] if (hr.ndim == 3) else (hr[0,0] if hr.ndim ==4 else hr))
        if hr2d is not None and hr2d.ndim == 2:
            land_mask = _get_land_mask(sample, (hr2d.shape[0], hr2d.shape[1]))
            _print_land_sea_stats(hr_key, hr2d, land_mask)

    for v in lr_vars:
        k = f"{v}_lr"
        print(f"\n[LR scaled] {k}: {_stats(sample.get(k))}")
        if f"{v}_lr_original" in sample:
            print(f"[LR orig ] {v}_lr_original: {_stats(sample.get(f'{v}_lr_original'))}")

    for g in ["lsm", "topo", "lsm_hr", "sdf"]:
        if g in sample:
            print(f"\n[GEO] {g}: {_stats(sample.get(g))}")

    # --- round-trip tests using back_transforms dict
    # HR round-trip: orig -> fwd(hr stats) -> inv(hr stats)  (requires orig)
    if f"{hr_var}_hr_original" in sample and sample.get(f"{hr_var}_hr_original") is not None:
        hr_orig = sample[f"{hr_var}_hr_original"]
        if torch.is_tensor(hr_orig):
            hr_orig_t = hr_orig.to(torch.float32)
        else:
            hr_orig_t = torch.tensor(hr_orig, dtype=torch.float32)

        # forward transform used in Dataset is not directly accessible here;
        # rebuild forward from stats (same params as dataset)
        domain_str_hr = f"{cfg['highres']['full_domain_dims'][0]}x{cfg['highres']['full_domain_dims'][1]}" if cfg["highres"]["full_domain_dims"] is not None else "full_domain"
        crop_hr = cfg["highres"]["cutout_domains"]
        crop_region_str_hr = "_".join(map(str, crop_hr)) if crop_hr is not None else "full"
        split = cfg["transforms"].get("scaling_split", "train")
        stats_root = cfg["paths"]["stats_load_dir"]
        hr_buffer = float(cfg["highres"].get("buffer_frac", 0.0))
        eps = float(cfg["transforms"].get("prcp_eps", 0.01)) if hr_var in ["prcp", "tp", "cape"] else 0.0

        fwd_hr = get_transforms_from_stats(
            variable=hr_var,
            model=cfg["highres"]["model"],
            domain_str=domain_str_hr,
            crop_region_str=crop_region_str_hr,
            scaling_split=split,
            transform_type=cfg["highres"]["scaling_method"],
            buffer_frac=hr_buffer,
            stats_file_path=stats_root,
            eps=eps,
        )
        inv_hr = back_transforms[hr_key]

        hr_norm = fwd_hr(hr_orig_t)
        hr_rec = inv_hr(hr_norm)

        a = _to_numpy(hr_orig_t)
        b = _to_numpy(hr_rec)
        print("\n[ROUNDTRIP HR] orig -> fwd(HR) -> inv(HR)")
        print(f"  MSE={_mse(a,b):.6g}  MAE={_mae(a,b):.6g}  rel_sum_err={(100*_rel_sum_err(a,b)):.3f}%") # type: ignore

    # LR round-trip tests for each condition if original exists
    domain_str_lr = f"{cfg['lowres']['full_domain_dims'][0]}x{cfg['lowres']['full_domain_dims'][1]}" if cfg["lowres"]["full_domain_dims"] is not None else "full_domain"
    crop_lr = cfg["lowres"]["cutout_domains"]
    crop_region_str_lr = "_".join(map(str, crop_lr)) if crop_lr is not None else "full"
    split = cfg["transforms"].get("scaling_split", "train")
    stats_root = cfg["paths"]["stats_load_dir"]
    lr_buffer = float(cfg["lowres"].get("buffer_frac", 0.0))

    for v, mth in zip(lr_vars, cfg["lowres"]["scaling_methods"]):
        if f"{v}_lr_original" not in sample or sample.get(f"{v}_lr_original") is None:
            continue

        lr_orig = sample[f"{v}_lr_original"]
        lr_orig_t = torch.tensor(lr_orig, dtype=torch.float32) if not torch.is_tensor(lr_orig) else lr_orig.to(torch.float32)

        eps = float(cfg["transforms"].get("prcp_eps", 0.01)) if v in ["prcp", "tp", "cape"] else 0.0
        fwd_lr = get_transforms_from_stats(
            variable=v,
            model=cfg["lowres"]["model"],
            domain_str=domain_str_lr,
            crop_region_str=crop_region_str_lr,
            scaling_split=split,
            transform_type=mth,
            buffer_frac=lr_buffer,
            stats_file_path=stats_root,
            eps=eps,
        )
        inv_lr = back_transforms[f"{v}_lr"]

        lr_norm = fwd_lr(lr_orig_t)
        lr_rec = inv_lr(lr_norm)

        a = _to_numpy(lr_orig_t)
        b = _to_numpy(lr_rec)
        print(f"\n[ROUNDTRIP LR] {v}: orig -> fwd(LR) -> inv(LR)")
        print(f"  MSE={_mse(a,b):.6g}  MAE={_mae(a,b):.6g}  rel_sum_err={(100*_rel_sum_err(a,b)):.3f}%") # type: ignore

    # --- Dual-LR mismatch probe (the exact failure mode you suspect)
    # If main variable exists in LR and sample[v_lr] has 2 channels: compare inverse behavior.
    main = cfg["highres"]["variable"]
    if main in lr_vars and torch.is_tensor(sample.get(f"{main}_lr")):
        lr_scaled = sample[f"{main}_lr"]
        if lr_scaled.ndim == 3 and lr_scaled.shape[0] == 2:
            print("\n[DUAL-LR] main LR has 2 channels. Testing inversion consequences.")
            ch0 = lr_scaled[0]
            ch1 = lr_scaled[1]

            # Invert both with the LR backtransform (this is what plotting/eval often does)
            inv_lr_main = back_transforms[f"{main}_lr"]
            x0 = inv_lr_main(ch0)
            x1 = inv_lr_main(ch1)

            print(f"  inv(LR) on channel0: {_stats(x0)}  sum={float(torch.sum(x0)):.4g}")
            print(f"  inv(LR) on channel1: {_stats(x1)}  sum={float(torch.sum(x1)):.4g}")

            # Now: take HR-scaled version (if available) and invert with LR inv — should show the “low sum / clamp” effect
            if torch.is_tensor(sample.get(f"{main}_hr")):
                hr_scaled = sample[f"{main}_hr"]
                x_bad = inv_lr_main(hr_scaled.squeeze(0) if hr_scaled.ndim == 3 and hr_scaled.shape[0] == 1 else hr_scaled)
                print("\n  [MISMATCH TEST] inv(LR) applied to HR-normalized field")
                print(f"    {_stats(x_bad)}  sum={float(torch.sum(x_bad)):.4g}")

    # --- Baseline remap test (LR norm -> HR z-space) if shapes allow
    # This is the “right” mapping for EDM residual baseline.
    if main in lr_vars and torch.is_tensor(sample.get(f"{main}_lr")) and torch.is_tensor(sample.get(f"{main}_hr")):
        # choose channel 1 if dual-lr, else single
        lr_scaled = sample[f"{main}_lr"]
        lr_chan = lr_scaled[1] if (lr_scaled.ndim == 3 and lr_scaled.shape[0] == 2) else (lr_scaled.squeeze(0) if lr_scaled.ndim == 3 else lr_scaled)

        hr_var = cfg["highres"]["variable"]
        hr_model = cfg["highres"]["model"]
        lr_model = cfg["lowres"]["model"]

        hr_domain_str = f"{cfg['highres']['full_domain_dims'][0]}x{cfg['highres']['full_domain_dims'][1]}" if cfg["highres"]["full_domain_dims"] is not None else "full_domain"
        lr_domain_str = f"{cfg['lowres']['full_domain_dims'][0]}x{cfg['lowres']['full_domain_dims'][1]}" if cfg["lowres"]["full_domain_dims"] is not None else "full_domain"
        hr_crop = cfg["highres"]["cutout_domains"]
        lr_crop = cfg["lowres"]["cutout_domains"]
        hr_crop_str = "_".join(map(str, hr_crop)) if hr_crop is not None else "full"
        lr_crop_str = "_".join(map(str, lr_crop)) if lr_crop is not None else "full"
        split = cfg["transforms"].get("scaling_split", "train")
        stats_root = cfg["paths"]["stats_load_dir"]
        hr_buffer = float(cfg["highres"].get("buffer_frac", 0.0))
        lr_buffer = float(cfg["lowres"].get("buffer_frac", 0.0))
        lr_method = cfg["lowres"]["scaling_methods"][lr_vars.index(main)]
        hr_method = cfg["highres"]["scaling_method"]
        eps = float(cfg["transforms"].get("prcp_eps", 0.01)) if main in ["prcp", "tp", "cape"] else 0.0

        mapped = lr_baseline_to_hr_zspace(
            lr_chan_norm=lr_chan,
            lr_variable=main,
            lr_model=lr_model,
            lr_domain_str=lr_domain_str,
            lr_crop_region_str=lr_crop_str,
            lr_split=split,
            lr_scaling_method=lr_method,
            lr_buffer_frac=lr_buffer,
            lr_stats_dir_root=stats_root,
            hr_variable=hr_var,
            hr_model=hr_model,
            hr_domain_str=hr_domain_str,
            hr_crop_region_str=hr_crop_str,
            hr_split=split,
            hr_buffer_frac=hr_buffer,
            hr_stats_dir_root=stats_root,
            hr_scaling_method=hr_method,
            eps=eps,
        )

        print("\n[REMAP] lr_baseline_to_hr_zspace output stats (should look like HR-normalized distribution):")
        print(f"  mapped: {_stats(mapped)}")
        print(f"  hr_scaled: {_stats(sample[f'{main}_hr'])}")

def main():
    """
        TODO:
            - Move all transform building logic to a function to avoid too much code in main()
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="Path to your YAML config.")
    ap.add_argument("--n", type=int, default=3, help="Number of samples to test (if --idx not given).")
    ap.add_argument("--idx", type=int, nargs="*", default=None, help="Explicit indices to test.")
    ap.add_argument("--no_cutouts", action="store_true", help="Force cutouts=False to simplify debugging.")
    ap.add_argument("--save_original", action="store_true", help="Force dataset save_original=True to enable roundtrip tests.")
    ap.add_argument("--verbose", action="store_true", help="Extra prints.")
    # Extend CLI with dataset-tester mode
    ap.add_argument("--mode", type=str, default='transforms', choices=['transforms', 'dataset'],
                    help='transforms: existing transform tests (default); dataset: dataset sanity tester (location, domains, context).')
    ap.add_argument('--outdir', type=str, default='./_dataset_tester_out', help='Output directory for dataset tester.')
    ap.add_argument("--dataset_save_stats", action="store_true", help="If set, write per-sample stats JSON (can be slow). Default: Off.")
    ap.add_argument("--dataset_plot_style", type=str, default='plotting_utils', choices=['plotting_utils', 'simple'], help="Plotting style for dataset tester quicklook.")
    ap.add_argument("--dataset_figsize", type=int, nargs=2, default=[15, 4], help="Figsize for plotting_utils plot_sample, e.g. --dataset_figsize 15 4")
    ap.add_argument("--dataset_random_crop", action="store_true", help="If set, dataset tester uses random cutouts. Default: fixed 128x128 bounds.")
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)




    # Dataset tester mode
    if args.mode == 'dataset':
        if args.idx is not None and len(args.idx) > 0:
            indices = args.idx
        else:
            indices = list(range(min(args.n, 3)))  # limit to 3 for tester

        run_dataset_tester(
            cfg,
            outdir=args.outdir,
            indices=indices,
            save_stats=args.dataset_save_stats,
            plot_style=args.dataset_plot_style,
            figsize=tuple(args.dataset_figsize),
            context_demo=True,
        )
        return




    # ---- Prepare minimal geo inputs if your dataset expects them
    # If your cfg points to lsm/topo paths elsewhere, you can load them here.
    # For now we attempt to load from cfg paths if present.
    lsm = None
    topo = None
    if "paths" in cfg:
        lsm_path = cfg["paths"].get("lsm_path", None)
        topo_path = cfg["paths"].get("topo_path", None)
        if lsm_path and os.path.exists(lsm_path):
            lsm = np.flipud(np.load(lsm_path)["data"]).copy()
        if topo_path and os.path.exists(topo_path):
            topo = np.flipud(np.load(topo_path)["data"]).copy()

    # ---- Build dataset paths (you likely already have these in your pipeline;
    # Here we assume cfg['paths']['data_dir'] etc. is consistent with your build_data_path helper.
    # If you prefer, pass direct zarr paths here.
    #
    # IMPORTANT: replace these with your own path builder if needed.

    full_domain_dims_hr = cfg["highres"].get("full_domain_dims", None)
    full_domain_dims_lr = cfg["lowres"].get("full_domain_dims", None)

    from sbgm.utils import build_data_path
    hr_zarr = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims_hr, 'train')
    lr_zarr_dict = {}
    for v in cfg["lowres"]["condition_variables"]:
        lr_zarr = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], v, full_domain_dims_lr, 'train')
        lr_zarr_dict[v] = lr_zarr

    # ---- Instantiate dataset
    dcfg = cfg.copy()
    if args.no_cutouts:
        dcfg["transforms"]["sample_w_cutouts"] = False
    if args.save_original:
        dcfg["visualization"]["show_both_orig_scaled"] = True  # matches your save_original usage

    ds = DANRA_Dataset_cutouts_ERA5_Zarr(
        hr_variable_dir_zarr=hr_zarr,
        hr_data_size=tuple(cfg["highres"]["data_size"]) if cfg["highres"]["data_size"] is not None else (128, 128),
        n_samples=int(cfg.get("data_handling", {}).get("n_samples_debug", 200)),
        cache_size=int(cfg.get("data_handling", {}).get("cache_size", 0)),
        hr_variable=cfg["highres"]["variable"],
        hr_model=cfg["highres"]["model"],
        hr_scaling_method=cfg["highres"]["scaling_method"],
        lr_conditions=cfg["lowres"]["condition_variables"],
        lr_model=cfg["lowres"]["model"],
        lr_scaling_methods=cfg["lowres"]["scaling_methods"],
        lr_cond_dirs_zarr=lr_zarr_dict,
        geo_variables=cfg.get("stationary_conditions", {}).get("geographic_conditions", {}).get("geo_variables", ["lsm", "topo"]),
        lsm_full_domain=lsm,
        topo_full_domain=topo,
        conditional_seasons=bool(cfg.get("stationary_conditions", {}).get("seasonal_conditions", {}).get("sample_w_cond_season", False)),
        use_sin_cos_embedding=bool(cfg.get("stationary_conditions", {}).get("seasonal_conditions", {}).get("use_sin_cos_embedding", False)),
        use_leap_years=bool(cfg.get("stationary_conditions", {}).get("seasonal_conditions", {}).get("use_leap_years", True)),
        cfg=dcfg,
        split=cfg.get("transforms", {}).get("scaling_split", "train"),
        shuffle=False,
        cutouts=bool(cfg.get("transforms", {}).get("sample_w_cutouts", False)),
        cutout_domains=cfg["highres"].get("cutout_domains", None),
        sdf_weighted_loss=bool(cfg.get("stationary_conditions", {}).get("geographic_conditions", {}).get("sample_w_sdf", False)),
        scale=bool(cfg.get("transforms", {}).get("scaling", True)),
        save_original=bool(dcfg.get("visualization", {}).get("show_both_orig_scaled", False)),
        n_classes=cfg.get("stationary_conditions", {}).get("seasonal_conditions", {}).get("n_seasons", None),
        lr_data_size=tuple(cfg["lowres"]["data_size"]) if cfg["lowres"]["data_size"] is not None else None,
        lr_cutout_domains=cfg["lowres"].get("cutout_domains", None),
        resize_factor=int(cfg["lowres"].get("resize_factor", 1)),
        fixed_cutout_hr=bool(cfg["highres"].get("stationary_cutout", {}).get("enabled", False)),
        fixed_hr_bounds=cfg["highres"].get("stationary_cutout", {}).get("bounds", None),
        fixed_cutout_lr=bool(cfg["lowres"].get("stationary_cutout", {}).get("enabled", False)),
        fixed_lr_bounds=cfg["lowres"].get("stationary_cutout", {}).get("bounds", None),
    )

    print(f"Dataset length: {len(ds)}")

    # ---- Build back-transforms using the same helper as training.py
    full_domain_dims_hr = cfg["highres"].get("full_domain_dims", None)
    full_domain_dims_lr = cfg["lowres"].get("full_domain_dims", None)

    domain_str_hr = f"{full_domain_dims_hr[0]}x{full_domain_dims_hr[1]}" if full_domain_dims_hr is not None else "full_domain"
    domain_str_lr = f"{full_domain_dims_lr[0]}x{full_domain_dims_lr[1]}" if full_domain_dims_lr is not None else "full_domain"

    crop_region_hr = cfg["highres"].get("cutout_domains", None)
    crop_region_lr = cfg["lowres"].get("cutout_domains", None)
    crop_region_str_hr = "_".join(map(str, crop_region_hr)) if crop_region_hr is not None else "full"
    crop_region_str_lr = "_".join(map(str, crop_region_lr)) if crop_region_lr is not None else "full"

    back_transforms = build_back_transforms_from_stats(
        hr_var=cfg["highres"]["variable"],
        hr_model=cfg["highres"]["model"],
        domain_str_hr=domain_str_hr,
        crop_region_str_hr=crop_region_str_hr,
        hr_scaling_method=cfg["highres"]["scaling_method"],
        hr_buffer_frac=float(cfg["highres"].get("buffer_frac", 0.0)),
        lr_vars=cfg["lowres"]["condition_variables"],
        lr_model=cfg["lowres"]["model"],
        lr_scaling_methods=cfg["lowres"]["scaling_methods"],
        domain_str_lr=domain_str_lr,
        crop_region_str_lr=crop_region_str_lr,
        lr_buffer_frac=float(cfg["lowres"].get("buffer_frac", 0.0)),
        split=cfg["transforms"].get("scaling_split", "train"),
        stats_dir_root=cfg["paths"]["stats_load_dir"],
        eps=float(cfg["transforms"].get("prcp_eps", 0.01)),
    )

    # ---- Extra: print log clamp ranges for prcp/tp/cape if present
    def _print_log_ranges(model: str, var: str):
        s = load_global_stats(
            variable=var,
            model=model,
            domain_str=domain_str_lr if model == cfg["lowres"]["model"] else domain_str_hr,
            crop_region_str=crop_region_str_lr if model == cfg["lowres"]["model"] else crop_region_str_hr,
            split=cfg["transforms"].get("scaling_split", "train"),
            dir_load=cfg["paths"]["stats_load_dir"],
        )
        if not s:
            return
        if "log_min" in s and "log_max" in s:
            print(f"  stats {model}/{var}: log_min={s['log_min']:.4g} log_max={s['log_max']:.4g} log_mean={s.get('log_mean', None)} log_std={s.get('log_std', None)}")

    print("\n[STATS clamp ranges]")
    for v in set(cfg["lowres"]["condition_variables"] + [cfg["highres"]["variable"]]):
        if v in ["prcp", "tp", "cape"]:
            _print_log_ranges(cfg["highres"]["model"], v)
            _print_log_ranges(cfg["lowres"]["model"], v)

    # ---- Select indices
    if args.idx is not None and len(args.idx) > 0:
        indices = args.idx
    else:
        indices = list(range(min(args.n, len(ds))))

    for i in indices:
        sample = ds[i]
        test_one_sample(sample, cfg=cfg, back_transforms=back_transforms, verbose=args.verbose)

if __name__ == "__main__":
    main()