import argparse
import os
import re
import torch
import yaml
import zarr

import matplotlib.pyplot as plt
import numpy as np

from typing import Any
from sbgm.utils import build_data_path
from sbgm.plotting_utils import plot_sample
from sbgm.variable_utils import get_units
from sbgm.data_modules import DANRA_Dataset_cutouts_ERA5_Zarr

def _to_np(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 2):  # [C,H,W] -> take first channel for quicklook
            x = x[0]
        return x.numpy()
    return np.asarray(x)



# --- ENVIRONMENT VARIABLE RESOLVER FOR ${env:VAR} ---
_ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")

def _resolve_env_in_obj(obj):
    """Recursively resolve ${env:VARNAME} placeholders in a nested config object."""
    if isinstance(obj, dict):
        return {k: _resolve_env_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_in_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_resolve_env_in_obj(v) for v in obj)
    if isinstance(obj, str):
        def _sub(m):
            var = m.group(1)
            val = os.environ.get(var, None)
            if val is None:
                raise KeyError(f"Environment variable '{var}' is not set but is required by config value: {obj}")
            return val
        return _ENV_PATTERN.sub(_sub, obj)
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_path", type=str, required=True, help="Path to YAML config (same as training).")
    ap.add_argument("--n", type=int, default=3, help="number of samples to show")
    ap.add_argument("--start", type=int, default=0, help="start index")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--show_original", action="store_true")
    ap.add_argument("--no_cutouts", action="store_true")
    args = ap.parse_args()

    # ------------------------------
    # Build dataset exactly like training_utils.get_dataloader
    # ------------------------------
    with open(args.cfg_path, "r") as f:
        cfg: Any = yaml.safe_load(f)

    # Expand ${env:...} placeholders to match LUMI-style config behaviour locally
    cfg = _resolve_env_in_obj(cfg)

    # Log types (optional)
    try:
        hr_unit, lr_units = get_units(cfg)
        print(f"[tester] HR: {cfg['highres']['model']} {cfg['highres']['variable']} [{hr_unit}]")
        for i, cond in enumerate(cfg['lowres']['condition_variables']):
            u = lr_units[i] if lr_units is not None and i < len(lr_units) else "?"
            print(f"[tester] LR {i+1}: {cfg['lowres']['model']} {cond} [{u}]")
    except Exception:
        pass

    # Sizes
    hr_data_size = tuple(cfg['highres']['data_size']) if cfg['highres'].get('data_size', None) is not None else (128, 128)

    lr_data_size = cfg['lowres'].get('data_size', None)
    if lr_data_size is None:
        lr_data_size_use = hr_data_size
    else:
        lr_data_size_use = tuple(lr_data_size)

    resize_factor = int(cfg['lowres'].get('resize_factor', 1))
    hr_data_size_use = hr_data_size
    if resize_factor > 1:
        hr_data_size_use = (hr_data_size[0] // resize_factor, hr_data_size[1] // resize_factor)
        lr_data_size_use = (lr_data_size_use[0] // resize_factor, lr_data_size_use[1] // resize_factor)

    # Full domain dims for path naming
    full_domain_dims = tuple(cfg['highres']['full_domain_dims']) if cfg['highres'].get('full_domain_dims', None) is not None else None

    split = str(args.split).lower()
    if split in ("gen", "generation", "test", "inference"):
        split_key = "test"
    elif split in ("val", "valid", "validation"):
        split_key = "valid"
    else:
        split_key = "train"

    # Paths for HR + LR
    hr_data_dir = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims, split_key)

    lr_cond_dirs = {}
    for cond in cfg['lowres']['condition_variables']:
        lr_cond_dirs[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], cond, full_domain_dims, split_key)

    # Stationary geo
    geo_cfg = cfg.get('stationary_conditions', {}).get('geographic_conditions', {}) or {}
    sample_w_geo = bool(geo_cfg.get('sample_w_geo', False))
    if bool(geo_cfg.get('sample_w_sdf', False)):
        sample_w_geo = True

    if sample_w_geo:
        geo_variables = geo_cfg.get('geo_variables', ['lsm', 'topo'])
        data_dir_lsm = cfg['paths']['lsm_path']
        data_dir_topo = cfg['paths']['topo_path']

        # training_utils flips inputs here
        data_lsm = np.flipud(np.load(data_dir_lsm)['data']).copy()
        data_topo = np.flipud(np.load(data_dir_topo)['data']).copy()

        # Optional scaling of topo to norm range (same logic as training_utils)
        if bool(cfg.get('transforms', {}).get('scaling', False)):
            topo_min = geo_cfg.get('topo_min', None)
            topo_max = geo_cfg.get('topo_max', None)
            if topo_min is None or topo_max is None:
                topo_min, topo_max = float(np.min(data_topo)), float(np.max(data_topo))

            norm_min = geo_cfg.get('norm_min', None)
            norm_max = geo_cfg.get('norm_max', None)
            if norm_min is None or norm_max is None:
                # NOTE: training_utils used lsm min/max as defaults
                norm_min, norm_max = float(np.min(data_lsm)), float(np.max(data_lsm))

            old_range = (topo_max - topo_min)
            new_range = (norm_max - norm_min)
            if old_range != 0:
                data_topo = ((data_topo - topo_min) * new_range / old_range) + norm_min
    else:
        geo_variables = None
        data_lsm = None
        data_topo = None

    # Cutouts and domains
    cutouts = bool(cfg.get('transforms', {}).get('sample_w_cutouts', False))

    # training_utils sets defaults if missing
    cutout_domains = tuple(cfg['highres'].get('cutout_domains', None) or (170, 350, 340, 520))
    lr_cutout_domains = tuple(cfg['lowres'].get('cutout_domains', None) or (170, 350, 340, 520))

    # Stationary cutout geometry (mirror training_utils policy)
    highres_stationary_cfg = cfg['highres'].get('stationary_cutout', {}) or {}
    stationary_cutout_hr = bool(highres_stationary_cfg.get('enabled', False))
    hr_bounds = highres_stationary_cfg.get('bounds', None)

    lowres_stationary_cfg = cfg['lowres'].get('stationary_cutout', {}) or {}
    stationary_cutout_lr = bool(lowres_stationary_cfg.get('enabled', False))
    lr_bounds = lowres_stationary_cfg.get('bounds', None)

    eval_stationary_cfg = cfg.get('evaluation', {}).get('stationary_cutout', {}) or {}
    stationary_cutout_gen_hr = bool(eval_stationary_cfg.get('hr_enabled', stationary_cutout_hr))
    stationary_cutout_gen_lr = bool(eval_stationary_cfg.get('lr_enabled', stationary_cutout_lr))
    hr_bounds_gen = eval_stationary_cfg.get('hr_bounds', None)
    lr_bounds_gen = eval_stationary_cfg.get('lr_bounds', None)

    fg_cfg = cfg.get('full_gen_eval', {}) or {}
    fg_stationary = fg_cfg.get('stationary_cutout', {}) or {}
    if hr_bounds_gen is None:
        hr_bounds_gen = fg_stationary.get('hr_bounds', None)
    if lr_bounds_gen is None:
        lr_bounds_gen = fg_stationary.get('lr_bounds', None)
    if hr_bounds_gen is None:
        hr_bounds_gen = hr_bounds
    if lr_bounds_gen is None:
        lr_bounds_gen = lr_bounds

    if split_key == "test":
        fixed_cutout_hr = stationary_cutout_gen_hr
        fixed_cutout_lr = stationary_cutout_gen_lr
        fixed_hr_bounds = hr_bounds_gen
        fixed_lr_bounds = lr_bounds_gen
    else:
        fixed_cutout_hr = stationary_cutout_hr
        fixed_cutout_lr = stationary_cutout_lr
        fixed_hr_bounds = hr_bounds
        fixed_lr_bounds = lr_bounds

    # Seasonality
    seas_cfg = cfg.get('stationary_conditions', {}).get('seasonal_conditions', {}) or {}
    conditional_seasons = bool(seas_cfg.get('sample_w_cond_season', False))
    use_sin_cos_embedding = bool(seas_cfg.get('use_sin_cos_embedding', False))
    use_leap_years = bool(seas_cfg.get('use_leap_years', True))

    if conditional_seasons:
        n_seasons = seas_cfg.get('n_seasons', None)
    else:
        n_seasons = None

    # Determine n_samples and cache_size similarly to training_utils
    data_zarr = zarr.open_group(hr_data_dir, mode='r')
    n_samples = len(list(data_zarr.keys()))

    cache_cfg = cfg.get('data_handling', {}).get('cache_size', 0)
    if int(cache_cfg) == 0:
        cache_size = max(1, n_samples // 2)
    else:
        cache_size = int(cache_cfg)

    ds = DANRA_Dataset_cutouts_ERA5_Zarr(
        hr_variable_dir_zarr=hr_data_dir,
        hr_data_size=hr_data_size_use,
        n_samples=n_samples,
        cache_size=cache_size,
        hr_variable=cfg['highres']['variable'],
        hr_model=cfg['highres']['model'],
        hr_scaling_method=cfg['highres']['scaling_method'],
        lr_conditions=cfg['lowres']['condition_variables'],
        lr_model=cfg['lowres']['model'],
        lr_scaling_methods=cfg['lowres']['scaling_methods'],
        lr_cond_dirs_zarr=lr_cond_dirs,
        geo_variables=geo_variables,
        lsm_full_domain=data_lsm,
        topo_full_domain=data_topo,
        conditional_seasons=conditional_seasons,
        use_sin_cos_embedding=use_sin_cos_embedding,
        use_leap_years=use_leap_years,
        cfg=cfg,
        split=split_key,
        shuffle=(split_key != 'test'),
        cutouts=cutouts,
        cutout_domains=list(cutout_domains) if cutouts else None,
        n_samples_w_cutouts=n_samples,
        sdf_weighted_loss=bool(geo_cfg.get('sample_w_sdf', False)),
        scale=bool(cfg.get('transforms', {}).get('scaling', False)),
        save_original=bool(cfg.get('visualization', {}).get('show_both_orig_scaled', False)) or bool(args.show_original),
        n_classes=n_seasons,
        lr_data_size=tuple(lr_data_size_use) if lr_data_size_use is not None else None,
        lr_cutout_domains=list(lr_cutout_domains) if lr_cutout_domains is not None else None,
        resize_factor=resize_factor,
        fixed_cutout_hr=fixed_cutout_hr,
        fixed_hr_bounds=fixed_hr_bounds,
        fixed_cutout_lr=fixed_cutout_lr,
        fixed_lr_bounds=fixed_lr_bounds,
    )

    print(f"[tester] Built dataset split='{split_key}': len={len(ds)}")
    if cutouts:
        print(f"[tester] cutouts enabled; fixed_cutout_hr={fixed_cutout_hr}, fixed_cutout_lr={fixed_cutout_lr}")
        print(f"[tester] HR bounds={fixed_hr_bounds} | LR bounds={fixed_lr_bounds}")
    if args.no_cutouts:
        ds.cutouts = False

    end = min(len(ds), args.start + args.n)
    for i in range(args.start, end):
        s = ds[i]
        date = s.get("date", f"idx={i}")

        # Make sure plot_sample includes originals if requested via CLI
        if isinstance(cfg, dict):
            cfg_vis = cfg.get("visualization", {}) or {}
            cfg_vis["show_both_orig_scaled"] = bool(args.show_original) or bool(cfg_vis.get("show_both_orig_scaled", False))
            cfg["visualization"] = cfg_vis

        # Use shared plotting (correct orientation, colormaps, dual-LR handling, etc.)
        plot_sample(s, cfg, figsize=(15, 4))

        # Add cutout points in the title for quick debugging
        plt.suptitle(
            f"{date} | hr_point={s.get('hr_points', None)} | lr_point={s.get('lr_points', None)}",
            fontsize=10
        )
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()