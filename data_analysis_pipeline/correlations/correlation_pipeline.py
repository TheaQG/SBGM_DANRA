"""
Module for running data correlation analysis.
For each HR-LR pair:
1. Instantiate DataLoader with correct variable, model, domain
2. Load aligned data (shared dates)
3. Apply transformations (e.g., detrending, normalization)
4. Compute correlations:
    - Temporal: over time for spatial means (and optionally for each grid point or single grid point)
    - Spatial: per-pixel correlations over time
5. Save results:
    - Correlation maps (.npy or .zarr)
    - Summary statistics (mean, median, std) (.npy or .json)
    - Visualizations (heatmaps, time series plots) (.png)
"""

import os
import logging
from itertools import product
import numpy as np

from data_analysis_pipeline.stats_analysis.data_loading import DataLoader
from data_analysis_pipeline.correlations.correlation_methods import compute_temporal_correlation, compute_spatial_correlation
from data_analysis_pipeline.correlations.correlation_plotting import plot_correlation_map, plot_temporal_series
from data_analysis_pipeline.stats_analysis.statistics import load_global_stats
from sbgm.special_transforms import transform_from_stats
# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def run_data_correlations(cfg):
    """
        Main correlation analysis entry point.
        Loops over HR x LR variable paris and runs selected correlation methods.
    """

    # === Load config sections ===
    hr_cfg = cfg["highres"]
    lr_cfg = cfg["lowres"]
    data_cfg = cfg["data"]
    corr_cfg = cfg["correlation"]
    plot_cfg = cfg["plotting"]

    # === Extract config entries ===
    figs_save_dir = cfg.get("paths", {}).get("output_dir", "./correlation_outputs")
    stats_save_dir = cfg.get("paths", {}).get("stats_save_dir", "./correlation_stats")
    stats_load_dir = cfg.get("paths", {}).get("stats_load_dir", stats_save_dir)
    os.makedirs(figs_save_dir, exist_ok=True)
    os.makedirs(stats_save_dir, exist_ok=True)
    normalize = corr_cfg.get("normalize", False)
    if not os.path.exists(stats_load_dir):
        logger.warning(f"Stats load directory '{stats_load_dir}' does not exist. Normalization will be skipped.")
        normalize = False
        
    correlation_types = corr_cfg.get("analysis_types", ["temporal", "spatial"])
    split = data_cfg.get("split", "all")
    method = corr_cfg.get("method", "pearson")

    hr_vars = hr_cfg.get("variables", [])
    domain_size_hr = hr_cfg.get("domain_size", [])
    domain_size_hr_str = "x".join(map(str, domain_size_hr)) if domain_size_hr else "full"
    crop_region_hr = hr_cfg.get("crop_region", [])
    crop_region_hr_str = "_".join(map(str, crop_region_hr)) if crop_region_hr else "full"
    model_hr = hr_cfg.get("model", "")

    lr_vars = lr_cfg.get("condition_variables", [])
    domain_size_lr = lr_cfg.get("domain_size", [])
    domain_size_lr_str = "x".join(map(str, domain_size_lr)) if domain_size_lr else "full"
    crop_region_lr = lr_cfg.get("crop_region", [])
    crop_region_lr_str = "_".join(map(str, crop_region_lr)) if crop_region_lr else "full"
    model_lr = lr_cfg.get("model", "")

    # dictionary of possible transformations to apply to data before correlation
    transformations_dict = corr_cfg.get("transformations", {})

    data_dir = data_cfg.get("data_dir", ".")
    n_workers = int(data_cfg.get("n_workers", 4))
    verbose = data_cfg.get("verbose", False)

    for hr_var, lr_var in product(hr_vars, lr_vars):
        logger.info(f"\n\n\n  ############## Running correlation analysis for HR variable '{hr_var}' ({model_hr}) and LR variable '{lr_var}' ({model_lr}) ##############\n\n")
        logger.info(f"  N workers: {data_cfg.get('n_workers', 4)}")

        # === Load data ===
        hr_loader = DataLoader(
            base_dir=data_dir,
            n_workers=n_workers,
            variable=hr_var,
            model=model_hr,
            domain_size=domain_size_hr,
            split=split,
            crop_region=crop_region_hr,
            verbose=verbose,
        )
        lr_loader = DataLoader(
            base_dir=data_dir,
            n_workers=n_workers,
            variable=lr_var,
            model=model_lr,
            domain_size=domain_size_lr,
            split=split,
            crop_region=crop_region_lr,
            verbose=verbose,
        )

        hr_data = hr_loader.load()
        lr_data = lr_loader.load()

        # Align datasets on shared dates
        shared_dates = sorted(set(hr_data["timestamps"]) & set(lr_data["timestamps"]))
        logger.info(f"  Found {len(shared_dates)} shared dates between HR '{hr_var}' and LR '{lr_var}'")

        hr_dict = {d: c for d, c in zip(hr_data["timestamps"], hr_data["cutouts"]) if d in shared_dates}
        lr_dict = {d: c for d, c in zip(lr_data["timestamps"], lr_data["cutouts"]) if d in shared_dates}

        save_str_add = ""
        if normalize:
            try:
                # Determine transformation method
                hr_transform_method = transformations_dict.get(hr_var, "zscore")
                lr_transform_method = transformations_dict.get(lr_var, "zscore")

                save_str_add = f"_transformed_{hr_transform_method}_and_{lr_transform_method}"
                logger.info("          Applying normalization based on global stats")
                # Load global stats for normalization
                hr_stats_dict = load_global_stats(
                                                hr_var,
                                                model_hr,
                                                domain_size_hr_str, 
                                                crop_region_hr_str,
                                                split=split,
                                                dir_load=stats_load_dir
                                                )
                lr_stats_dict = load_global_stats(
                                                lr_var,
                                                model_lr,
                                                domain_size_lr_str, 
                                                crop_region_lr_str,
                                                split=split,
                                                dir_load=stats_load_dir
                                                )
                logger.info(f"          Using transformation '{hr_transform_method}' for HR variable")
                logger.info(f"          Using transformation '{lr_transform_method}' for LR variable")
                # Apply normalization to each cutout
                for date in shared_dates:
                    hr_dict[date] = transform_from_stats(data=hr_dict[date], transform_type=hr_transform_method, cfg=cfg, stats=hr_stats_dict)
                    
                    if not isinstance(hr_dict[date], np.ndarray):
                        hr_dict[date] = np.array(hr_dict[date])
        
                    lr_dict[date] = transform_from_stats(data=lr_dict[date], transform_type=lr_transform_method, cfg=cfg, stats=lr_stats_dict)
                    if not isinstance(lr_dict[date], np.ndarray):
                        lr_dict[date] = np.array(lr_dict[date])

            except Exception as e:
                logger.warning(f"          Failed to load global stats for normalization: {e}. Skipping normalization.")
                hr_stats_dict = None
                lr_stats_dict = None
                save_str_add = ""
        else:
            logger.info("          Skipping normalization")

        for corr_type in correlation_types:
            logger.info(f"          Running {corr_type} correlation using method '{method}'")
            
            if corr_type == "temporal":
                result = compute_temporal_correlation(hr_dict, lr_dict, method=method)
                plot_temporal_series(
                    hr_series=result['series_hr'],
                    lr_series=result['series_lr'],
                    dates=shared_dates,
                    variable1=hr_var,
                    variable2=lr_var,
                    model1=model_hr,
                    model2=model_lr,
                    save_path=os.path.join(figs_save_dir, f"temporal_series_{hr_var}_{lr_var}_{model_hr}_vs_{model_lr}{save_str_add}.png"),
                    show=plot_cfg.get("show", False)
                )

            elif corr_type == "spatial":
                corr_map = compute_spatial_correlation(hr_dict, lr_dict, method=method)
                np.save(os.path.join(stats_save_dir, f"spatial_corr_map_{hr_var}_{lr_var}_{model_hr}_vs_{model_lr}{save_str_add}.npy"), corr_map)
                plot_correlation_map(
                    corr_map=corr_map,
                    variable1=hr_var,
                    variable2=lr_var,
                    model1=model_hr,
                    model2=model_lr,
                    save_path=os.path.join(figs_save_dir, f"spatial_corr_map_{hr_var}_{lr_var}_{model_hr}_vs_{model_lr}{save_str_add}.png"),
                    show=plot_cfg.get("show", False)
                )
            else:
                logger.warning(f"          Unknown correlation type '{corr_type}' specified. Skipping.")


