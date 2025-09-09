import os
import logging
import numpy as np

from data_analysis_pipeline.stats_analysis.data_loading import DataLoader
from data_analysis_pipeline.stats_analysis.plotting import plot_cutout_example, visualize_statistics
from data_analysis_pipeline.stats_analysis.statistics import compute_statistics


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def run_data_statistics(cfg):
    all_results = {}

    split = cfg.get("data", {}).get("split", "all")

    data_dir = cfg.get("data", {}).get("data_dir", ".")
    fig_save_dir = cfg.get("plotting", {}).get("fig_save_dir", ".")
    stats_save_dir = cfg.get("statistics", {}).get("stats_save_dir", ".")
    logger.info(f"  PATHS:")
    logger.info(f"          DATA: {data_dir}")
    logger.info(f"          FIGS SAVE: {fig_save_dir}")
    logger.info(f"          STATS SAVE: {stats_save_dir}")

    n_workers = int(cfg.get("data", {}).get("n_workers", 1))

    logger.info(f"Running data statistics with split='{split}', data_dir='{data_dir}', n_workers={n_workers}")
    

    def process_variable(var_cfg, variable, agg_method, agg_time, level):
        logger.info(f"Processing {level.upper()} variable: {variable} ({var_cfg.get('model', '')}) | agg_method={agg_method} | agg_time={agg_time}")

        loader = DataLoader(
            base_dir=data_dir,
            n_workers=n_workers,
            variable=variable,
            model=var_cfg.get("model", ""),
            domain_size=var_cfg.get("domain_size", []),
            split=split,
            crop_region=var_cfg.get("crop_region", []),
            verbose=cfg.get("data", {}).get("verbose", False),
        )
        raw_data = loader.load()
        logger.info(f"Loaded {len(raw_data['cutouts'])} cutouts for {variable}")

        # === Basic Statistics ===
        if cfg.get("statistics", {}).get("save_global_pixel_stats", True):
            save_glob_stats = True
            logger.info(f"Saving global statistics for {variable}")
        else:
            save_glob_stats = False
            logger.info(f"Not saving global statistics for {variable}")

        crop_region_str = "_".join(map(str, var_cfg.get("crop_region", []))) if var_cfg.get("crop_region", []) else "full"
        domain_size_str = "x".join(map(str, var_cfg.get("domain_size", []))) if var_cfg.get("domain_size", []) else "full"
        logger.info(f"Crop region: {crop_region_str} | Domain size: {domain_size_str}")


        global_stats, cutout_stats, time_series_stats = compute_statistics(
                                                                raw_data,
                                                                print_stats=True,
                                                                return_all=True,
                                                                save_glob_stats=save_glob_stats,
                                                                variable=variable,
                                                                model=var_cfg.get("model", ""),
                                                                split=split,
                                                                domain_str=domain_size_str,
                                                                crop_region_str=crop_region_str,
                                                                cfg=cfg,
                                                                stats_save_path=stats_save_dir,
                                                                log_stats=variable in ['prcp', 'cape'],
                                                                pool_pixels=True,
                                                                )
        all_results[f"{level}__{variable}"] = {
            "global": global_stats,
            "cutout": cutout_stats,
            "timeseries": time_series_stats,
        }
        
        if cfg.get("plotting", {}).get("save_cutout_example", False):
            current_fig_save_path = os.path.join(fig_save_dir, var_cfg.get("model", ""), variable, split)
            logger.info(f"Saving cutout example for variable {variable}")
            plot_cutout_example(raw_data, variable, cfg, current_fig_save_path)

        if cfg.get("plotting", {}).get("visualize_data", False):
            current_fig_save_path = os.path.join(fig_save_dir, var_cfg.get("model", ""), variable, split)
            logger.info(f"Generating raw data visualizations at {current_fig_save_path}")
            if variable in ['temp', 'ewvf', 'nwvf', 'msl', 'z_pl_1000', 'z_pl_250', 'z_pl_500', 'z_pl_850']:
                transforms = ['zscore']
                log_scale = False
            elif variable in ['prcp', 'cape']:
                transforms = ['log', 'log_zscore']
                log_scale = True
                # Make sure all data is positive for log plots
                raw_data['cutouts'] = [np.where(c <= 0, 1e-8, c) for c in raw_data['cutouts']]
            else:
                transforms = ['zscore']
                log_scale = False
                         
                         
            visualize_statistics(
                variable,
                raw_data,
                {
                    "global": global_stats,
                    "cutout": cutout_stats,
                    "timeseries": time_series_stats,
                },
                cfg,
                model=var_cfg.get("model", ""),
                load_global=True,
                domain_str=domain_size_str,
                crop_region_str=crop_region_str,
                split=split,
                fig_save_path=current_fig_save_path,
                dir_load_glob=stats_save_dir,
                aggregated=False,
                show_transformed=True,
                transforms=transforms,
                log_scale=log_scale,
            )

        if cfg.get("statistics", {}).get("aggregate", False):
            logger.info(f"Computing aggregated statistics for {level.upper()} variable '{variable}' using method: {agg_method}, time: {agg_time}")
            global_stats, cutout_stats, time_series_stats = compute_statistics(
                                                                    raw_data,
                                                                    aggregate=True,
                                                                    agg_method=agg_method,
                                                                    agg_time=cfg.get("data", {}).get("aggregation_time", "monthly"),
                                                                    return_all=True,
                                                                    print_stats=True,
                                                                    save_glob_stats=save_glob_stats,
                                                                    variable=variable,
                                                                    model=var_cfg.get("model", ""),
                                                                    split=split,
                                                                    domain_str=domain_size_str,
                                                                    crop_region_str=crop_region_str,
                                                                    cfg=cfg,
                                                                    stats_save_path=stats_save_dir,
                                                                    log_stats=variable in ['prcp', 'cape'],
                                                                    pool_pixels=True,
                                                                    )
            all_results[f"{level}__agg__{variable}"] = {
                "global": global_stats,
                "cutout": cutout_stats,
                "timeseries": time_series_stats,
            }

            if cfg.get("plotting", {}).get("visualize_aggregated", False):
                current_fig_save_path = os.path.join(fig_save_dir, var_cfg.get("model", ""), variable, split)
                logger.info(f"Generating aggregated data visualizations at {current_fig_save_path}")
                if variable in ['temp', 'ewvf', 'nwvf', 'msl', 'z_pl_1000', 'z_pl_250', 'z_pl_500', 'z_pl_850']:
                    transforms = ['zscore']
                    log_scale = False
                elif variable in ['prcp', 'cape']:
                    transforms = ['log', 'log_zscore']
                    log_scale = True
                    # Make sure all data is positive for log plots
                    raw_data['cutouts'] = [np.where(c <= 0, 1e-8, c) for c in raw_data['cutouts']]
                else:
                    transforms = ['zscore']
                    log_scale = False

                
                visualize_statistics(
                    variable,
                    raw_data,
                    {
                        "global": global_stats,
                        "cutout": cutout_stats,
                        "timeseries": time_series_stats,
                    },
                    cfg,
                    model=var_cfg.get("model", ""),
                    fig_save_path=current_fig_save_path,
                    aggregated=True,
                    agg_method=agg_method,
                    agg_time=agg_time,
                    show_transformed=True,
                    transforms=transforms,
                    log_scale=log_scale,
                    load_global=True,
                    domain_str=domain_size_str,
                    crop_region_str=crop_region_str,
                    split=split,
                    dir_load_glob=stats_save_dir,
                )

    # === Process HR variables ===
    for variable, agg_method in zip(cfg.get("highres", {}).get("variables", []), cfg.get("highres", {}).get("agg_methods", [])):
        agg_time = cfg.get("data", {}).get("aggregation_time", "monthly")
        logger.info(f"\nProcessing highres variable: {variable} ({cfg.get('highres', {}).get('model', '')})| agg_method={agg_method}\n")
        process_variable(cfg.get("highres", {}), variable, agg_method, agg_time, level="highres")

    # === Process LR variables ===
    for variable, agg_method in zip(cfg.get("lowres", {}).get("condition_variables", []), cfg.get("lowres", {}).get("agg_methods", [])):
        agg_time = cfg.get("data", {}).get("aggregation_time", "monthly")
        logger.info(f"\nProcessing lowres variable: {variable} ({cfg.get('lowres', {}).get('model', '')})| agg_method={agg_method}\n")
        process_variable(cfg.get("lowres", {}), variable, agg_method, agg_time, level="lowres")


    logger.info("Finished running full data statistics pipeline. Final results:")
    for key, value in all_results.items():
        
        for inner_key, inner_value in value.items():
            if inner_key in cfg.get("statistics", {}).get("print_results", []):
                logger.info(f" - {key} | {inner_key}: {inner_value}")
            # else:
            #     logger.info(f" - {key}: (not printed, not in print_results list)")

    return all_results