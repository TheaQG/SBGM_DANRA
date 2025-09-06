import os
import logging

from data_analysis_pipeline.stats_analysis.data_loading import DataLoader
from data_analysis_pipeline.stats_analysis.plotting import plot_cutout_example, visualize_statistics
from data_analysis_pipeline.stats_analysis.transformations import get_transforms_from_stats, get_backtransforms_from_stats
from data_analysis_pipeline.stats_analysis.statistics import compute_statistics, aggregate_data
from data_analysis_pipeline.stats_analysis.global_stats import save_global_stats

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
    

    def process_variable(var_cfg, variable, agg_method, level):
        logger.info(f"Processing {level.upper()} variable: {variable} ({var_cfg.get('model', '')})| agg_method={agg_method}")

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
        global_stats, cutout_stats, time_series_stats = compute_statistics(raw_data, print_stats=True, return_all=True)
        all_results[f"{level}__{variable}"] = {
            "global": global_stats,
            "cutout": cutout_stats,
            "timeseries": time_series_stats,
        }

        # === Save flobal stats for training transforms ===
        if cfg.get("statistics", {}).get("save_global_pixel_stats", False):
            current_stats_save_path = os.path.join(stats_save_dir, var_cfg.get("model", ""), variable, split)
            logger.info(f"Saving global statistics to: {current_stats_save_path}")
            save_global_stats(
                raw_data,
                variable=variable,
                model=var_cfg.get("model", ""),
                domain=var_cfg.get("domain_size", ""),
                crop_region=var_cfg.get("crop_region", []),
                cfg=cfg,
                stats_save_path=current_stats_save_path,
            )
        
        if cfg.get("plotting", {}).get("save_cutout_example", False):
            current_fig_save_path = os.path.join(fig_save_dir, var_cfg.get("model", ""), variable, split)
            logger.info(f"Saving cutout example for variable {variable}")
            plot_cutout_example(raw_data, variable, cfg, current_fig_save_path)

        if cfg.get("plotting", {}).get("visualize_data", False):
            current_fig_save_path = os.path.join(fig_save_dir, var_cfg.get("model", ""), variable, split)
            logger.info(f"Generating raw data visualizations at {current_fig_save_path}")
            visualize_statistics(
                variable,
                raw_data,
                {
                    "global": global_stats,
                    "cutout": cutout_stats,
                    "timeseries": time_series_stats,
                },
                cfg,
                fig_save_path=current_fig_save_path,
                aggregated=False,
                show_transformed=True,
            )

        if cfg.get("statistics", {}).get("aggregate", False):
            logger.info(f"Computing aggregated statistics for {level.upper()} variable '{variable}' using method: {agg_method}")
            global_stats, cutout_stats, time_series_stats = compute_statistics(
                                                                    raw_data,
                                                                    aggregate=True,
                                                                    agg_method=agg_method,
                                                                    agg_time=cfg.get("data", {}).get("aggregation_time", "monthly"),
                                                                    return_all=True
                                                                    )
            all_results[f"{level}__agg__{variable}"] = {
                "global": global_stats,
                "cutout": cutout_stats,
                "timeseries": time_series_stats,
            }

            if cfg.get("plotting", {}).get("visualize_aggregated", False):
                current_fig_save_path = os.path.join(fig_save_dir, var_cfg.get("model", ""), variable, split)
                logger.info(f"Generating aggregated data visualizations at {current_fig_save_path}")
                visualize_statistics(
                    variable,
                    raw_data,
                    {
                        "global": global_stats,
                        "cutout": cutout_stats,
                        "timeseries": time_series_stats,
                    },
                    cfg,
                    fig_save_path=current_fig_save_path,
                    aggregated=True,
                    agg_method=agg_method,
                    show_transformed=True,
                )

    # === Process HR variables ===
    for variable, agg_method in zip(cfg.get("highres", {}).get("variables", []), cfg.get("highres", {}).get("agg_methods", [])):
        logger.info(f"\nProcessing highres variable: {variable} ({cfg.get('highres', {}).get('model', '')})| agg_method={agg_method}\n")
        process_variable(cfg.get("highres", {}), variable, agg_method, level="highres")

    # === Process LR variables ===
    for variable, agg_method in zip(cfg.get("lowres", {}).get("condition_variables", []), cfg.get("lowres", {}).get("agg_methods", [])):
        logger.info(f"\nProcessing lowres variable: {variable} ({cfg.get('lowres', {}).get('model', '')})| agg_method={agg_method}\n")
        process_variable(cfg.get("lowres", {}), variable, agg_method, level="lowres")


    logger.info("Finished running full data statistics pipeline. Final results:")
    for key, value in all_results.items():
        if key in cfg.get("statistics", {}).get("print_results", []):
            for inner_key, inner_value in value.items():
                logger.info(f" - {key} | {inner_key}: {inner_value}")
        else:
            logger.info(f" - {key}: (not printed, not in print_results list)")

    return all_results