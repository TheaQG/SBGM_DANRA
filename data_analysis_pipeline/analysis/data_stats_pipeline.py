import os
import logging

from data_analysis_pipeline.analysis.data_loading import DataLoader
from data_analysis_pipeline.analysis.plotting import plot_cutout_example, visualize_statistics
from data_analysis_pipeline.analysis.transformations import get_transforms_from_stats, get_backtransforms_from_stats
from data_analysis_pipeline.analysis.statistics import compute_statistics, aggregate_data
from data_analysis_pipeline.analysis.global_stats import save_global_stats

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

    n_workers = int(cfg.get("data", {}).get("n_workers", 1))

    logger.info(f"Running data statistics with split='{split}', data_dir='{data_dir}', n_workers={n_workers}")
    
    # Process high-resolution variables
    highres_cfg = cfg.get("highres", {})
    highres_vars = highres_cfg.get("variables", [])
    highres_agg_methods = highres_cfg.get("agg_methods", [])
    highres_crop_region_str = "_".join(map(str, highres_cfg.get("crop_region", [])))
    for variable, agg_method in zip(highres_vars, highres_agg_methods):
        logger.info(f"Processing HR variable: {variable} ({highres_cfg.get('model', '')})| agg_method={agg_method}")
        # Load the data
        loader = DataLoader(base_dir=data_dir,
                            n_workers=n_workers,
                            variable=variable,
                            model=highres_cfg.get("model", ""),
                            domain_size=highres_cfg.get("domain_size", []),
                            split=split,
                            crop_region=highres_cfg.get("crop_region", []),
                            subdir="highres")
        raw_data = loader.load()
        logger.info(f"Loaded {len(raw_data['cutouts'])} cutouts for {variable}")


        # Compute basic statistics
        basic_stats = compute_statistics(raw_data)
        all_results["highres__" + variable] = basic_stats
        logger.info(f"Computed basic statistics for HR variable '{variable}'")

        # Save global statistics if specified
        if cfg.get("statistics", {}).get("save_global_pixel_stats", False):
            # Create a directory based on stats_save_dir/model/variable/split
            current_stats_save_path = os.path.join(stats_save_dir, highres_cfg.get("model", ""), variable, split)
            logger.info(f"Saving global statistics to: {current_stats_save_path}")
            save_global_stats(raw_data,
                            domain=highres_cfg.get("domain_size", ""),
                            crop_region=highres_cfg.get("crop_region", []),
                            cfg=cfg,
                            stats_save_path=current_stats_save_path
                            )
            
        # Load the global stats for the transformation
        # !!! NEEDS IMPLEMENTATION

        # Plot and save if specified
        if cfg.get("plotting", {}).get("save_cutout_example", False):
            logger.info(f"Saving cutout example for variable {variable}")
            plot_cutout_example(raw_data, variable, cfg)
        if cfg.get("plotting", {}).get("visualize_data", False):
            current_fig_save_path = os.path.join(fig_save_dir, highres_cfg.get("model", ""), variable, split)
            logger.info(f"Generating raw data visualizations at {current_fig_save_path}")
            visualize_statistics(variable,
                                 raw_data,
                                 basic_stats,
                                 cfg,
                                 fig_save_path=current_fig_save_path,
                                 aggregated=False,
                                 show_transformed=True
                                 )

        # Compute aggregated statistics and plot if specified
        if cfg.get("statistics", {}).get("aggregate", False):
            logger.info(f"Computing aggregated statistics for HR variable '{variable}' using method: {agg_method}")
            aggregated_stats = compute_statistics(raw_data, aggregate=True, agg_method=agg_method)
            all_results["highres__agg__" + variable] = aggregated_stats
            if cfg.get("plotting", {}).get("visualize_aggregated", False):
                current_fig_save_path = os.path.join(fig_save_dir, highres_cfg.get("model", ""), variable, split)
                logger.info(f"Generating aggregated data visualizations at {current_fig_save_path}")
                visualize_statistics(variable,
                                     raw_data,
                                     aggregated_stats,
                                     cfg,
                                     fig_save_path=current_fig_save_path,
                                     aggregated=True,
                                     agg_method=agg_method,
                                     show_transformed=True
                                     )

    # Process high-resolution variables
    lowres_cfg = cfg.get("lowres", {})
    lowres_vars = lowres_cfg.get("variables", [])
    lowres_agg_methods = lowres_cfg.get("agg_methods", [])
    for variable, agg_method in zip(lowres_vars, lowres_agg_methods):
        logger.info(f"Processing LR variable: {variable} ({lowres_cfg.get('model', '')})| agg_method={agg_method}")
        # Load data
        loader = DataLoader(base_dir=data_dir,
                            n_workers=n_workers,
                            variable=variable,
                            model=lowres_cfg.get("model", ""),
                            domain_size=lowres_cfg.get("domain_size", []),
                            split=split,
                            crop_region=lowres_cfg.get("crop_region", []),
                            subdir="lowres")
        raw_data = loader.load()
        logger.info(f"Loaded {len(raw_data['cutouts'])} cutouts for {variable}")

        # Compute basic statistics and save in variable
        basic_stats = compute_statistics(raw_data)
        all_results["lowres__" + variable] = basic_stats
        logger.info(f"Computed basic statistics for LR variable '{variable}'")

        # Save global statistics if specified
        if cfg.get("statistics", {}).get("save_global_pixel_stats", False):
            current_stats_save_path = os.path.join(stats_save_dir, lowres_cfg.get("model", ""), variable, split)
            logger.info(f"Saving global statistics to: {current_stats_save_path}")
            save_global_stats(raw_data,
                              domain=lowres_cfg.get("domain_size", ""),
                              crop_region=lowres_cfg.get("crop_region", []),
                              cfg=cfg,
                              stats_save_path=current_stats_save_path
                              )

        if cfg.get("plotting", {}).get("save_cutout_example", False):
            logger.info(f"Saving cutout example for variable {variable}")
            plot_cutout_example(raw_data, variable, cfg)
        if cfg.get("plotting", {}).get("visualize_data", False):
            current_fig_save_path = os.path.join(fig_save_dir, lowres_cfg.get("model", ""), variable, split)
            logger.info(f"Generating raw data visualizations at {current_fig_save_path}")
            visualize_statistics(variable,
                                 raw_data,
                                 basic_stats,
                                 cfg,
                                 fig_save_path=current_fig_save_path,
                                 aggregated=False,
                                 show_transformed=True
                                 )

        if cfg.get("statistics", {}).get("aggregate", False):
            logger.info(f"Computing aggregated statistics for LR variable '{variable}' using method: {agg_method}")
            aggregated_stats = compute_statistics(raw_data, aggregate=True, agg_method=agg_method)
            all_results["lowres__agg__" + variable] = aggregated_stats
            if cfg.get("plotting", {}).get("visualize_aggregated", False):
                current_fig_save_path = os.path.join(fig_save_dir, lowres_cfg.get("model", ""), variable, split)
                logger.info(f"Generating aggregated data visualizations at {current_fig_save_path}")
                visualize_statistics(variable,
                                     raw_data,
                                     aggregated_stats,
                                     cfg,
                                     fig_save_path=current_fig_save_path,
                                     aggregated=True,
                                     agg_method=agg_method,
                                     show_transformed=True
                                     )

    logger.info("Finished running full data statistics pipeline. Final results:")
    for key, value in all_results.items():
        logger.info(f" - {key}: {value}")
    return all_results