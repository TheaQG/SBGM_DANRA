"""
    Main comparison module to orchestrate different types of data comparisons.
    Includes:
        - Field comparisons (single-day)
        - Time series comparisons (multi-day)
        - Distributional comparisons (full dataset)
"""

import logging
from data_analysis_pipeline.comparison.comparison_pipeline import run_comparison_pipeline
import copy 
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def comparison_main(cfg):
    """
        Entry point for data split + Zarr conversion. Run this from launch_splits.py
    """
    # === Extract variables, modes, and other settings from cfg === #
    variables = cfg.comparison.get("variables", [cfg.comparison.get("variable", "temp")])
    transformations = cfg.comparison.get("transformations", {var: "zscore" for var in variables})
    mode_blocks = cfg.comparison.get("modes", [{"mode": cfg.comparison.get("mode", "distribution")}])
    base_save_path = cfg.comparison.get("save_path", "./figures/comparison")

    for variable in variables:
        for mode_cfg in mode_blocks:
            # Create a deep copy of the full cfg for isolation
            cfg_copy = copy.deepcopy(cfg)

            # Set variable and mode
            cfg_copy.comparison.variable = variable
            cfg_copy.comparison.mode = mode_cfg["mode"]

            # Apply per-variable transformations (or fallback to none)
            if isinstance(transformations, dict):
                cfg_copy.comparison.transformations = transformations.get(variable, "none")

            # Merge optional mode-level comparison settings (e.g. date, comparison_type)
            for k, v in mode_cfg.items():
                cfg_copy.comparison[k] = v

            # Set a variable-specific save path subfolder
            if base_save_path:
                cfg_copy.comparison.save_path = os.path.join(base_save_path, variable)
                os.makedirs(cfg_copy.comparison.save_path, exist_ok=True)


            logger.info(f"[INFO] Running comparison for variable: {variable} | mode: {mode_cfg['mode']}")
            run_comparison_pipeline(cfg_copy)