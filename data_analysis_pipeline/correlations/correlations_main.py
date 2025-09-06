import logging
from data_analysis_pipeline.correlations.correlations_pipeline import run_data_correlations


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def correlations_main(cfg):
    """
        Entry point for data split + Zarr conversion. Run this from launch_splits.py
    """

    logger.info("[INFO] Running main data correlations")
    run_data_correlations(cfg)