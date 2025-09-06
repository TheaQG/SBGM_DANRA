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

def run_data_correlations(cfg):
    pass