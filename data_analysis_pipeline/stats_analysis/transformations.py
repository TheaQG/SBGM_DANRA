import os
import json

from typing import Optional
from sbgm.special_transforms import (
    Scale,
    ScaleBackTransform,
    ZScoreTransform,
    ZScoreBackTransform,
    PrcpLogTransform,
    PrcpLogBackTransform
    )


def load_global_stats(filepath):
    """
        Load global statistics from a file.
    """
    with open(filepath, "r") as f:
        stats = json.load(f)
    
    
    return stats

from typing import Optional

def get_transforms_from_stats(transform_type: str,
                              cfg,
                              stats: Optional[dict] = None,
                              stats_file_path: str = '',
                              ):
    """
        Build transformations from stats, either given stats or given file path
        Must provide either stats or stats_file_path
    """
    if stats_file_path:
        print(f"[INFO] Loading stats from {stats_file_path}")
    if stats and stats_file_path:
        print(f"[WARNING] Both stats and stats_file_path provided, using provided stats.")
        stats_file_path = ''

    if stats is None and stats_file_path:
        if not os.path.exists(stats_file_path):
            raise ValueError(f"Stats file not found: {stats_file_path}")
        stats = load_global_stats(stats_file_path)
    if stats is None:
        raise ValueError(f"Failed to load stats from {stats_file_path}")

    if transform_type == "zscore":
        return ZScoreTransform(mean=stats["mean"], std=stats["std"])
    elif transform_type == "scale01":
        return Scale(0, 1, data_min_in=stats["min"], data_max_in=stats["max"])
    elif transform_type == "scale_minus1_1":
        return Scale(-1, 1, data_min_in=stats["min"], data_max_in=stats["max"])
    elif transform_type in ["log_zscore", "log_01", "log_minus1_1", "log"]:
        return PrcpLogTransform(scale_type=transform_type,
                                glob_mean_log=stats["log_mean"],
                                glob_std_log=stats["log_std"],
                                glob_min_log=stats["log_min"],
                                glob_max_log=stats["log_max"],
                                buffer_frac=cfg.get("data", {}).get("buffer_frac", 0.5))
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

def get_backtransforms_from_stats(transform_type: str,
                                  cfg,
                                  stats: Optional[dict] = None,
                                  stats_file_path: str = '',
                                  ):
    """
        Build backtransformations from stats, either given stats or given file path
        Must provide either stats or stats_file_path
    """
    if stats_file_path:
        print(f"[INFO] Loading stats from {stats_file_path}")
    if stats and stats_file_path:
        print(f"[WARNING] Both stats and stats_file_path provided, using provided stats.")
        stats_file_path = ''

    if stats is None and stats_file_path:
        if not os.path.exists(stats_file_path):
            raise ValueError(f"Stats file not found: {stats_file_path}")
        stats = load_global_stats(stats_file_path)
    if stats is None:
        raise ValueError(f"Failed to load stats from {stats_file_path}")

    if transform_type == "zscore":
        return ZScoreBackTransform(mean=stats["mean"], std=stats["std"])
    elif transform_type == "scale01":
        return ScaleBackTransform(0, 1, data_min_in=stats["min"], data_max_in=stats["max"])
    elif transform_type == "scale_minus1_1":
        return ScaleBackTransform(-1, 1, data_min_in=stats["min"], data_max_in=stats["max"])
    elif transform_type in ["log_zscore", "log_01", "log_minus1_1", "log"]:
        return PrcpLogBackTransform(scale_type=transform_type,
                                glob_mean_log=stats["log_mean"],
                                glob_std_log=stats["log_std"],
                                glob_min_log=stats["log_min"],
                                glob_max_log=stats["log_max"],
                                buffer_frac=cfg.get("data", {}).get("buffer_frac", 0.5)
                                )
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")