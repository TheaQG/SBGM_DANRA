import os
import json
import numpy as np


def save_global_stats(data_dict,
                      domain,
                      crop_region,
                      cfg,
                      stats_save_path
                      ):
    """
        Compute global pixel-wise statistics over the stack of cutouts
        and save them for use in training normalization.
    """
    save_dir = stats_save_path
    os.makedirs(save_dir, exist_ok=True)

    cutouts = data_dict["cutouts"]
    stacked = np.stack([x.values for x in cutouts]) # Shape: (T, H, W)

    global_mean = np.mean(stacked, axis =0)
    global_std = np.std(stacked, axis =0)
    global_min = np.min(stacked, axis =0)
    global_max = np.max(stacked, axis =0)

    # To avoid issues with log(0), we add a small constant
    if global_min >= 0:
        log_stack = np.log(stacked + 1e-8)
        log_mean = np.mean(log_stack, axis=0)
        log_std = np.std(log_stack, axis=0)
        log_min = np.min(log_stack, axis=0)
        log_max = np.max(log_stack, axis=0)
    else:
        log_mean = log_std = log_min = log_max = None


    stats = {
        "mean": global_mean,
        "std": global_std,
        "min": global_min,
        "max": global_max,
        "log_mean": log_mean,
        "log_std": log_std,
        "log_min": log_min,
        "log_max": log_max
    }

    variable = data_dict.get("variable", "unknown")
    model = data_dict.get("model", "unknown")
    split = cfg.get("data", {}).get("split", "unknown")
    crop_region_str = "_".join(map(str, crop_region))

    filename = f"global_stats__{model}__{domain}__crop__{crop_region_str}__{variable}__{split}.json"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'w') as f:
        json.dump({k: v.tolist() for k, v in stats.items()}, f)

    print(f"[INFO] Global statistics saved to {filepath}")

def load_global_stats():
    pass