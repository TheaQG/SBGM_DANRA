import os
import json
import logging
import numpy as np


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def save_global_stats(data_dict,
                      variable,
                      model,
                      domain,
                      crop_region,
                      cfg,
                      stats_save_path,
                      pool_pixels=True
                      ):
    """
        Compute global pixel-wise statistics over the stack of cutouts
        and save them for use in training normalization.
    """
    save_dir = stats_save_path
    os.makedirs(save_dir, exist_ok=True)

    cutouts = data_dict["cutouts"]
    # Gives the stats for each pixel position across time, returns shape (H, W)
    stacked = np.stack(cutouts) #np.stack([x.values for x in cutouts]) # Shape: (T, H, W)

    # Pool across pixels if specified, otherwise keep spatial dimensions
    if pool_pixels:
        stacked = stacked.flatten()  # Shape: (T * H * W,)

    global_mean = np.mean(stacked)
    global_std = np.std(stacked)
    global_min = np.min(stacked)
    global_max = np.max(stacked)

    # To avoid issues with log(0), we add a small constant
    if global_min >= 0:
        log_stack = np.log(stacked + 1e-8)
        log_mean = np.mean(log_stack)
        log_std = np.std(log_stack)
        log_min = np.min(log_stack)
        log_max = np.max(log_stack)
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

    split = cfg.get("data", {}).get("split", "unknown")
    crop_region_str = "_".join(map(str, crop_region))

    filename = f"global_stats__{model}__{domain}__crop__{crop_region_str}__{variable}__{split}.json"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'w') as f:
        for k, v in stats.items():
            if v is None:
                stats[k] = None
                logger.warning(f"{k} is None, saving as null in JSON.")
            else:
                stats[k] = float(v)
        json.dump(stats, f)


    print(f"[INFO] Global statistics saved to {filepath}")

def load_global_stats():
    pass



# I am still not getting what I need. I have the length of cutouts corresponding to number of days in the dataset, which prints correctly to "[INFO] Length of cutouts: 10958" (py-code: "logger.info(f"Length of cutouts: {len(cutouts)}")") and shape of a single cutout to "[INFO] Shape of single cutout: (180, 180)" (py-code: logger.info(f"Shape of single cutout: {cutouts[0].shape}").
# But then when I want to compute the statistics (i.e. of the whole stack of pixels), I still get a 2D shape out.