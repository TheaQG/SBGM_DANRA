import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from sbgm.utils import get_units, get_cmaps, get_cmap_for_variable, get_unit_for_variable

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def plot_cutout_example(data,
                        variable,
                        cfg,
                        fig_save_path):
    """
        Plot a single cutout example from the data (2D field).
        Can plot either a random cutout or the one corresponding to a specified date

        Args:
            data (dict): with keys 'cutouts' (list of 2D np.ndarrays) and optionally 'timestamps' (list of datetime objects)
            variable (str): name of the variable (e.g. "temp", "prcp")
            cfg (dict): Configuration dictionary 
    """

    cutouts = data["cutouts"]
    timestamps = data.get("timestamps", None)

    # === Select index ===
    specific_date = cfg.get("plotting", {}).get("example_date", None)
    if specific_date:
        specific_date = str(specific_date)
        match_idx = [i for i, ts in enumerate(timestamps) if ts.strftime("%Y%m%d") == specific_date] # Match YYYYMMDD format
        if not match_idx:
            logger.warning(f"Specified example_date {specific_date} not found in timestamps. Using random cutout instead.")
            idx = np.random.randint(len(cutouts))
        else:
            idx = match_idx[0]
            logger.info(f"Found matching date {specific_date} at index {idx}.")
    else:
        idx = np.random.randint(len(cutouts))
        logger.info(f"No specific date provided. Using random index {idx}.")
    
    cutout = cutouts[idx]

    cmap = get_cmap_for_variable(variable)
    unit = get_unit_for_variable(variable)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cutout, cmap=cmap)
    ax.invert_yaxis()
    ax.set_title(f"{variable} on {timestamps[idx].strftime('%Y-%m-%d') if timestamps else 'N/A'}", fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(unit)

    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    # === Save or show ===
    plotting = cfg.get("plotting", {})
    show = plotting.get("show", False)
    save = plotting.get("save", True)

    if save:
        os.makedirs(fig_save_path, exist_ok=True)
        save_path = os.path.join(fig_save_path, f"example_cutout_{variable}_{idx}.png")
        logger.info(f"Saving example cutout plot to {save_path}")
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved cutout plot to {save_path}/example_cutout_{variable}_{idx}.png")
    
    if show:
        plt.show()
    plt.close(fig)


def visualize_statistics(variable,
                         data,
                         stats_dict,
                         cfg,
                         fig_save_path,
                         aggregated=False,
                         agg_method=None,
                         show_transformed=False
                         ):
    """
        Visualize dataset statistics and distributions.
        Args:
            data: list of 2D np.ndarrays (cutouts) or 3D np.ndarrays (stacks; N, H, W)
            stats_dict: dict of per-timestep statistics (keys: mean, std, min, max etc.)
            cfg: configuration dictionary
            aggregated: boolean indicating if the statistics are aggregated
            agg_method: aggregation method used (if any)
            show_transformed: boolean indicating if the transformed data should be shown (i.e. the standardize/normalized)
    """
    
    plotting = cfg.get("plotting", {})
    show = plotting.get("show", False)
    save = plotting.get("save", True)
    fig_path = fig_save_path

    if save:
        os.makedirs(fig_path, exist_ok=True)

    suffix = f"_agg_{agg_method}" if aggregated and agg_method else "_daily"
    agg_str = f"{agg_method} aggregated " if aggregated and agg_method else "daily "

    # Prepare data stack
    if isinstance(data, dict) and "cutouts" in data:
        cutouts = data["cutouts"]
    elif isinstance(data, list):
        cutouts = data    
    else:
        raise ValueError("Unexpected data format. Provide a dict with 'cutouts' key or a list of cutouts.")

    stack = np.stack(cutouts)  # Shape: (T, H, W)
    flat = stack.flatten() # Shape: (T * H * W,)


    # === 1. Time series plots ===
    if "timeseries" in stats_dict:
        fig, ax = plt.subplots(figsize=(10, 5))
        for k in plotting.get("plot_stats", ['mean']):
            if k in stats_dict["timeseries"]:
                ax.plot(stats_dict["timeseries"][k], label=k)
        ax.set_title(f"Time Series of {variable} statistics, {agg_str}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        if save:
            logger.info(f"Saving time series plot to {fig_path}/time_series_stats{suffix}.png")
            fig.savefig(os.path.join(fig_path, f"time_series_stats{suffix}.png"), dpi=300)
        if show:
            plt.show()
        plt.close(fig)


    # === 2. Histogram of all pixel values ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Main histogram
    ax1.hist(flat, bins=100, color='blue', alpha=0.7, label="Raw")

    transformed = None
    if show_transformed and "global" in stats_dict:
        if "mean" in stats_dict["global"] and "std" in stats_dict["global"]:
            mean_val = np.mean(stats_dict["global"]["mean"])
            std_val = np.mean(stats_dict["global"]["std"])
            transformed = (flat - mean_val) / std_val # !!!!! IMPLEMENT TRANSFORMATIONS FROM SPECIAL_TRANSFORMS !!!!!
            ax1.hist(transformed, bins=100, color='orange', alpha=0.5, label="Z-Score")

        log_flat = np.log(flat[flat > 0] + 1e-8)
        ax1.hist(log_flat, bins=100, color='green', alpha=0.5, label="Log")
    ax1.set_title(f"Histogram of {variable} pixel values, {agg_str}")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    if show_transformed and transformed is not None:
        # Zoomed histogram (of transformed data)
        ax2.hist(transformed, bins=100, color='orange', alpha=0.5, label="Z-Score")
        ax2.set_title(f"Zoomed Histogram of {variable} pixel values, {agg_str}")
        ax2.set_xlabel("Pixel Value")
        ax2.set_ylabel("Frequency")
        ax2.legend()

    if save:
        logger.info(f"Saving histogram plot to {fig_path}/histogram_pixels_{variable}_{suffix}.png")
        fig.savefig(os.path.join(fig_path, f"histogram_pixels_{variable}_{suffix}.png"), dpi=300)
    if show:
        plt.show()
    plt.close(fig)


    # === 3. Histogram of time-series statistics ===
    if "timeseries" in stats_dict:
        keys = plotting.get("plot_stats", ['mean', 'std', 'min', 'max', 'median', 'percentile_25', 'percentile_75'])

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        for ax, k in zip(axs.flatten(), keys):
            if k in stats_dict["timeseries"]:
                ax.hist(stats_dict["timeseries"][k], bins=100, alpha=0.7, label=k)
                ax.set_title(f"Histogram of {variable} {k}, {agg_str}")

        fig.tight_layout()
        if save:
            logger.info(f"Saving histogram of time-series stats to {fig_path}/histogram_time_series_{variable}_{suffix}.png")
            fig.savefig(os.path.join(fig_path, f"histogram_time_series_{variable}_{suffix}.png"), dpi=300)
        if show:
            plt.show()
        plt.close(fig)


    # # === 4. Global summary bar plot ===
    # values = [np.mean(stats_dict[k]) for k in keys if k in stats_dict]
    # labels = [k for k in keys if k in stats_dict]
    
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.bar(labels, values, color='skyblue', alpha=0.7)
    # ax.set_title(f"Global Summary of {variable}, {agg_str}")

    # if save:
    #     fig.savefig(os.path.join(fig_path, f"global_summary_{variable}_{suffix}.png"), dpi=300)
    # if show:
    #     plt.show()
    # plt.close(fig)


def visualize_data(data, stats_dict, cfg, aggregated=False):
    """
        Visualize the data and statistics.
        Time-series, histograms, pixel-wise distributions, etc.
    """
    pass