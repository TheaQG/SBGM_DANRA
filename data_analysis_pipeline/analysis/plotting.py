import numpy as np
import matplotlib.pyplot as plt
import os



def plot_cutout_example(data, variable, cfg):
    """
        Plot an example cutout from the data.
    """
    pass


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
    # Convert data to ndarray
    if isinstance(data, list):
        stack = np.stack([d.values if hasattr(d, "values") else d for d in data])
    else:
        stack = data.values if hasattr(data, "values") else data

    plotting = cfg.get("plotting", {})
    show = plotting.get("show", False)
    save = plotting.get("save", True)
    fig_path = plotting.get("fig_save_dir", "./figures")

    if save:
        os.makedirs(fig_path, exist_ok=True)

    suffix = f"_agg_{agg_method}" if aggregated and agg_method else "_daily"
    agg_str = f"{agg_method} aggregated " if aggregated and agg_method else "daily "

    # === 1. Time series plots ===
    fig, ax = plt.subplots(figsize=(10, 5))
    for k in plotting.get("plot_stats", ['mean']):
        if k in stats_dict:
            ax.plot(stats_dict[k], label=k)
    ax.set_title(f"Time Series of {variable} statistics, {agg_str}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    if save:
        fig.savefig(os.path.join(fig_path, f"time_series_stats{suffix}.png"), dpi=300)
    if show:
        plt.show()
    plt.close(fig)


    # === 2. Histogram of all pixel values ===
    flat = stack.flatten()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Main histogram
    ax1.hist(flat, bins=100, color='blue', alpha=0.7, label="Raw")

    transformed = None
    if show_transformed:
        if "mean" in stats_dict and "std" in stats_dict:
            mean_val = np.mean(stats_dict["mean"])
            std_val = np.mean(stats_dict["std"])
            transformed = (flat - mean_val) / std_val
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
        fig.savefig(os.path.join(fig_path, f"histogram_pixels_{variable}_{suffix}.png"), dpi=300)
    if show:
        plt.show()
    plt.close(fig)


    # === 3. Histogram of time-series statistics ===
    keys = cfg.get("plotting", {}).get("plot_stats", ['mean', 'std', 'min', 'max'])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for ax, k in zip(axs.flatten(), keys):
        if k in stats_dict:
            ax.hist(stats_dict[k], bins=100, alpha=0.7, label=k)
            ax.set_title(f"Histogram of {variable} {k}, {agg_str}")

    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(fig_path, f"histogram_time_series_{variable}_{suffix}.png"), dpi=300)
    if show:
        plt.show()
    plt.close(fig)


    # === 4. Global summary bar plot ===
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [np.mean(stats_dict[k]) for k in keys if k in stats_dict]
    labels = [k for k in keys if k in stats_dict]
    ax.bar(labels, values, color='skyblue', alpha=0.7)

    ax.set_title(f"Global Summary of {variable}, {agg_str}")

    if save:
        fig.savefig(os.path.join(fig_path, f"global_summary_{variable}_{suffix}.png"), dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def visualize_data(data, stats_dict, cfg, aggregated=False):
    """
        Visualize the data and statistics.
        Time-series, histograms, pixel-wise distributions, etc.
    """
    pass