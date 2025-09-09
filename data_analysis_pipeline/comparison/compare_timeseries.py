"""
Time series comparison module (pandas-free).
  - Aggregate per-day differences: bias over time, variance differences, time-varying correlations
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from data_analysis_pipeline.comparison.compare_fields import compute_field_stats
from typing import Dict, Union

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def compute_daily_metrics_over_time(dict_data1, dict_data2):
    """
    Accepts dictionaries with keys "cutouts" and "timestamps" for two datasets.
    Computes metrics for each matching day and returns a timeseries list of dicts
    """
    timeseries = []

    timestamps1 = dict_data1.get("timestamps", [])
    timestamps2 = dict_data2.get("timestamps", [])

    shared_dates = sorted(set(timestamps1) & set(timestamps2))

    for i, date in enumerate(shared_dates):
        try:
            idx1 = timestamps1.index(date)
            idx2 = timestamps2.index(date)
        except ValueError as e:
            logger.warning(f"Date {date} not found in both datasets: {e}")
            continue

        data1 = dict_data1["cutouts"][idx1]
        data2 = dict_data2["cutouts"][idx2]

        stats = compute_field_stats(data1, data2)
        stats['date'] = date
        timeseries.append(stats)

    return timeseries


def plot_daily_metrics_over_time(timeseries, save_path='./figures', title="Time Series", fname='daily_metrics', show=False):
    """
    Plots time series of each metric over time in subplots.
    """
    if not timeseries or 'date' not in timeseries[0]:
        raise ValueError("Timeseries must contain 'date' as a key in each dictionary.")

    # Extract metrics and dates
    metrics = [k for k in timeseries[0] if k != 'date']
    n_metrics = len(metrics)
    dates = [entry['date'] for entry in timeseries]
    values = {metric: [entry[metric] for entry in timeseries] for metric in metrics}

    # Create subplot grid
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        axs[i].plot(dates, values[metric], marker='o', linestyle='-')
        axs[i].set_title(f"{metric}")
        axs[i].set_xlabel('Date')
        axs[i].set_ylabel(metric)
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].grid(True)

    # Hide unused subplots
    for j in range(n_metrics, len(axs)):
        fig.delaxes(axs[j])

    if save_path:
        plt.savefig(f"{save_path}/{fname}_timeseries.png")
        logger.info(f"      Saved time series plot to {save_path}/{fname}_timeseries.png")

    if show:
        plt.show()
    plt.close()


def compare_over_time(dict_data1, dict_data2, model1, model2, variable, save_path='./figures', show=False):
    """
    Computes and plots daily metrics over time.
    Returns summary statistics (mean and std) for each metric.
    """
    timeseries = compute_daily_metrics_over_time(dict_data1, dict_data2)

    if show or save_path:
        title = f"Daily Metrics Over Time: {variable} ({model1} vs {model2})"
        fname = f"{variable}_{model1}_vs_{model2}"
        plot_daily_metrics_over_time(timeseries, title=title, fname=fname, save_path=save_path, show=show)

    # Compute summary stats
    summary_stats = {}
    for entry in timeseries:
        for k, v in entry.items():
            if k == 'date':
                continue
            summary_stats.setdefault(f'{k}_values', []).append(v)

    metrics = [k for k in timeseries[0] if k != 'date']
    summary_stats = {}

    for metric in metrics:
        values = [entry[metric] for entry in timeseries]
        summary_stats[f"{metric}_mean"] = np.mean(values)
        summary_stats[f"{metric}_std"] = np.std(values)

    return summary_stats