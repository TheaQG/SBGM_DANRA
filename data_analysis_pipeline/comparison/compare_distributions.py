"""
    Distributional comparison module.
        - Compare full data distributions: PDFs, CDFs, histograms, energy Power spectra
        - Include statistical tests: KS-test, Wasserstein distance, etc.
"""
import os
import logging
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from numpy.fft import fft2, fftshift
from collections import defaultdict
from typing import Optional
from sbgm.utils import get_unit_for_variable, get_color_for_variable

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def compute_2d_power_spectrum(
                data: np.ndarray,
                detrend: bool = False) -> np.ndarray:
    """
        Compute the 2D power spectrum of a 2D field.
        Optionally removes the mean or linear trend first
    """

    if detrend:
        data = data - np.mean(data)
    
    fft_data = fft2(data)
    power_spectrum = np.abs(fft_data)**2
    power_spectrum = fftshift(power_spectrum)  # Shift zero frequency to center
    return power_spectrum

def radial_average(ps_2d: np.ndarray) -> np.ndarray:
    """
        Compute the isotropic (radially averaged) 1D power spectrum from a 2D power spectrum.

    """
    y, x = np.indices(ps_2d.shape)
    center = np.array(ps_2d.shape) // 2
    r = np.hypot(x - center[1], y - center[0])
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), ps_2d.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)  # Avoid division by zero
    
    return radialprofile

def compare_power_spectra(
    data_model1: np.ndarray,
    data_model2: np.ndarray,
    model1: str = 'Model 1',
    model2: str = 'Model 2',
    dx_model1: float = 2.5,
    dx_model2: float = 2.5,
    variable: str = 'Variable',
    save_path: Optional[str]= None,
    loglog: bool = True,
    return_metrics: bool = True,
    show: bool = False
    ):
    """
        Compare radially averaged power spectra of two 2D datasets.
        - Low spatial frequencies corresponds to large-scale structures (waves, fronts, smooth gradients)
        - High spatial frequencies corresponds to small-scale structures (local turbulence, variability, topographic noise/effects)
    """
    # Compute radial power spectra (1D)
    ps1 = radial_average(compute_2d_power_spectrum(data_model1))
    ps2 = radial_average(compute_2d_power_spectrum(data_model2))

    title = f"{variable} | {model1} vs {model2} | "
    fname = f"{variable}_{model1}_vs_{model2}_power_spectrum".replace(" ", "_")

    # Frequency bins converted to wavelengths
    nx = data_model1.shape[1]
    logger.info(f"Data shape: {data_model1.shape}, nx: {nx}")
    dx = dx_model1  # grid spacing in km
    wavelengths = (nx * dx) / np.arange(1, len(ps1)+1, dtype=np.float64)  # Avoid division by zero
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    if loglog:
        ax.loglog(wavelengths, ps1, label=model1, color='blue')
        ax.loglog(wavelengths, ps2, label=model2, color='orange')
    else:
        ax.plot(wavelengths, ps1, label=model1, color='blue')
        ax.plot(wavelengths, ps2, label=model2, color='orange')

    ax.set_title(f'{title}Radially Averaged Power Spectrum Comparison')
    ax.set_xlabel('Wavelength (km)')
    ax.set_ylabel('Power Spectrum Density')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{fname}.png'), dpi=300)
    if show:
        plt.show()
    plt.close()

    # Compute evaluation metrics
    if return_metrics:
        eps = 1e-8  # Small constant to avoid division by zero
        # (1) Mean squared error
        mse_spec = np.mean((ps1 - ps2)**2) 
        # (2) Log mean squared error
        log_mse_spec = np.mean((np.log(ps1 + eps) - np.log(ps2 + eps))**2)
        # (3) Ratio spectrum
        ratio = ps2 / (ps1 + eps)


        metrics = {
            'mse_spectrum': float(mse_spec),
            'log_mse_spectrum': float(log_mse_spec),
            'mean_ratio': float(np.mean(ratio)),
            'std_ratio': float(np.std(ratio)),
            'min_ratio': float(np.min(ratio)),
            'max_ratio': float(np.max(ratio))
        }

        return metrics

def batch_compare_power_spectra(
    dataset1: dict,
    dataset2: dict,
    model1: str = 'Model 1',
    model2: str = 'Model 2',
    dx_model1: float = 2.5,
    dx_model2: float = 2.5,
    variable: str = 'Variable',
    show_plot: bool = False,
    loglog: bool = True,
    save_path: str = './figures/power_spectra_comparison',
    return_all_metrics: bool = True,
    show: bool = False
    ):
    """
        Compare power spectra for multiple field pairs over time
        Assumes dataset1 and dataset2 are dictionaries with time keys and 2D numpy array values (i.e. mapping date -> 2D field)

        Returns average metrics and optionally all daily metrics
    """
    title = f"{variable} | {model1} vs {model2} | "
    fname = f"{variable}_models_{model1}_vs_{model2}_power_spectrum".replace(" ", "_")

    all_metrics = defaultdict(list) # Store lists of metrics for each time point
    spectra_1 = []
    spectra_2 = []


    shared_dates = sorted(set(k for k in dataset1 if isinstance(dataset1[k], np.ndarray)) & set(k for k in dataset2 if isinstance(dataset2[k], np.ndarray)))
    logger.info(f"Found {len(shared_dates)} shared dates for batch power spectra comparison.")
    if not shared_dates:
        raise ValueError("No overlapping dates between the two datasets.")

    for date in shared_dates:
        field1 = dataset1[date]
        field2 = dataset2[date]

        # Compute 1D spectra
        ps1 = radial_average(compute_2d_power_spectrum(field1))
        ps2 = radial_average(compute_2d_power_spectrum(field2))
        spectra_1.append(ps1)
        spectra_2.append(ps2)

        # Mectrics
        eps = 1e-8
        mse_spec = np.mean((ps1 - ps2)**2)
        log_mse_spec = np.mean((np.log(ps1 + eps) - np.log(ps2 + eps))**2)
        ratio = ps2 / (ps1 + eps)

        all_metrics['date'].append(date)
        all_metrics['mse_spectrum'].append(float(mse_spec))
        all_metrics['log_mse_spectrum'].append(float(log_mse_spec))
        all_metrics['mean_ratio'].append(float(np.mean(ratio)))
        all_metrics['std_ratio'].append(float(np.std(ratio)))
        all_metrics['min_ratio'].append(float(np.min(ratio)))
        all_metrics['max_ratio'].append(float(np.max(ratio)))

    # Convert to arrays 
    spectra_1 = np.stack(spectra_1)
    spectra_2 = np.stack(spectra_2)

    # Mean +/- std spectra
    mean_ps1 = np.mean(spectra_1, axis=0)
    std_ps1 = np.std(spectra_1, axis=0)
    mean_ps2 = np.mean(spectra_2, axis=0)
    std_ps2 = np.std(spectra_2, axis=0)

    first_sample = next(v for v in dataset1.values() if isinstance(v, np.ndarray) and v.ndim == 2)
    nx = first_sample.shape[1]
    dx = dx_model1  # grid spacing in km
    logger.info(f"Data shape: {first_sample.shape}, nx: {nx}")
    logger.info(f"Grid spacing dx: {dx} km")
    wavelengths = (nx * dx) / np.arange(1, len(mean_ps1)+1, dtype=np.float64)  # Avoid division by zero

    # Plot mean spectra
    if show_plot or save_path:
        fig, ax = plt.subplots(figsize=(10, 6))
        if loglog:
            ax.loglog(wavelengths, mean_ps1, label=f'{model1} Mean', color='blue')
            ax.loglog(wavelengths, mean_ps2, label=f'{model2} Mean', color='orange')
            # ax.fill_between(wavelengths, mean_ps1 - std_ps1, mean_ps1 + std_ps1, color='blue', alpha=0.3)
            # ax.fill_between(wavelengths, mean_ps2 - std_ps2, mean_ps2 + std_ps2, color='orange', alpha=0.3)
        else:
            ax.plot(wavelengths, mean_ps1, label=f'{model1} Mean', color='blue', linewidth=2)
            ax.plot(wavelengths, mean_ps2, label=f'{model2} Mean', color='orange', linewidth=2)
            # ax.fill_between(wavelengths, mean_ps1 - std_ps1, mean_ps1 + std_ps1, color='blue', alpha=0.3)
            # ax.fill_between(wavelengths, mean_ps2 - std_ps2, mean_ps2 + std_ps2, color='orange', alpha=0.3)

        ax.set_title(f'{title}\nMean (over time) Radially Averaged Power Spectrum Comparison')
        ax.set_xlabel('Wavelength (km)')

        # Log-scale wavelength ticks (rounded powers of 2)
        tick_vals = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        tick_vals = [tv for tv in tick_vals if wavelengths.min() <= tv <= wavelengths.max()]
        ax.set_xticks(tick_vals)
        ax.set_xscale('log')
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.tick_params(axis='x', which='major', labelsize=10)


        # Shade region below Nyquist wavelength (2 * dx = 5 km)
        nyquist_limit = 2 * 2.5  # = 5 km
        ax.axvspan(wavelengths.min(), nyquist_limit, color='gray', alpha=0.2, label='Below Nyquist')

        # Reverse x-axis to show large scales on left
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[1], xlim[0])
        ax.set_ylabel('Power Spectrum Density')
        ax.legend()
        ax.grid(True, which='both', ls='--', alpha=0.5)

        important_scales = {
            'Large-scale front': 256,
            'Mesoscale': 64,
            'Convective': 8
        }

        for label, wl in important_scales.items():
            if wavelengths.min() <= wl <= wavelengths.max():
                ax.axvline(wl, linestyle='--', color='gray', alpha=0.5)
                ax.text(wl, ax.get_ylim()[1], label, rotation=90, va='top', ha='right', fontsize=8)

        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, f'{fname}.png'), dpi=300)
        if show:
            plt.show()
        plt.close()

    # Average metrics
    avg_metrics = {
        k: float(np.mean(v)) for k, v in all_metrics.items() if k != 'date'
    }
    std_metrics = {
        k: float(np.std(v)) for k, v in all_metrics.items() if k != 'date'
    }

    if return_all_metrics:
        return avg_metrics | std_metrics, all_metrics # '|' merges two dicts in Python 3.9+
    else:
        return avg_metrics | std_metrics







def compute_distribution_stats(data_model1, data_model2):
    """
        Return comparison statistics between flattened distributions
    """
    data1 = data_model1.flatten()
    data2 = data_model2.flatten()

    ks_stat, ks_pvalue = ks_2samp(data_model1.flatten(), data_model2.flatten())
    w_distance = wasserstein_distance(data_model1.flatten(), data_model2.flatten())
    
    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'wasserstein_distance': w_distance
    }

def plot_histograms(data_model1,
                    data_model2,
                    bins=100,
                    model1 ='Model 1',
                    model2 ='Model 2',
                    variable='Variable',
                    log=False,
                    save=True,
                    show=False,
                    save_path='./figures'):
    """
        Plot overlaid histograms of two datasets for visual comparison.
    """

    title = f"{variable} | {model1} vs {model2} | Histogram Comparison"
    fname = f"{variable}_models_{model1}_vs_{model2}_histogram".replace(" ", "_")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(data_model1.flatten(), bins=bins, alpha=0.5, label=f"{model1}, {variable}", density=True)
    ax.hist(data_model2.flatten(), bins=bins, alpha=0.5, label=f"{model2}, {variable}", density=True)
    
    if log:
        ax.set_yscale('log')
    
    ax.set_xlabel(f'{variable} ({get_unit_for_variable(variable)})')
    ax.set_ylabel('Density')
    ax.set_title(title)
    
    plt.legend()
    
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.savefig(f"{save_path}/{fname}", dpi=300)
    if show:
        plt.show()

def compare_distributions(
    data_model1: np.ndarray,
    data_model2: np.ndarray,
    model1: str = 'Model 1',
    model2: str = 'Model 2',
    variable: str = 'Variable',
    bins: int = 100,
    log_hist: bool = False,
    save_figures: bool = True,
    show: bool = False,
    save_path: str = './figures/distribution_comparison',
    return_metrics: bool = True
    ):
    """
        Wrapper function to compute and plot distribution comparison between two datasets.
        
        Parameters:
            data_model1 (np.ndarray): First data field.
            data_model2 (np.ndarray): Second data field.
            model1 (str): Name of the first model.
            model2 (str): Name of the second model.
            variable (str): Variable name for titles/labels.
            bins (int): Number of bins for histograms.
            log_hist (bool): Whether to use logarithmic scale for histogram y-axis.
            save_figures (bool): Whether to save the histogram figure.
            show (bool): Whether to display the histogram figure.
            save_path (str): Directory to save figures.
            return_metrics (bool): Whether to return computed statistics.

        Returns:
            dict: Dictionary containing KS statistic, p-value, and Wasserstein distance if return_metrics is True.
    """
    if show or save_figures:
        plot_histograms(
            data_model1,
            data_model2,
            bins=bins,
            model1=model1,
            model2=model2,
            variable=variable,
            log=log_hist,
            save=save_figures,
            show=show,
            save_path=save_path
        )
    
    if return_metrics:
        stats = compute_distribution_stats(data_model1, data_model2)
        return stats