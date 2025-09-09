import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def plot_temporal_series(hr_series, lr_series, dates, variable1, variable2, model1, model2, save_path=None, show=False):
    """
        Plot temporal series of HR and LR data.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, hr_series, label=f'{variable1} ({model1}) (mean)', marker='o', markersize=3)
    ax.plot(dates, lr_series, label=f'{variable2} ({model2}) (mean)', marker='x', markersize=3)
    ax.set_xlabel('Date')
    ax.set_ylabel(f"{variable1} / {variable2}")
    ax.set_title(f"Temporal correlation of spatial mean {variable1} / {variable2} | {model1} vs {model2}")
    # Add a text box with correlation coefficient
    if len(hr_series) == len(lr_series) and len(hr_series) > 1:
        corr_coef = np.corrcoef(hr_series, lr_series)[0, 1]
        textstr = f'Correlation: {corr_coef:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

    ax.legend()
    ax.grid(True)
    if save_path:
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved temporal series plot to {save_path}")
    if show:
        plt.show()
    plt.close()

def plot_correlation_map(corr_map, variable1, variable2, model1, model2, save_path=None, show=False):
    """
        Plot spatial correlation map.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr_map, cmap='RdBu_r') #, vmin=-1, vmax=1)
    ax.set_title(f"Spatial correlation map of {variable1} / {variable2} | {model1} vs {model2}")
    ax.invert_yaxis()
    fig.colorbar(cax, ax=ax, label='Correlation coefficient')
    plt.axis('off')
    if save_path:
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved correlation map plot to {save_path}")
    if show:
        plt.show()
    plt.close()