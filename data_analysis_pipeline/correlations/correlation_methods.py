"""
    Module for computing correlations between high-resolution (HR) and low-resolution (LR) datasets.
    Implemented:
        - Temporal correlation (domain mean per day)
        - Spatial correlation (grid-point wise over time)

    TODO:
        - Lagged correlation analysis (temporal): to explore lead/lag relationships (i.e. if LR influences HR with a time delay) - use numpy.correlate or scipy.signal.correlate
        - Composite correlation maps: To understand spatial patterns associated with high/low values of one variable (similar to composites in climate science) - use numpy.where to select dates based on thresholds
        - Canonical correlation analysis (CCA): To identify pairs of linear combinations of LR and HR that are maximally correlated (sklearn.cross_decomposition.CCA)
        - Feature importance via Random Forest: Use ML to rank which LR variables are most predictive of HR variable (e.g. RandomForestRegressor feature_importances_)
        - Mutual information: To detect non-linear dependencies missed by correlation (use sklearn.feature_selection.mutual_info_regression)
        
"""

from scipy.stats import pearsonr, spearmanr
import numpy as np
import logging


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def compute_temporal_correlation(hr_data, lr_data, method='pearson'):
    """
    Compute correlation between HR and LR time series (domain mean per day).
    """
    hr_series = np.array([np.mean(hr_data[date]) for date in sorted(hr_data)])
    lr_series = np.array([np.mean(lr_data[date]) for date in sorted(lr_data)])

    if method == 'pearson':
        corr, _ = pearsonr(hr_series, lr_series)
    elif method == 'spearman':
        corr, _ = spearmanr(hr_series, lr_series)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    return {
        'correlation': corr,
        'series_hr': hr_series,
        'series_lr': lr_series
    }

def compute_spatial_correlation(hr_data, lr_data, method="pearson"):
    """ ¨
        Compute spatial (grid-point wise) correlation over time.
        Returns a 2D map of correlation values.
    """
    hr_stack = np.array([hr_data[date] for date in sorted(hr_data)])
    lr_stack = np.array([lr_data[date] for date in sorted(lr_data)])

    if hr_stack.shape != lr_stack.shape:
        raise ValueError("HR and LR data must have the same shape for spatial correlation.")
    
    T, H, W = hr_stack.shape
    corr_map = np.full((H, W), np.nan)

    for i in range(H):
        for j in range(W):
            x = hr_stack[:, i, j]
            y = lr_stack[:, i, j]
            if method == 'pearson':
                corr, _ = pearsonr(x, y)
            elif method == 'spearman':
                corr, _ = spearmanr(x, y)
            else:
                raise ValueError(f"Unknown method '{method}' for spatial correlation.")
            corr_map[i, j] = corr

    return corr_map