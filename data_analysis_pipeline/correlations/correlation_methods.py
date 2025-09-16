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
import math
from collections import defaultdict
from datetime import datetime

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


# === Seasonality helpers ===
def _month_id(dt): # (year,month)
    return (dt.year, dt.month)

def _week_id(dt): # ISO week (year, week)
    iso = dt.isocalendar()
    return (iso[0], iso[1])

def _year_id(dt): # year
    return dt.year

def build_climatology(ts, timestamps, method="monthly"):
    """
        ts: 1D array (time, )
        timestamps: list of datetime
        method: "monthly", "weekly", "yearly", or 'DOY' (day of year)
        returns: dict mapping season-key -> mean value
    """
    ts = np.asarray(ts)
    groups = defaultdict(list)

    if method == "monthly":
        keys = [_month_id(d.replace(year=2001)) for d in timestamps] # Use a dummy year to group by month
    # elif method == "weekly":
    #     keys = [_week_id(d.replace(year=2001)) for d in timestamps] # Use a dummy year to group by week
    # elif method == "yearly":
    #     keys = [_year_id(d) for d in timestamps]
    elif method == "DOY":
        # map Feb29 to Feb28 to avoid tiny groups
        keys = [(d.month, 28 if (d.month == 2 and d.day == 29) else d.day) for d in timestamps]
    else:
        raise ValueError(f"Unknown seasonality method: {method}")
    
    for k, v in zip(keys, ts):
        groups[k].append(v)
    return {k: float(np.nanmean(v)) for k, v in groups.items()}
    
def remove_seasonality_ts(ts, timestamps, method="monthly"):
    """
        Subtracts seasonal cycle (monthly or day-of-year) from a 1D time series.
    """
    ts = np.asarray(ts, dtype=float)
    clim = build_climatology(ts, timestamps, method=method)

    if method == "monthly":
        keys = [_month_id(d.replace(year=2001)) for d in timestamps]
    # elif method == "weekly":
    #     keys = [_week_id(d.replace(year=2001)) for d in timestamps]
    # elif method == "yearly":
    #     keys = [_year_id(d) for d in timestamps]
    else: # 'doy'
        keys = [(d.month, 28 if (d.month == 2 and d.day == 29) else d.day) for d in timestamps]
    
    anomalies = np.array([t - clim[k] for t,k in zip(ts, keys)], dtype=float)
    return anomalies

def remove_seasonality_stack(stack, timestamps, method="monthly"):
    """
        For 3D stacks shaped (time, y, x). Subtract seasonal cycle per-pixel.
    """
    T, H, W = stack.shape
    out = np.empty_like(stack, dtype=float)
    # build indices once
    if method == "monthly":
        keys = [_month_id(d.replace(year=2001)) for d in timestamps]
        key_set = sorted(set(keys))
    # elif method == "weekly":
    #     keys = [_week_id(d.replace(year=2001)) for d in timestamps]
    #     key_set = sorted(set(keys))
    # elif method == "yearly":
    #     keys = [_year_id(d) for d in timestamps]
    #     key_set = sorted(set(keys))
    else: # 'doy'
        keys = [(d.month, 28 if (d.month == 2 and d.day == 29) else d.day) for d in timestamps]
        key_set = sorted(set(keys))

    # pre-allocate masks per key for efficiency
    key_to_idx = {k: np.where(np.array(keys) == k)[0] for k in key_set}

    for y in range(H):
        row = stack[:, y, :]
        for x in range(W):
            ts = row[:, x].astype(float)
            # compute per-key means
            mu = {k: np.nanmean(ts[idx]) for k, idx in key_to_idx.items()}
            out[:, y, x] = np.array([ts[t] - mu[keys[t]] for t in range(T)], dtype=float)
    return out

# === Temporal aggregation ===
def aggregate_ts(ts, timestamps, freq="monthly", how="mean"):
    """
    freq: 'weekly' or 'monthly'
    how: 'mean' or 'sum'
    returns (agg_values, agg_dates) aligned for correlation/plotting
    """
    ts = np.asarray(ts, dtype=float)
    groups = defaultdict(list)
    for v, d in zip(ts, timestamps):
        key = _week_id(d) if freq == "weekly" else (d.year, d.month)
        groups[key].append(v)

    keys_sorted = sorted(groups.keys())
    agg_values, agg_dates = [], []
    for key in keys_sorted:
        vals = np.array(groups[key], dtype=float)
        agg_values.append(np.nanmean(vals) if how == "mean" else np.nansum(vals))
        # pick group mid-point for plotting
        if freq == "weekly":
            year, week = key
            # pick the Wednesday of that ISO week
            agg_dates.append(datetime.fromisocalendar(year, week, 3))
        else:
            y, m = key
            agg_dates.append(datetime(y, m, 15))
    return np.array(agg_values), agg_dates

# === Correlation helpers ===
def corrcoef_1d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 3:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0,1])

def spatial_corr_map(hr_stack, lr_stack, remove_seasonality=None):
    """
        hr_stack, lr_stack: (time, y, x)
        remove_seasonality: None | "monthly" | "doy" (later "weekly", "yearly")
        returns r_map (y, x)
    """
    if remove_seasonality:
        hr_stack = remove_seasonality_stack(hr_stack, timestamps=None, method=remove_seasonality) # timestamps must be captured in closure or passed as argument
        lr_stack = remove_seasonality_stack(lr_stack, timestamps=None, method=remove_seasonality)
    T, H, W = hr_stack.shape
    R = np.full((H, W), np.nan, dtype=float)
    # vecorized-ish loop over pixels
    for y in range(H):
        a = hr_stack[:, y, :]
        b = lr_stack[:, y, :]
        # compute correlation along time for each x in this row
        for x in range(W):
            R[y, x] = corrcoef_1d(a[:, x], b[:, x])
    return R
















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