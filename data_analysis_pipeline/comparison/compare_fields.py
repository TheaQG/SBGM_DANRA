"""
    Single-day field comparison module.
        - Field-level: Bias, RMSE, correlation, etc.
        - Spatial: Error maps, differen fields, ratio maps
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from sbgm.variable_utils import get_cmap_for_variable, get_unit_for_variable

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def compute_field_stats(data_model1, data_model2, mask=None):
    """
    Compute basic statistics between two 2D data fields.
    
    Parameters:
        data_model1 (np.ndarray): First data field.
        data_model2 (np.ndarray): Second data field.
        mask (np.ndarray, optional): Mask to apply to the data fields.
        
    Returns:
        dict: Dictionary containing bias, RMSE, and correlation.
    """

    if mask is not None:
        data_model1 = data_model1[mask]
        data_model2 = data_model2[mask]

    diff = data_model1 - data_model2
    return {
        'bias': np.mean(diff),
        'rmse': np.sqrt(np.mean(diff**2)),
        'corr': np.corrcoef(data_model1.flatten(), data_model2.flatten())[0, 1],
        'std_diff': np.std(diff)
    }

def plot_difference_map(
            data_model1,
            data_model2,
            model1="HR model",
            model2="LR model",
            variable="variable",
            title='Difference Map',
            save_path=None,
            show=False):
    """
    Plot the difference map between two data fields and the two fields themselves.
    
    Parameters:
        data_model1 (np.ndarray): First data field.
        data_model2 (np.ndarray): Second data field.
        title (str): Title of the plot.
        save_path (str, optional): Path to save the plot. If None, the plot is shown instead.
    """
    diff_map = data_model1 - data_model2
    unit = get_unit_for_variable(variable)

    # Shared vmin/vmax for HR/LR panels
    finite1 = np.asarray(data_model1)[np.isfinite(data_model1)]
    finite2 = np.asarray(data_model2)[np.isfinite(data_model2)]
    if finite1.size and finite2.size:
        vmin = min(finite1.min(), finite2.min())
        vmax = max(finite1.max(), finite2.max())
    else:
        vmin, vmax = None, None

    # Symmetric range for the difference
    finite_diff = np.asarray(diff_map)[np.isfinite(diff_map)]
    dmax = float(np.max(np.abs(finite_diff))) if finite_diff.size else None

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs = axs.ravel()  # Flatten

    panels = [
        (data_model1, f'{variable} - {model1}', get_cmap_for_variable(variable), (vmin, vmax)),
        (data_model2, f'{variable} - {model2}', get_cmap_for_variable(variable), (vmin, vmax)),
        (diff_map, f'{variable} - Difference ({model1} - {model2})', 'bwr', (-dmax, dmax) if dmax is not None else (None, None))
    ]

    for ax, (arr, ttl, cmap, lims) in zip(axs, panels):
        vmin_i, vmax_i = lims
        im = ax.imshow(arr, cmap=cmap, vmin=vmin_i, vmax=vmax_i, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ttl)

        # Compact, consistent colorbars using axes_grid1 (same width across panels)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax, orientation="vertical")
        cb.set_label(f"[{unit}]" if unit else "")

    fig.suptitle(title, fontsize=16)
    
    if save_path:
        out = f"{save_path}/{variable}_{model1}_vs_{model2}_difference_map.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        logger.info(f"      Saved difference map to {out}")
    if show:
        plt.show()
    plt.close()


def compare_single_day_fields(
                    data_model1,
                    data_model2,
                    mask=None,
                    variable="variable",
                    model1="Model 1",
                    model2="Model 2",
                    save_path=None,
                    show=False,
                    print_results=True):
    """
    Compare two 2D fields from the same date.
    Supports inputs as either raw arrays or {'cutout': ..., 'timestamp': ...} dicts.
    """

    # === DICT-WRAPPED INPUTS ===
    timestamp1 = None
    timestamp2 = None
    if isinstance(data_model1, dict):
        timestamp1 = data_model1.get("timestamp", None)
        data_model1 = data_model1.get("cutouts", data_model1)
    if isinstance(data_model2, dict):
        timestamp2 = data_model2.get("timestamp", None)
        data_model2 = data_model2.get("cutouts", data_model2)

    if timestamp1 and timestamp2 and timestamp1 != timestamp2:
        logger.warning(f"Comparing fields with different timestamps: {timestamp1} vs {timestamp2}")

    date_str = timestamp1.strftime("%Y%m%d") if timestamp1 else None

    # === COMPUTE STATS ===
    stats = compute_field_stats(data_model1, data_model2, mask)

    if print_results:
        logger.info(f"\nComparison stats for {variable} on {date_str if date_str else 'N/A'}:")
        logger.info(f"  Models: {model1} vs {model2}")
        logger.info(f"  Bias: {stats['bias']:.3f}, RMSE: {stats['rmse']:.3f}, Corr: {stats['corr']:.3f}, Std Diff: {stats['std_diff']:.3f}\n")

    # === PLOT DIFF MAP ===
    if show or save_path:
        title = f'{variable} | {model1} vs {model2} | {date_str if date_str else "N/A"}'
        plot_difference_map(
            data_model1=data_model1,
            data_model2=data_model2,
            model1=model1,
            model2=model2,
            variable=variable,
            title=title,
            save_path=save_path if save_path else None,
            show=show
            )

    return stats
    