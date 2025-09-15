
import torch
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Union

from sbgm.utils import _squeeze_geo_value
from sbgm.variable_utils import get_units, get_cmaps, get_cmap_for_variable


# Set up logging
logger = logging.getLogger(__name__)


def plot_sample(sample,
                cfg,
                figsize=(15, 4)):
    """
        Plot a single sample (dictionary from the dataset class) in a consistent layout
        
        Expected keys in sample:
            - HR image: f"{var}_hr" (and optionally f"{var}_hr_original")
            - LR condition(s): keys ending with "_lr" (and optionally "_lr_original")
            - HR mask for ocean masking: "lsm_hr" (used only for HR images)
            - Extra keys (e.g. geo variables) if provided via extra keys

        Parameters:
            - sample: Dictionary containing the sample
            - cfg: Configuration dictionary containing model and variable information
            - figsize: Tuple for figure size

        Returns:
            - fig: The matplotlib Figure object
    """
    # Extract parameters from cfg
    hr_model = cfg['highres']['model']
    hr_units, lr_units = get_units(cfg)
    lr_model = cfg['lowres']['model']
    var = cfg['highres']['variable']
    show_ocean = cfg['visualization']['show_ocean']
    force_matching_scale = cfg['visualization']['force_matching_scale']
    global_min = cfg['highres']['scaling_params'] if 'scaling_params' in cfg['highres'] else None
    global_max = cfg['highres']['scaling_params'] if 'scaling_params' in cfg['highres'] else None
    extra_keys = cfg['stationary_conditions']['geographic_conditions']['geo_variables']
    hr_cmap, lr_cmap_dict = get_cmaps(cfg)
    default_lr_cmap = 'inferno'
    extra_cmap_dict = {"topo": "terrain", "sdf": "coolwarm", "lsm": "binary"}

    # Build list of keys for "variable" images:
    hr_key = f"{var}_hr"
    # Find LR keys from sample (assume keys ending with '_lr'): sort alphabetically for consistency
    lr_keys = sorted([k for k in sample.keys() if k.endswith('_lr')])

    # Scaled keys: HR and LR images
    scaled_keys = [hr_key] + lr_keys
    # Original keys: If available, ending with '_original'
    original_keys = []
    for key in scaled_keys:
        orig_key = key + "_original"
        if orig_key in sample:
            original_keys.append(orig_key)
    # Combing: extra keys (e.g. geo) will be appended later
    plot_keys = scaled_keys + original_keys
    if extra_keys is not None:
        plot_keys += extra_keys

    n_keys = len(plot_keys)

    # Create subplots in one row (one column per key)
    fig, axs = plt.subplots(1, n_keys, figsize=figsize)
    fig.suptitle(f"Sample from train dataset, {var} (HR: {hr_model}, LR: {lr_model})", fontsize=16)
    # Ensure axs is iterable (if only one subplot, wrap in list)
    if n_keys == 1:
        axs = np.array([axs])

    # Loop over each key and plot
    for idx, key in enumerate(plot_keys):
        ax = axs[idx]
        if key not in sample or sample[key] is None:
            ax.axis('off')
            continue

        # Get the image data; if a tensor, convert to np array
        img_data = sample[key]
        if torch.is_tensor(img_data):
            img_data = img_data.squeeze().cpu().numpy()
        img_data = _squeeze_geo_value(img_data, key)

        # For HR images (keys ending with '_hr' or '_hr_original'), if show_ocean is False, apply masking using lsm_hr
        if not show_ocean and (key.endswith("_hr") or key.endswith("_hr_original")):
            if "lsm_hr" in sample and sample["lsm_hr"] is not None:
                mask = sample["lsm_hr"].squeeze().cpu().numpy()
                # Assume mask values below 1 indicates ocean - set pixels to NaN
                img_data = np.where(mask < 1, np.nan, img_data)

        # Determine the colormap based on the key:
        if key.endswith('_hr') or key.endswith('_hr_original'):
            cmap = hr_cmap
        elif key.endswith('_lr') or key.endswith('_lr_original'):
            # Remove suffix to get the base condition name
            base = None
            if key.endswith('_lr'):
                base = key[:-3]
            elif key.endswith('_lr_original'):
                base = key[:-12]
            if lr_cmap_dict is not None and base is not None and base in lr_cmap_dict:
                cmap = lr_cmap_dict[base]
            else:
                cmap = default_lr_cmap
        else:
            # For extra keys, use the provided cmap_dict or default to 'viridis'
            if extra_cmap_dict is not None and key in extra_cmap_dict:
                cmap = extra_cmap_dict[key]
            else:
                cmap = 'viridis' # Default colormap for extra keys

        # Determine vmin and vmax: if force_matching_scale is True and dicts are provided, use them, otherwise compute from data
        if force_matching_scale and global_min is not None and global_max is not None:
            vmin = global_min.get(key, np.nanmin(img_data)) # get min from dict or compute from data
            vmax = global_max.get(key, np.nanmax(img_data)) # get max from dict or compute from data
        else:
            vmin = np.nanmin(img_data)
            vmax = np.nanmax(img_data)

        # Plot the image 
        im = ax.imshow(img_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.invert_yaxis()  # Invert y-axis to match the original image orientation
        ax.set_xticks([])
        ax.set_yticks([])

        # Set column title
        base = None

        if key.endswith('_hr'):
            title = f"HR {hr_model} ({var})\nscaled"
        elif key.endswith('_hr_original'):
            title = f"HR {hr_model} ({var})\noriginal [{hr_units}]"
        elif key.endswith('_lr'):
            base = key[:-3]
            title = f"LR {lr_model} ({base})\nscaled"
        elif key.endswith('_lr_original'):
            base = key[:-12]
            title = f"LR {lr_model} ({base})\noriginal [{lr_units[lr_keys.index(base)]}]"
        elif extra_keys is not None and key in extra_keys:
            if key == "topo":
                title = f"Topography"
            elif key == "sdf":
                title = f"SDF"
            elif key == "lsm":
                title = f"Land/Sea Mask"
            else:
                title = f"{key}"
        else:
            title = f"{key}"
        ax.set_title(title, fontsize=10)



        # Create an axes divider to add a colorbar and (for variable images) a boxplot
        divider = make_axes_locatable(ax)
        if key.endswith('_hr') or key.endswith('_lr') or key.endswith('_hr_original') or key.endswith('_lr_original'):
            bax = divider.append_axes("right", size="10%", pad=0.1)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            # Boxplot settings
            flierprops = dict(marker='o', markerfacecolor='none', markersize=2,
                              linestyle='None', markeredgecolor='darkgreen', alpha=0.4)
            medianprops = dict(linestyle='-', linewidth=2, color='black')
            meanpointprops = dict(marker='x', markerfacecolor='firebrick', markersize=5, markeredgecolor='firebrick')
            # Exclude Nans.
            if torch.is_tensor(img_data):
                mask = ~torch.isnan(img_data) # type: ignore
                img_bp = img_data[mask].flatten().cpu().numpy()
            else:
                mask = ~np.isnan(img_data)
                img_bp = img_data[mask].flatten()
            if len(img_bp) > 0:
                bax.boxplot(img_bp,
                            vert=True,
                            widths=2,
                            showmeans=True,
                            meanprops=meanpointprops,
                            flierprops=flierprops,
                            medianprops=medianprops,)
            bax.set_xticks([])
            bax.set_yticks([])
            bax.set_frame_on(False)
        else:
            # For extra keys, just add a colorbar
            cax = divider.append_axes("right", size="5%", pad=0.1)
            bax = None

        fig.colorbar(im, cax=cax, orientation='vertical')

    fig.tight_layout()

    return fig, axs



def plot_sample_with_boxplot(
        hr: Union[np.ndarray, dict], # Expecting a 2D array or dict with multiple days
        lr: Optional[Union[np.ndarray, dict]] = None,
        gen: Optional[Union[np.ndarray, dict]] = None,
        variable: str = "Variable",
        hr_model: str = "HR Model",
        lr_model: Optional[str] = None,
        gen_model: Optional[str] = None,
        dates: Optional[Union[str, list]] = None,
        save_path: Optional[str] = None,
        show: bool = False,
        cmap_default: str = "viridis",
        combine_into_grid: bool = False,
        n_rows_max: int = 5,
    ):
    """
        Plots HR, LR and generated images side-by-side with boxplots adjacent to each image.
        Accepts either single arrays or dicts + dates for multiple days.
        If combine_into_grid is True, multiple dates will be plotted in a grid layout in a single figure.
        - If hr is a dict and date is a list -> loop throuhg each date
        - If hr is a dict and date is a single str -> lookup that date once
        - If hr is a NumPy array, date is ignored
    """

    if save_path is None:
        save_path = f"./comparison/{variable}/"

    # Get cmap for variable if possible
    try:
        cmap_default = get_cmap_for_variable(variable)
    except ValueError:
        pass

    # === MULTIPLE DAYS ===
    if isinstance(dates, list):
        # === IF COMBINING INTO GRID PLOT IN ONE FIGURE ===
        if combine_into_grid:
            dates = dates[:n_rows_max]
            fields = [('Gen', gen), (f'{hr_model}', hr), (f'{lr_model}', lr)]
            fields = [(name, f) for name, f in fields if f is not None]
            n_fields = len(fields)
            n_rows = len(dates)

            fig = plt.figure(figsize=(5 * n_fields * 1.5, 3.5 * n_rows))
            gs = GridSpec(n_rows, n_fields * 2, width_ratios=[4, 1] * n_fields, figure=fig)

            for row_idx, d in enumerate(dates):
                row_data = []
                for label, dataset in fields:
                    if isinstance(dataset, dict) and d in dataset:
                        row_data.append((label, dataset[d]))
                    else:
                        row_data.append((label, None))

                vmin = min(np.min(x[1]) for x in row_data if x[1] is not None)
                vmax = max(np.max(x[1]) for x in row_data if x[1] is not None)

                for i, (label, data) in enumerate(row_data):
                    ax_img = fig.add_subplot(gs[row_idx, i * 2])
                    if data is not None:
                        # Ensure data is a NumPy array before plotting
                        if isinstance(data, dict):
                            logger.warning(f"Cannot plot dictionary for label '{label}'. Skipping.")
                            ax_img.set_title(f"{label} (invalid data type)")
                            ax_img.axis('off')
                            continue
                        if not isinstance(data, np.ndarray):
                            data = np.array(data)
                        # Set date title to only be date (not time)
                        try:
                            d_title = d.split(' ')[0] if ' ' in d else d
                        except Exception as e:
                            logger.warning(f"Error extracting date title from '{d}': {e}")
                            d_title = d
                        
                        im = ax_img.imshow(data, cmap=cmap_default, vmin=vmin, vmax=vmax)
                        ax_img.set_title(f"{label} ({d_title})", fontsize=10)
                        ax_img.axis('off')
                        ax_img.invert_yaxis()  # Invert y-axis to match the original image orientation
                        plt.colorbar(im, ax=ax_img, shrink=0.8)
                    else:
                        ax_img.set_title(f"{label} (missing)")
                        ax_img.axis('off')

                    ax_box = fig.add_subplot(gs[row_idx, i * 2 + 1])
                    if data is not None:
                        ax_box.boxplot(
                                data.flatten(),
                                vert=True,
                                widths=1,
                                showmeans=True,
                                meanprops=dict(marker='x', markerfacecolor='firebrick', markersize=5, markeredgecolor='firebrick'),
                                flierprops=dict(marker='o', markerfacecolor='none', markersize=2, linestyle='None', markeredgecolor='darkgreen', alpha=0.4),
                                medianprops=dict(linestyle='-', linewidth=2, color='black'),
                                patch_artist=True,
                                )

                    # ax_box.set_title("Box", fontsize=8)
                    ax_box.set_xticks([])
                    ax_box.tick_params(axis='y', labelsize=6)
                    ax_box.set_frame_on(False)


            fig.suptitle(f"{variable} | Multiple Dates", fontsize=16)
            fig.tight_layout()
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                path = os.path.join(save_path, f"{variable}_{hr_model}_vs_{lr_model}_boxplot__qualitative_visual.png")
                plt.savefig(path, dpi=300, bbox_inches='tight')
            if show:
                plt.show()
            plt.close()
            return  # Don't fall through to single plot

        # === IF NOT COMBINING, PLOT EACH DATE SEPARATELY IN MULTIPLE FIGURES ===
        for d in dates:
            plot_sample_with_boxplot(
                hr=hr, lr=lr, gen=gen,
                variable=variable,
                hr_model=hr_model,
                lr_model=lr_model,
                gen_model=gen_model,
                dates=d,
                save_path=os.path.join(save_path, f"{variable}_{d}_boxplot__qualitative_visual.png") if save_path else None,
                show=show,
                cmap_default=cmap_default
            )
        return 

    # === DICTIONARY LOOKUP ===
    if isinstance(hr, dict):
        if dates not in hr:
            logger.warning(f"Date '{dates}' not found in HR data dictionary. Skipping plot.")
            return
        hr = hr[dates]
        lr = lr.get(dates) if lr and isinstance(lr, dict) else None
        gen = gen.get(dates) if gen and isinstance(gen, dict) else None

    # === SINGLE PLOT ===

    fields = [('HR', hr, hr_model)]
    if gen is not None:
        fields.insert(0, ('Generated', gen, gen_model if gen_model else "Gen Model"))
    if lr is not None:
        fields.append(('LR', lr, lr_model if lr_model else "LR Model"))

    n_fields = len(fields)
    fig = plt.figure(figsize=(5 * n_fields * 1.5, 5)) # 5 for each image, 1.5 for boxplot
    gs = GridSpec(1, n_fields * 2, width_ratios=[4, 1] * n_fields, figure=fig)

    vmin = min(np.nanmin(f[1]) for f in fields if isinstance(f[1], np.ndarray))
    vmax = max(np.nanmax(f[1]) for f in fields if isinstance(f[1], np.ndarray))


    for i, (label, data, model) in enumerate(fields):
        if data is None:
            continue
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        ax_img = fig.add_subplot(gs[0, i * 2])
        im = ax_img.imshow(data, cmap=cmap_default, vmin=vmin, vmax=vmax)
        ax_img.set_title(f"{label} ({model})", fontsize=14)
        ax_img.axis('off')
        ax_img.invert_yaxis()  # Invert y-axis to match the original image orientation

        cbar = plt.colorbar(im, ax=ax_img, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)

        ax_box = fig.add_subplot(gs[0, i * 2 + 1])
        ax_box.boxplot(data.flatten(), vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', color='blue'),
                          medianprops=dict(color='red'),
                          flierprops=dict(marker='o', markerfacecolor='none', markersize=5, markeredgecolor='blue', alpha=0.5))
        ax_box.set_xticks([])
        ax_box.tick_params(axis='y', labelsize=8)


    suptitle = f"{variable} | {dates}" if dates else variable
    fig.suptitle(suptitle, fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close()
    
    return 


def plot_samples(samples, cfg, n_samples_threshold=3, figsize=(15, 8)):
    """
    Plot a batch of samples (provided as a list of sample dictionaries) in a grid where each row is a sample and
    each column corresponds to a particular key (e.g., HR, LR, originals, geo).
    
    If the number of samples exceeds n_samples_threshold, only the first n_samples_threshold will be plotted.
    
    Parameters:
      - sample_list: List of sample dictionaries.
      - cfg: Configuration dictionary containing model and variable information.
      - figsize: Overall figure size.
      
    Returns:
      - fig: The matplotlib Figure object.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Extract configuration for plotting
    hr_model = cfg['highres']['model']
    lr_model = cfg['lowres']['model']
    var = cfg['highres']['variable']
    hr_units, lr_units = get_units(cfg)
    hr_cmap, lr_cmap_dict = get_cmaps(cfg)
    default_lr_cmap = 'viridis'
    extra_cmap_dict = {"topo": "terrain", "lsm": "binary", "sdf": "coolwarm"}
    show_ocean = cfg.get('visualization', {}).get('show_ocean', False)
    force_matching_scale = cfg.get('visualization', {}).get('force_matching_scale', True)
    global_min = cfg.get('visualization', {}).get('global_min', None)
    global_max = cfg.get('visualization', {}).get('global_max', None)
    extra_keys = cfg.get('visualization', {}).get('extra_keys', None)


    # If single batch dict is passed, unpack it to a list
    if isinstance(samples, dict):
        # Figure out batch size from first tensor we find:
        batch_size = None
        for v in samples.values():
            if torch.is_tensor(v):
                batch_size = v.shape[0]
                break
            if isinstance(v, list) and all(torch.is_tensor(x) for x in v):
                batch_size = len(v)
                break
        if batch_size is None:
            raise ValueError("No tensor found in the sample dictionary to determine batch size.")
        
        sample_list = []
        for i in range(batch_size):
            single = {}
            for k, v in samples.items():
                if torch.is_tensor(v):
                    # Slice tensor on batch dim
                    single[k] = v[i]
                elif isinstance(v, (list, tuple)) and len(v) == batch_size:
                    # Truly per-sample list
                    single[k] = v[i]
                else:
                    # Some constant list or metadata: leave as-is
                    single[k] = v
            sample_list.append(single)
    else:
        sample_list = samples

    # logger.info(f"Plotting first {n_samples_threshold} samples out of {len(sample_list)} provided.")
    sample_list = sample_list[:n_samples_threshold]
    
    # Construct the keys:
    # HR key is "var_hr" (e.g., "prcp_hr")
    hr_key = f"{var}_hr"
    # Assume LR keys end with '_lr'
    lr_keys = sorted([key for key in sample_list[0].keys() if key.endswith('_lr')])
    scaled_keys = [hr_key] + lr_keys

    # Determine original keys if available.
    original_keys = []
    for key in scaled_keys:
        orig_key = key + "_original"
        if orig_key in sample_list[0]:
            original_keys.append(orig_key)
    
    # Build final list of keys. Append extra keys if provided.
    plot_keys = scaled_keys + original_keys
    if extra_keys is not None:
        plot_keys += extra_keys

    num_samples = len(sample_list)
    num_keys = len(plot_keys)

    # Create a grid with rows = number of samples and columns = number of keys
    fig, axs = plt.subplots(num_samples, num_keys, figsize=figsize)
    # Set figure title 
    fig.suptitle(f"Sample images for {var} (HR: {hr_model} and LR: {lr_model})", fontsize=16)
    if num_samples == 1:
        axs = np.expand_dims(axs, axis=0)
    if num_keys == 1:
        axs = np.expand_dims(axs, axis=1)

    for row, sample in enumerate(sample_list):
        for col, key in enumerate(plot_keys):
            ax = axs[row, col]
            if key not in sample or sample[key] is None:
                ax.axis('off')
                continue
            # Retrieve image data
            img_data = sample[key]
            if torch.is_tensor(img_data):
                img_data = img_data.squeeze().cpu().numpy()
            img_data = _squeeze_geo_value(img_data, key)
            # For HR images mask out ocean using lsm_hr if needed.
            if not show_ocean and (key.endswith('_hr') or key.endswith('_hr_original')):
                if "lsm_hr" in sample and sample["lsm_hr"] is not None:
                    mask = sample["lsm_hr"].squeeze().cpu().numpy()
                    img_data = np.where(mask < 1, np.nan, img_data)
            # Determine color limits.
            if force_matching_scale and global_min is not None and global_max is not None:
                vmin = global_min.get(key, np.nanmin(img_data))
                vmax = global_max.get(key, np.nanmax(img_data))
            else:
                vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
            # Choose colormap:
            if key.endswith('_hr') or key.endswith('_hr_original'):
                cmap = hr_cmap
            elif key.endswith('_lr') or key.endswith('_lr_original'):
                if key.endswith('_lr'):
                    base = key[:-3]
                else:
                    base = key[:-12]
                if lr_cmap_dict is not None and base in lr_cmap_dict:
                    cmap = lr_cmap_dict[base]
                else:
                    cmap = default_lr_cmap
            else:
                if extra_cmap_dict is not None and key in extra_cmap_dict:
                    cmap = extra_cmap_dict[key]
                else:
                    cmap = 'viridis'
            im = ax.imshow(img_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            divider = make_axes_locatable(ax)
            # For keys that correspond to variable fields, add a boxplot next to the colorbar.
            if key.endswith('_hr') or key.endswith('_lr') or key.endswith('_hr_original') or key.endswith('_lr_original'):
                bax = divider.append_axes("right", size="10%", pad=0.1)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                flierprops = dict(marker='o', markerfacecolor='none', markersize=2,
                                  linestyle='none', markeredgecolor='darkgreen', alpha=0.4)
                medianprops = dict(linestyle='-', linewidth=2, color='black')
                meanpointprops = dict(marker='x', markerfacecolor='firebrick', markersize=5, markeredgecolor='firebrick')
                img_flat = img_data[~np.isnan(img_data)].flatten()
                if len(img_flat) > 0:
                    bax.boxplot(img_flat,
                                vert=True,
                                widths=2,
                                patch_artist=True,
                                showmeans=True,
                                meanprops=meanpointprops,
                                medianprops=medianprops,
                                flierprops=flierprops)
                bax.set_xticks([])
                bax.set_yticks([])
                bax.set_frame_on(False)
            else:
                cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, cax=cax)

            base = None


            # Set column title (only for top row)
            if row == 0:
                if key.endswith('_hr'):
                    title = f"HR {hr_model} ({var})\nscaled"
                elif key.endswith('_hr_original'):
                    title = f"HR {hr_model} ({var})\noriginal [{hr_units}]"
                elif key.endswith('_lr'):
                    base = key[:-3]
                    title = f"LR {lr_model} ({base})\nscaled"
                elif key.endswith('_lr_original'):
                    base = key[:-12]
                    title = f"LR {lr_model} ({base})\noriginal [{lr_units[lr_keys.index(base)]}]"
                elif extra_keys is not None and key in extra_keys:
                    if key == "topo":
                        title = f"Topography"
                    elif key == "sdf":
                        title = f"SDF"
                    elif key == "lsm":
                        title = f"Land/Sea Mask"
                    else:
                        title = f"{key}"
                else:
                    title = f"{key}"
                ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig, axs



def plot_samples_and_generated(
        samples,
        generated,
        cfg,
        *,
        transform_back_bf_plot=False,
        back_transforms=None,
        n_samples_threshold=3,
        figsize=(15, 8),
):
    """
    Like ``plot_samples`` but adds an extra left-most column with “Generated”
    images (one per sample).

    Parameters
    ----------
    samples : dict | list[dict]
        The usual batch/list accepted by ``plot_samples``.
    generated : torch.Tensor | np.ndarray | list
        Shape (B,1,H,W) or list/tuple of length B with 2-D arrays.
    transform_back_bf_plot : bool, default False
        Apply inverse scaling before display.
    back_transforms : dict[str, Callable], optional
        Mapping *plot-key* → inverse-transform function.  Only used when
        *transform_back_bf_plot* is ``True``.
    """
    # Extract configuration for plotting
    hr_model = cfg['highres']['model']
    lr_model = cfg['lowres']['model']
    var = cfg['highres']['variable']
    hr_units, lr_units = get_units(cfg)
    hr_cmap, lr_cmap_dict = get_cmaps(cfg)
    default_lr_cmap = 'viridis'
    extra_cmap_dict = {"topo": "terrain", "lsm": "binary", "sdf": "coolwarm"}
    show_ocean = cfg.get('visualization', {}).get('show_ocean', False)
    force_matching_scale = cfg.get('visualization', {}).get('force_matching_scale', True)
    global_min = cfg.get('visualization', {}).get('global_min', None)
    global_max = cfg.get('visualization', {}).get('global_max', None)
    extra_keys = cfg.get('visualization', {}).get('extra_keys', None)
    scaling = cfg.get('visualization', {}).get('scaling', True)


    # ------------------------------------------------------------------ utils
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def maybe_inverse(k, arr):
        logger.info(f"Applying inverse transformation for key: {k}")
        if transform_back_bf_plot and back_transforms and k in back_transforms:
            logger.info(f"Found inverse transformation for key: {k}")
            return back_transforms[k](arr)
        logger.info(f"No inverse transformation found for key: {k}")
        return arr
    # logger.info(f'Samples: {samples}')
    # logger.info(f'Generated: {generated}')
    # -------------------------------------------------------- unpack samples
    if isinstance(samples, dict):              # turn single batch-dict → list
        B = None
        for v in samples.values():
            if torch.is_tensor(v):
                B = v.shape[0]
                break
            if isinstance(v, list) and v and torch.is_tensor(v[0]):
                B = len(v)
                break
        if B is None:
            raise ValueError("Could not determine batch size (B) from samples dictionary.")
        sample_list = []
        for i in range(B):
            d = {}
            for k, v in samples.items():
                if torch.is_tensor(v):
                    d[k] = v[i]
                elif isinstance(v, (list, tuple)) and len(v) == B:
                    d[k] = v[i]
                else:
                    d[k] = v
            sample_list.append(d)
    else:
        sample_list = list(samples)

    sample_list = sample_list[:n_samples_threshold]

    # ------------------------------------------------------- generated batch
    # logger.info(f"Generated shape: {generated.shape}")
    gen_np = to_numpy(generated)
    if gen_np.ndim == 4:               # (B, 1, H, W), multiple samples with 1 channel
        gen_np = gen_np[:, 0, :, :]
    elif gen_np.ndim == 3:             # (B, H, W), multiple samples
        pass
    elif gen_np.ndim == 2:             # (H, W), single samples
        gen_np = np.expand_dims(gen_np, axis=0)
    else:
        raise ValueError(f"Unexpected shape for generated samples: {gen_np.shape}")

    gen_np = gen_np[:len(sample_list)]
    # logger.info(f"Generated shape after slicing: {gen_np.shape}")

    # inject into dicts
    gen_key = "generated"
    for d, im in zip(sample_list, gen_np):
        d[gen_key] = im

    # --------------------------------------------------- assemble key order
    hr_key = f"{var}_hr"
    lr_keys = sorted(k for k in sample_list[0] if k.endswith("_lr"))
    original_keys = [k + "_original"
                     for k in (hr_key, *lr_keys)
                     if k + "_original" in sample_list[0]]

    plot_keys = [gen_key, hr_key, *lr_keys, *original_keys]
    if extra_keys:
        plot_keys.extend(extra_keys)

    # ----------------------------------------------------------- colourlims
    if force_matching_scale and global_min is not None and global_max is not None:
        # share HR limits with generated if user hasn’t provided any
        global_min.setdefault(gen_key, global_min.get(hr_key))
        global_max.setdefault(gen_key, global_max.get(hr_key))

    # -------------------------------------------------------------- figure
    n_rows, n_cols = len(sample_list), len(plot_keys)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(f"Generated vs. data – {var}  "
                 f"(HR {hr_model} / LR {lr_model})", fontsize=16)

    # Ensure axs is always 2D
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs[np.newaxis, :]
    if n_cols == 1:
        axs = axs[:, np.newaxis]

    fig.suptitle(f"Generated vs. conditions – {var} (HR {hr_model} / LR {lr_model}) ")
    
    # --------------------------------------------------------- draw images
    for r, sample in enumerate(sample_list):
        for c, key in enumerate(plot_keys):
            ax = axs[r, c]
            if key not in sample or sample[key] is None:
                ax.axis("off")
                continue
            
            # # Print key and shape for debugging
            # logger.info(f"Key: {key}")
            # logger.info(f"Shape: {sample[key].shape}")

            img = to_numpy(sample[key]).squeeze()
            img = _squeeze_geo_value(img, key)
            img = maybe_inverse(key, img)

            # mask ocean for HR & generated columns
            if not show_ocean and key in {gen_key, hr_key, f"{hr_key}_original"}:
                if "lsm_hr" in sample and sample["lsm_hr"] is not None:
                    mask = to_numpy(sample["lsm_hr"]).squeeze()
                    img = np.where(mask < 1, np.nan, img)

            # choose colormap
            if key in {gen_key, hr_key, f"{hr_key}_original"}:
                cmap = hr_cmap
            elif key.endswith("_lr") or key.endswith("_lr_original"):
                base = key.replace("_lr", "").replace("_lr_original", "")
                cmap = (lr_cmap_dict or {}).get(base, default_lr_cmap)
            else:
                cmap = (extra_cmap_dict or {}).get(key, "viridis")

            if force_matching_scale and global_min is not None and global_max is not None:
                vmin = global_min.get(key, np.nanmin(img))
                vmax = global_max.get(key, np.nanmax(img))
            else:
                vmin, vmax = np.nanmin(img), np.nanmax(img)

            # Check dimensions of image. if (1, dim, dim), squeeze
            if img.ndim == 3 and img.shape[0] == 1:
                img = img.squeeze(0)

            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])

            # colour-bar
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

            # column headers
            if r == 0:
                if scaling:
                    if transform_back_bf_plot and back_transforms and key in back_transforms:
                        titles = {
                            gen_key: "Generated",
                            hr_key: f"HR {hr_model}, {var}\nback-transformed [{hr_units}]",
                            **{k: f"LR {lr_model} ({k[:-3]})\nback-transformed [{lr_units[lr_keys.index(k[:-3])] if k[:-3] in lr_keys else 'unknown'}]" for k in lr_keys},
                            **{k: f"LR {lr_model} ({k[:-12]})\nscaled" for k in original_keys},
                        }
                    else:
                        titles = {
                            gen_key: "Generated",
                            hr_key: f"HR {hr_model}, {var}\nscaled",
                            **{k: f"LR {lr_model} ({k[:-3]})\nscaled" for k in lr_keys},
                            **{k: f"LR {lr_model} ({k[:-12]})\noriginal [{lr_units[lr_keys.index(k[:-12])] if k[:-12] in lr_keys else 'unknown'}]" for k in original_keys},
                        }
                else:
                    titles = {
                        gen_key: "Generated",
                        hr_key: f"HR {hr_model}, {var}\nno scaling [{hr_units}]",
                        **{k: f"LR {lr_model} ({k[:-3]})\nno scaling [{lr_units[lr_keys.index(k[:-3])] if k[:-3] in lr_keys else 'unknown'}]" for k in lr_keys},
                        **{k: f"LR {lr_model}" for k in lr_keys},
                    }
                
                ax.set_title(titles.get(key, key), fontsize=9)

    fig.tight_layout()
    return fig, axs