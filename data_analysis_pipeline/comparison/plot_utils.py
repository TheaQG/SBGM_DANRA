import os
import logging
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sbgm.variable_utils import get_cmap_for_variable, get_unit_for_variable

logger = logging.getLogger(__name__)

def plot_samples_grid(
    *,
    hr: dict,
    lr: dict,
    hr_model: str,
    lr_model: str,
    variable: str,
    dates: list,
    combine_into_grid: bool = True,
    save_path: str = "./figures/comparison",
    show: bool = False,
):
    """
    Publication-style qualitative comparison figure (HR vs LR) for one or more days.

    Parameters
    ----------
    hr, lr : dict[datetime -> 2D ndarray]
        Mapping from date to 2D field for each dataset.
    hr_model, lr_model : str
        Pretty labels for figure titles/legends.
    variable : str
        Variable name used to choose colormap/units.
    dates : list[datetime]
        Dates to plot (rows). Only dates present in both dicts are plotted.
    combine_into_grid : bool
        If True, all rows are collected into a single figure. If False, one figure per day.
    save_path : str
        Directory where the figure(s) will be written.
    show : bool
        If True, display the figure window as well.

    Returns
    -------
    None (saves PNGs to disk)
    """
    os.makedirs(save_path, exist_ok=True)

    # Filter to shared dates and keep ordering of `dates`
    shared = [d for d in dates if d in hr and d in lr]
    if not shared:
        logger.warning("plot_samples_grid: No overlapping dates between HR and LR.")
        return

    # Colormap + units
    cmap_hr = get_cmap_for_variable(variable)
    cmap_lr = get_cmap_for_variable(variable)  # same palette for comparability
    units = get_unit_for_variable(variable)

    def _finite_minmax(*arrays):
        vals = []
        for a in arrays:
            if a is None:
                continue
            a = np.asarray(a)
            a = a[np.isfinite(a)]
            if a.size:
                vals.append(a)
        if not vals:
            return None, None
        allv = np.concatenate(vals)
        return float(np.nanmin(allv)), float(np.nanmax(allv))

    def _add_colorbar_and_boxplot(fig, ax, im, img_2d, vlim):
        divider = make_axes_locatable(ax)
        bax = divider.append_axes("right", size="10%", pad=0.1)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax, orientation="vertical")
        vals = np.asarray(img_2d)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            bax.boxplot(
                vals,
                vert=True,
                widths=0.9,
                showmeans=True,
                meanprops=dict(marker="x", markerfacecolor="firebrick", markersize=5, markeredgecolor="firebrick"),
                flierprops=dict(marker="o", markerfacecolor="none", markersize=2, linestyle="None", markeredgecolor="darkgreen", alpha=0.35),
                medianprops=dict(linestyle="-", linewidth=1.2, color="black"),
            )
            if vlim and np.isfinite(vlim).all():
                bax.set_ylim(vlim[0], vlim[1])
        bax.set_xticks([])
        bax.set_yticks([])
        bax.set_frame_on(False)

    def _plot_one_figure(day_list):
        n_rows = len(day_list)
        n_cols = 2  # HR | LR
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows), squeeze=False)
        fig.suptitle(f"Qualitative comparison – {variable} ({hr_model} vs {lr_model})", fontsize=14)

        for r, d in enumerate(day_list):
            fld_hr = np.asarray(hr[d])
            fld_lr = np.asarray(lr[d])

            # Fair visual comparison: per-row vmin/vmax pooled across HR & LR
            vmin, vmax = _finite_minmax(fld_hr, fld_lr)
            if vmin is None or vmax is None:
                vmin, vmax = np.nanmin(fld_hr), np.nanmax(fld_hr)

            for c, (fld, model_label, cmap) in enumerate(
                [(fld_hr, f"HR {hr_model}", cmap_hr), (fld_lr, f"LR {lr_model}", cmap_lr)]
            ):
                ax = axs[r, c]
                im = ax.imshow(fld, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"{model_label}\n[{units}]", fontsize=11)
                # Add Y label with date (leftmost column only)
                if c == 0:
                    ax.set_ylabel(d.strftime("%Y-%m-%d"), fontsize=10)

                _add_colorbar_and_boxplot(fig, ax, im, fld, (vmin, vmax))

        # fig.tight_layout(rect=[0, 0, 1, 0.97])
        fname = f"{variable}_{hr_model}_vs_{lr_model}_qualitative"
        if len(day_list) == 1:
            fname += f"_{day_list[0].strftime('%Y%m%d')}"
        out = os.path.join(save_path, f"{fname}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"      Saved qualitative comparison to {out}")
        if show:
            plt.show()
        plt.close(fig)

    if combine_into_grid:
        _plot_one_figure(shared)
    else:
        for d in shared:
            _plot_one_figure([d])
