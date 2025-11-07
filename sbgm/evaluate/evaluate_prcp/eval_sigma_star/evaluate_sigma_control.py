"""
Main entrypoint for σ*-dependent evaluation.
"""

import logging
from pathlib import Path
from sbgm.evaluate.evaluate_prcp.eval_sigma_star.metrics_sigma_control import evaluate_sigma_control
from sbgm.evaluate.evaluate_prcp.eval_sigma_star.plot_sigma_control import plot_sigma_control, plot_sigma_control_examples_grid, plot_sigma_control_psd_curves
from sbgm.utils import get_model_string

logger = logging.getLogger(__name__)

def run(cfg, make_plots=True):
    model_name = get_model_string(cfg)
    sigma_grid = cfg.full_gen_eval.sigma_star_grid
    base_gen = Path(cfg.paths.sample_dir) / "generation" / model_name
    out_dir = Path(cfg.paths.sample_dir) / "evaluation" / model_name / "prcp" / "sigma_control"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[SigmaControl] Evaluating σ* grid {sigma_grid} for {model_name}")

    metrics_paths = evaluate_sigma_control(cfg, sigma_grid, base_gen, out_dir)

    figures_dir = Path(out_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if make_plots:
        plot_sigma_control(
            metrics_paths["summary"],
            figures_dir,
            combined=bool(getattr(getattr(cfg, "full_gen_eval", {}), "sigma_control_plot_combined", True)),
        )
        plot_sigma_control_examples_grid(
            cfg,
            sigma_star_grid=sigma_grid,
            gen_base_dir=base_gen,
            out_dir=out_dir,
            n_members=int(getattr(getattr(cfg, "full_gen_eval", {}), "example_n_members", 3)),
            date=getattr(getattr(cfg, "full_gen_eval", {}), "example_date", None),
            land_only=bool(getattr(getattr(cfg, "full_gen_eval", {}), "eval_land_only", True)),
            fname="examples_sigma_grid.png",
        )
        # PSD curves per sigma_star (ensemble-average across dates)
        plot_sigma_control_psd_curves(out_dir)

    logger.info(f"[SigmaControl] Done. Results in {out_dir}")
    return out_dir