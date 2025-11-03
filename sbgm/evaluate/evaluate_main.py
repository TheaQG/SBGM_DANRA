from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import torch

from sbgm.utils import get_model_string
from sbgm.evaluate.evaluation import EvaluationConfig, EvaluationRunner

logger = logging.getLogger(__name__)


def _default_gen_dir(cfg) -> Path:
    model_str = get_model_string(cfg)
    sample_root = cfg["paths"]["sample_dir"]
    return Path(sample_root) / "generation" / model_str


def _default_eval_dir(cfg) -> Path:
    model_str = get_model_string(cfg)
    sample_root = cfg["paths"]["sample_dir"]
    return Path(sample_root) / "evaluation" / model_str


def evaluation_main(cfg):
    """
        Launch modular evaluation process based on provided config.
    """
    fe = cfg.get("full_gen_eval", {})

    # standard flags
    do_prob = bool(fe.get("do_prob", True))
    do_scale = bool(fe.get("do_scale", True))      # we can wire this later
    do_ext = bool(fe.get("do_ext", False))          # later
    do_dist = bool(fe.get("do_dist", True))    # new

    gen_dir = fe.get("gen_dir", None)
    eval_dir = fe.get("eval_dir", None)

    # seeds like old version
    seed = int(fe.get("seed", 1234))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # device is still read from your YAML
    device = torch.device(cfg["training"]["device"])

    # directories
    gen_root = Path(gen_dir) if gen_dir is not None else _default_gen_dir(cfg)
    eval_root = Path(eval_dir) if eval_dir is not None else _default_eval_dir(cfg)
    eval_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"[evaluation_main] gen_root: {gen_root}")
    logger.info(f"[evaluation_main] eval_root: {eval_root}")

    # build NEW evaluation config (note: this is sbgm/evaluate/evaluation.py)
    ev_cfg = EvaluationConfig(
        gen_dir=str(gen_root),
        out_dir=str(eval_root),
        
        eval_land_only=bool(fe.get("eval_land_only", False)),
        prefer_phys=bool(fe.get("prefer_phys", True)),
        region_mask_path=fe.get("region_mask_path", None),
        grid_km_per_px=float(fe.get("grid_km_per_px", 2.5)),
        lr_grid_km_per_px=float(fe.get("lr_grid_km_per_px", 31.0)),
        seasons=tuple(fe.get("seasons", ("ALL", "DJF", "MAM", "JJA", "SON"))),
        
        hr_dx_km=float(fe.get("hr_dx_km", fe.get("grid_km_per_px", 2.5))),
        lr_dx_km=float(fe.get("lr_dx_km", fe.get("lr_grid_km_per_px", 31.0))),

        # FSS evaluation config fields
        thresholds_mm=tuple(fe.get("thresholds_mm", (1.0, 5.0, 10.0))),
        fss_scales_km=tuple(fe.get("fss_scales_km", (5, 10, 20))),
        fss_thresholds_mm=tuple(fe.get("fss_thresholds_mm", fe.get("thresholds_mm", (1.0, 5.0, 10.0)))),
        compute_lr_fss=bool(fe.get("compute_lr_fss", True)),

        # ISS evaluation config fields  
        iss_thresholds_mm=tuple(fe.get("iss_thresholds_mm", fe.get("thresholds_mm", (1.0, 5.0, 10.0)))),
        iss_scales_km=tuple(fe.get("iss_scales_km", (5, 10, 20))),
        compute_lr_iss=bool(fe.get("compute_lr_iss", True)),
        
        # Reliability, Spread-Skill, PIT config fields
        reliability_bins=int(fe.get("reliability_bins", 10)),
        spread_skill_bins=int(fe.get("spread_skill_bins", 10)),
        pit_bins=int(fe.get("pit_bins", 20)),

        # Variogram
        variogram_p=float(fe.get("variogram_p", 0.5)),
        variogram_max_pairs=int(fe.get("variogram_max_pairs", 5000)),

        # scale-specific fields with sensible fallbacks to the old names
        low_k_max=float(fe.get("low_k_max", 1.0 / 200.0)),
        high_k_min=float(fe.get("high_k_min", 1.0 / 20.0)),
        make_plots=bool(fe.get("make_plots", True)),
        
        # Distributional evaluation config fields
        dist_n_bins=int(fe.get("pixel_dist_n_bins", 80)),
        dist_vmax_percentile=float(fe.get("pixel_dist_vmax_percentile", 99.5)),
        dist_include_lr=bool(fe.get("pixel_dist_include_lr", True)),
        dist_save_cap=int(fe.get("pixel_dist_save_cap", 200_000)),
    )

    # map YAML flags -> new modular task names
    tasks: list[str] = []
    if do_prob:
        tasks.append("prcp_probabilistic")
    if do_scale:
        tasks.append("prcp_scale")
    if do_ext:
        tasks.append("prcp_extremes")
    if do_dist:
        tasks.append("prcp_distributional")

    runner = EvaluationRunner(
        cfg_yaml=cfg,
        eval_cfg=ev_cfg,
        device=device,
        baseline_eval_dirs=None,
        plot_only=bool(fe.get("plot_only", False)),
    )

    runner.run(tasks=tasks)

    logger.info(f"[evaluation_main_new] Done. Outputs at: {eval_root}")
    return eval_root