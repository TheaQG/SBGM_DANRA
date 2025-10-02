"""
    Entry point like generation_main, but for evaluation after generation has been done.
    Reads generation outputs from cfg or CLI, builds mask, and runs EvaluationRunner.
"""

import os
import logging
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
import yaml

from sbgm.utils import get_model_string
from sbgm.training_utils import get_model  # only for model string & dims if needed
from sbgm.evaluate_sbgm.evaluation import EvaluationRunner, EvaluationConfig

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
    cfg_full_gen_eval = cfg.get("full_gen_eval", {})
    do_prob = bool(cfg_full_gen_eval.get("do_prob", True))
    do_cap = bool(cfg_full_gen_eval.get("do_cap", True))
    do_ext = bool(cfg_full_gen_eval.get("do_ext", True))
    gen_dir = cfg_full_gen_eval.get("gen_dir", None)  # If None, use default
    eval_dir = cfg_full_gen_eval.get("eval_dir", None)

    # Logging & seed
    seed = int(cfg.get("evaluation", {}).get("seed", 1234))
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); np.random.seed(seed)
    # logger.info(f"[evaluation_main] Configuration:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device(cfg["training"]["device"])

    # Directories
    gen_root = Path(gen_dir) if gen_dir is not None else _default_gen_dir(cfg)
    eval_root = Path(eval_dir) if eval_dir is not None else _default_eval_dir(cfg)
    eval_root.mkdir(parents=True, exist_ok=True)

    # Land mask (optional)
    try:
        mask = None  # replace with your own loader if you have one
        # mask = load_land_mask_if_any(cfg)  # returns torch.bool [H,W]
    except Exception as e:
        logger.warning(f"[evaluation_main] No land mask loaded: {e}")
        mask = None

    # Build EvaluationConfig
    ev_cfg = EvaluationConfig(
        gen_dir=str(gen_root),
        out_dir=str(eval_root),
        grid_km_per_px=float(cfg.get("data", {}).get("grid_km_per_px", 2.0)),
        fss_scales_km=tuple(cfg.get("evaluation", {}).get("fss_scales_km", (5,10,20))),
        thresholds_mm=tuple(cfg.get("evaluation", {}).get("thresholds_mm", (1.0,5.0,10.0))),
        wet_threshold_mm=float(cfg.get("evaluation", {}).get("wet_threshold_mm", 1.0)),
        reliability_bins=int(cfg.get("evaluation", {}).get("reliability_bins", 10)),
        spread_skill_bins=int(cfg.get("evaluation", {}).get("spread_skill_bins", 10)),
        pit_bins=int(cfg.get("evaluation", {}).get("pit_bins", 20)),
        psd_ignore_low_k_bins=int(cfg.get("evaluation", {}).get("psd_ignore_low_k_bins", 1)),
        random_ref_kind=str(cfg.get("evaluation", {}).get("random_ref_kind", "phase_randomized")),
        seasons=tuple(cfg.get("evaluation", {}).get("seasons", ("ALL","DJF","MAM","JJA","SON"))),
        seed=seed,
    )

    runner = EvaluationRunner(cfg_yaml=cfg, eval_cfg=ev_cfg, device=device, mask=mask)
    runner.run_all(do_prob=do_prob, do_cap=do_cap, do_ext=do_ext)

    logger.info(f"[evaluation_main] Done. Outputs at: {eval_root}")
    return eval_root