import os 
import logging
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf

from sbgm.training_utils import get_model, get_final_gen_dataloader
from sbgm.evaluate_sbgm.generation import GenerationRunner, GenerationConfig
from sbgm.utils import get_model_string

logger = logging.getLogger(__name__)

def _resolve_out_dir(cfg) -> Path:
    """ Default output dir: <paths.sample_dir>/generation/<model_name>/"""
    model_name_str = get_model_string(cfg)
    gen_dir = Path(cfg["paths"]["sample_dir"]) / 'generation' / model_name_str
    gen_dir.mkdir(parents=True, exist_ok=True)
    return gen_dir

def _build_generation_config(cfg, out_root: Path) -> GenerationConfig:
    # Ensemble size: prefer evaluation.n_gen_samples, fall back to data_handling.n_gen_samples, else 32
    # Set the configuration cfg as full_gen_eval
    cfg_full_gen_eval = cfg.get('full_gen_eval', cfg)
    M = int(cfg_full_gen_eval.get('ensemble_size', cfg.data_handling.get('n_gen_samples', 32)))
    
    edm = cfg.get('edm', {})
    gen_cfg = GenerationConfig(
        output_root=str(out_root),
        ensemble_size=M,
        sampler_steps=int(edm.get('sampling_steps', 40)),
        seed = int(cfg_full_gen_eval.get('seed', 1234)),
        use_edm = bool(edm.get('enabled', True)),
        sigma_min = float(edm.get('sigma_min', 0.002)),
        sigma_max = float(edm.get('sigma_max', 80.0)),
        rho = float(edm.get('rho', 7.0)),
        S_churn = float(edm.get('S_churn', 0.0)),
        S_min = float(edm.get('S_min', 0.0)),
        S_max = float(edm.get('S_max', float('inf'))),
        S_noise = float(edm.get('S_noise', 1.0)),
        predict_residual=bool(edm.get('predict_residual', False)),
        save_space="physical", # Always save physical space samples for evaluation
        max_dates=int(cfg_full_gen_eval.get('max_dates', -1)),  # -1 means all dates
    )
    return gen_cfg

def generation_main(cfg):
    """
        Entry point used by launch_generation.run()
    """
    # ----------------------- Seed & logging -----------------------
    seed = int(cfg.full_gen_eval.get('seed', 1234))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # logger.info(f"[generation_main] Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # ----------------------- Device -----------------------
    device = cfg.training.device

    # ----------------------- Model & checkpoint -----------------------
    # Matches prior script: use 'network_params' (not EMA) for sampling unless changing policy later
    model, ckpt_dir, ckpt_name = get_model(cfg)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "network_params" not in ckpt:
        raise KeyError(f"Checkpoint missing 'network_params': {ckpt_path}")
    model.load_state_dict(ckpt["network_params"])
    model.eval()
    logger.info(f"[generation_main] Loaded checkpoint: {ckpt_path}")

    # ----------------------- Data (ensure deterministic test loop) -----------------------
    # If your get_gen_dataloader reads these from cfg, set them explicitly:
    cfg.setdefault('data_handling', {})
    cfg.data_handling['split'] = 'test'             # or 'evaluation' if that's your naming
    cfg.data_handling['batch_size'] = 1             # 1 date per iteration
    cfg.data_handling['shuffle'] = False            # preserve chronological/file order
    cfg.data_handling['drop_last'] = False          # keep last sample even if incomplete

    gen_dataloader = get_final_gen_dataloader(cfg)

    # ----------------------- Output root -----------------------
    out_root = _resolve_out_dir(cfg)

    # ----------------------- Runner config -----------------------
    gen_cfg = _build_generation_config(cfg, out_root)

    # ----------------------- Run -----------------------
    runner = GenerationRunner(model=model, cfg=cfg, device=device, out_root=out_root, gen_config=gen_cfg)
    runner.run(gen_dataloader)

    logger.info(f"[generation_main] Done. Artifacts at: {out_root}")
    return out_root    
