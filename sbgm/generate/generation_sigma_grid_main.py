import os
import logging
from pathlib import Path
import numpy as np
import torch

from sbgm.training_utils import get_model, get_final_gen_dataloader
from sbgm.generate.generation import GenerationRunner, GenerationConfig
from sbgm.utils import get_model_string

logger = logging.getLogger(__name__)

def _resolve_base_out_dir(cfg) -> Path:
    """Base dir: <paths.sample_dir>/generation/<model_name>/"""
    model_name_str = get_model_string(cfg)
    base = Path(cfg["paths"]["sample_dir"]) / 'generation' / model_name_str
    base.mkdir(parents=True, exist_ok=True)
    return base

def _build_generation_config(cfg, out_root: Path) -> GenerationConfig:
    cfg_full_gen_eval = cfg.get('full_gen_eval', cfg)
    M = int(cfg_full_gen_eval.get('ensemble_size', cfg.data_handling.get('n_gen_samples', 32)))
    edm = cfg.get('edm', {})

    return GenerationConfig(
        output_root=str(out_root),
        ensemble_size=M,
        sampler_steps=int(edm.get('sampling_steps', 40)),
        seed=int(cfg_full_gen_eval.get('seed', 1234)),
        use_edm=bool(edm.get('enabled', True)),
        sigma_min=float(edm.get('sigma_min', 0.002)),
        sigma_max=float(edm.get('sigma_max', 80.0)),
        rho=float(edm.get('rho', 7.0)),
        S_churn=float(edm.get('S_churn', 0.0)),
        S_min=float(edm.get('S_min', 0.0)),
        S_max=float(edm.get('S_max', float('inf'))),
        S_noise=float(edm.get('S_noise', 1.0)),
        predict_residual=bool(edm.get('predict_residual', False)),
        save_space="physical",
        max_dates=int(cfg_full_gen_eval.get('max_dates', -1)),
    )

def generation_sigma_grid_main(cfg):
    """
    Generate ensembles across a grid of sigma_star values.
    For each sigma_star, outputs go to:
      <sample_dir>/generation/<model_name>/sigma_star=<val>/
    """
    # ----------------------- Seed -----------------------
    seed = int(cfg.full_gen_eval.get('seed', 1234))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # ----------------------- Device -----------------------
    device = cfg.training.device

    # ----------------------- Model & checkpoint -----------------------
    model, ckpt_dir, ckpt_name = get_model(cfg)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "network_params" not in ckpt:
        raise KeyError(f"Checkpoint missing 'network_params': {ckpt_path}")
    model.load_state_dict(ckpt["network_params"])
    model.eval()
    logger.info(f"[generation_sigma_grid_main] Loaded checkpoint: {ckpt_path}")

    # ----------------------- Data (deterministic) -----------------------
    cfg.setdefault('data_handling', {})
    cfg.data_handling['split'] = 'test'
    cfg.data_handling['batch_size'] = 1
    cfg.data_handling['shuffle'] = False
    cfg.data_handling['drop_last'] = False
    gen_dataloader = get_final_gen_dataloader(cfg)

    # ----------------------- Sigma* values -----------------------
    grid = cfg.get('full_gen_eval', {}).get('sigma_star_grid', [1.0])
    if isinstance(grid, (float, int)):
        grid = [float(grid)]
    grid = [float(x) for x in grid]
    logger.info(f"[generation_sigma_grid_main] sigma_star_grid = {grid}")

    # ----------------------- Sigma* ramp settings (optional late-step control) -----------------------
    scfg = cfg.get('full_gen_eval', {}).get('sigma_control', {})
    ramp_mode = str(scfg.get('sigma_star_mode', cfg.get('edm', {}).get('sigma_star_mode', 'global')))
    ramp_start_frac = float(scfg.get('ramp_start_frac', cfg.get('edm', {}).get('ramp_start_frac', 0.60)))
    ramp_end_frac   = float(scfg.get('ramp_end_frac',   cfg.get('edm', {}).get('ramp_end_frac',   0.85)))
    ramp_start_sigma = scfg.get('ramp_start_sigma', cfg.get('edm', {}).get('ramp_start_sigma', None))
    ramp_end_sigma   = scfg.get('ramp_end_sigma',   cfg.get('edm', {}).get('ramp_end_sigma',   None))

    # ----------------------- Base output -----------------------
    base_out = _resolve_base_out_dir(cfg)

    # ----------------------- Loop over sigma* -----------------------
    for sstar in grid:
        # 1) Set effective sigma_star in config (read by GenerationRunner via edm_cfg)
        cfg.setdefault('edm', {})
        cfg.edm['sigma_star'] = float(sstar)
        # --- Push ramp settings into cfg.edm for sampler ---
        cfg.edm['sigma_star_mode'] = ramp_mode
        cfg.edm['ramp_start_frac'] = ramp_start_frac
        cfg.edm['ramp_end_frac']   = ramp_end_frac
        cfg.edm['ramp_start_sigma'] = ramp_start_sigma
        cfg.edm['ramp_end_sigma']   = ramp_end_sigma

        # 2) Subdir for this sigma*
        subdir = base_out / f"sigma_star={sstar:.2f}"
        subdir.mkdir(parents=True, exist_ok=True)

        # 3) Build runner config, pointing to subdir
        gen_cfg = _build_generation_config(cfg, subdir)

        # 4) Run generator
        logger.info(f"[generation_sigma_grid_main] Generating for sigma_star={sstar:.2f} -> {subdir}")
        runner = GenerationRunner(model=model, cfg=cfg, device=device, out_root=subdir, gen_config=gen_cfg)
        runner.run(gen_dataloader)

    logger.info(f"[generation_sigma_grid_main] Done. Outputs at: {base_out}")
    return base_out