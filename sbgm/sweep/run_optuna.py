#! /usr/bin/env python3
"""
    run_optuna.py - Bayesian hyperparameter optimization for the SBGM for climate variable downscaling
    -------------------------------------------------------------
    - Works locally *or* on SLURM clusters via a SLURM array (one trial per task).
    - Samples **high-impact** hyperparameters by default. Add --enable-medium to include am additional medium-impact block.
    - Saves the fully-materialised YAML config for every trial to 'configs/generated/<trial_id>.yaml'.
      so that you can run the SBGM with the best hyperparameters later.

    Usage (local):
        python run_optuna.py --n-trials 100 --study-name sbgm_baseline \
            --enable-medium True --storage sqlite://sbgm_baseline.db

    Usage (SLURM):
    python run_optuna.py --n_trials 1 --study-name sbgm_hpc \
        --enable-medium True --storage grpc://lumi-login01:50051 \
"""

import argparse
import yaml
import pathlib

import torch
import os
import tempfile

import random
import numpy as np

from datetime import datetime

from functools import partial

import optuna
from optuna.samplers import GPSampler
from optuna.pruners import SuccessiveHalvingPruner

import logging
logger = logging.getLogger(__name__)


# ==========================================================
# Paths and constants, and helpers
# ==========================================================

# Base dir is this file's parent directory
BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_CFG_PATH = BASE_DIR / "cfg/default_config.yaml"
GENERATED_CFG_DIR = BASE_DIR / "configs/generated"
GENERATED_CFG_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

# Fixes reproducibility inside each trial
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Flattened-key update utility (e.g. "training.learning_rate")
def deep_update(cfg: dict, flat_kv: dict):
    for flat_key, value in flat_kv.items():
        node = cfg  # Start at the root of the config
        *parents, leaf = flat_key.split('.')
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value  # Set the final value in the nested structure

# ===========================================================
# Search-space helpers - to be tweaked
# ===========================================================

def sample_high_impact(trial: optuna.trial.Trial) -> dict:
    """
        Sample high-impact hyperparameters for the SBGM.
        Return a dict {flattened.key: value} with the highest-impact hyperparameters.
    """
    return {
        # Optimiser/learning dynamics
        "training.leargning_rate":      trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "training.optimizer":           trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        "sampler.n_timesteps":          trial.suggest_int("n_steps", 200, 1500),
        "sampler.time_embedding":       trial.suggest_int("t_embed", 128, 512, step=64),
        "classifier_free_guidance.guidance_scale":
                                        trial.suggest_float("cfg_scale", 0.5, 6.0, log=True),
        "sampler.block_layers":         trial.suggest_categorical("block_layers", [[2, 2, 2, 2], [3, 3, 3, 3], [3, 4, 6, 3]]),
        "sampler.num_heads":            trial.suggest_int("n_heads", 2, 16, step=2),
    }

def sample_medium_impact(trial: optuna.trial.Trial) -> dict:
    """
        Sample medium-impact hyperparameters for the SBGM.
        Return a dict {flattened.key: value} with the medium-impact hyperparameters.
    """
    return {
        "training.batch_size":          trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
        "training.ema_decay":           trial.suggest_float("ema_decay", 0.95, 0.9999),
        "training.weight_decay":        trial.suggest_float("wd", 0.0, 1e-3, log=True),
        "sampler.last_fmap_channels":   trial.suggest_categorical("last_fmap_channels", [256, 512, 768]),
    }

def sample_low_impact(trial: optuna.trial.Trial) -> dict:
    """
        Sample low-impact hyperparameters for the SBGM.
        Return a dict {flattened.key: value} with the low-impact hyperparameters.
    """
    return {
        "data.input_channels": trial.suggest_int("input_channels", 1, 3),
        "data.output_channels": trial.suggest_int("output_channels", 1, 3),
        "model.use_residual": trial.suggest_categorical("use_residual", [True, False]),
        "model.use_attention": trial.suggest_categorical("use_attention", [True, False]),
    }

# ===========================================================
# Utility - merge dot-notation keys into nested dict
# ===========================================================

def compose_cfg(base_cfg: dict, flat_params:dict) -> dict:
    """
        Insert flattened.dot.keys into a nested YAML structure (in-place copy)
    """

    import copy
    
    cfg = copy.deepcopy(base_cfg)  # Start with a copy of the base config
    for key, value in flat_params.items():
        node = cfg # Start at the root of the config
        *path, last = key.split('.') # Split the key into path components
        for p in path:
            node = node.setdefault(p, {})
        node[last] = value  # Set the final value in the nested structure
    return cfg

# ===========================================================
# Optuna objective - one *trial*
# ===========================================================

def objective(trial: optuna.trial.Trial,
              enable_medium: bool = False,
              enable_low: bool = False,
              n_epochs: int = None) -> float:
    
    # 1. Assemble a concrete config dict
    cfg = yaml.safe_load(open(BASE_DIR.parent / "cfg/default_config.yaml"))
    deep_update(cfg, sample_high_impact(trial))  # Always sample high-impact
    if enable_medium:
        deep_update(cfg, sample_medium_impact(trial))
    if enable_low:
        deep_update(cfg, sample_low_impact(trial))

    # Allow per-trial shortening of epochs for multi-fidelity pruning
    if n_epochs:
        cfg['training']['epochs'] = n_epochs

    # 2. Persist concrete YAML for reproducibility
    out_cfg = GENERATED_CFG_DIR / f"trial_{trial.number:05d}.yaml"
    yaml.safe_dump(cfg, open(out_cfg, 'w'))

    # 3. Reproducibility: set seeds
    set_seed(cfg["training"].get("seed", 42) + trial.number)

    # 4. Build a model, data, optimiser, and training pipeline
    from sbgm.score_unet import loss_fn, marginal_prob_std_fn, diffusion_coeff_fn, ScoreNet
    from sbgm.training_utils import get_dataloader, get_optimizer, get_scheduler

    model = ScoreNet(
        input_channels=cfg['data']['input_channels'],
        output_channels=cfg['data']['output_channels'],
        n_timesteps=cfg['sampler']['n_timesteps'],
        time_embedding_dim=cfg['sampler']['time_embedding'],
        block_layers=cfg['sampler']['block_layers'],
        num_heads=cfg['sampler']['num_heads'],
        last_fmap_channels=cfg['sampler']['last_fmap_channels'],
        use_residual=cfg['model'].get('use_residual', True),
        use_attention=cfg['model'].get('use_attention', True),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = get_optimizer(
        model,
        cfg['training']['optimizer'],
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training'].get('weight_decay', 0.0),
    )

    scheduler = get_scheduler(
        optimizer,
        cfg['training'].get('scheduler', None),
        n_epochs=cfg['training']['epochs'],
        n_steps_per_epoch=cfg['data']['n_steps_per_epoch'],
    )

    train_loader, val_loader, gen_loader = get_dataloader(
        cfg['data']['dataset'],
        batch_size=cfg['training']['batch_size'],
        n_timesteps=cfg['sampler']['n_timesteps'],
        input_channels=cfg['data']['input_channels'],
        output_channels=cfg['data']['output_channels'],
        shuffle=True,
        num_workers=cfg['data'].get('num_workers', 0),
    )

    from sbgm.training import TrainingPipeline_general
    pipeline = TrainingPipeline_general(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        gen_loader=gen_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        trial=trial,  # Pass the Optuna trial for logging
    )

    # 5. Train
    _, val_loss = pipeline.train(
        n_epochs=cfg['training']['epochs'],
        n_steps_per_epoch=cfg['data']['n_steps_per_epoch'],
        early_stopping_patience=cfg['training'].get('early_stopping_patience', 10),
        ema_decay=cfg['training']['ema_decay'],
        guidance_scale=cfg['classifier_free_guidance']['guidance_scale'],
    )

    # 6. Report to Optuna (lower is better). Enable pruning (i.e. early stopping) after each epoch
    trial.report(val_loss, step=cfg['training']['epochs'])
    if trial.should_prune():
        raise optuna.TrialPruned(f"Trial {trial.number} pruned at epoch {cfg['training']['epochs']} with loss {val_loss:.4f}")
    
    return val_loss

    

# ===========================================================
# Main function to run the optimization
# ===========================================================

if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser()
    p.add_argument("--study-name", default="sbgm_optuna")
    p.add_argument("--storage", default="sqlite:///sbgm_optuna.db")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--enable-medium", action="store_true")
    p.add_argument("--enable-low", action="store_true")
    p.add_argument("--epochs", type=int, default=None, help="Override cfg.training.epochs")

    args = p.parse_args()

    sampler = optuna.samplers.GPSampler(seed=42)  # Use a fixed seed for reproducibility
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)

    study = optuna.create_study(direction="minimize",
                                study_name=args.study_name,
                                storage=args.storage,
                                load_if_exists=True,
                                sampler=sampler,
                                pruner=pruner)
    
    # Partial so that Optuna only sees trial

    study.optimize(partial(objective,
                           enable_medium=args.enable_medium,
                           enable_low=args.enable_low,
                           n_epochs=args.epochs),
                   n_trials=args.n_trials)