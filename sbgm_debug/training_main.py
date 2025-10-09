# sbgm/training_main.py
import os
import torch
import logging
import random

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional

from sbgm_debug.training_utils import get_model_string, get_model, get_optimizer, get_dataloader, get_scheduler
from sbgm_debug.plotting_utils import plot_sample
from sbgm_debug.training import TrainingPipeline_general
from sbgm_debug.score_unet import marginal_prob_std_fn, diffusion_coeff_fn

# Set up logging
logger = logging.getLogger(__name__)


# === Typed params (read once) ===
@dataclass(frozen=True)
class TrainParams:
    experiment_name: str
    device: str
    seed: int
    epochs: int
    batch_size: int
    verbose: bool
    mixed_precision: bool
    load_checkpoint: Optional[str]
    eval_use_ema: bool

    # paths
    path_save: str
    training_sample_dir: str
    log_dir: str
    checkpoint_dir: str

    # visualization (lightweight flags only, not full cfg)
    plot_initial_sample: bool
    show_figs: bool

    @staticmethod
    def from_cfg(cfg: dict) -> "TrainParams":
        t = cfg["training"]
        p = cfg["paths"]
        v = cfg.get("visualization", {})
        exp = cfg.get("experiment_name", cfg.get("experiment", {}).get("name", "experiment"))
        ema = t.get("ema", {})
        return TrainParams(
            experiment_name = exp,
            device          = t.get("device", "cuda"),
            seed            = t.get("seed", 0),
            epochs          = int(t.get("epochs", 1)),
            batch_size      = int(t.get("batch_size", 1)),
            verbose         = bool(t.get("verbose", True)),
            mixed_precision = bool(t.get("mixed_precision", t.get("use_mixed_precision", True))),
            load_checkpoint = (t.get("load_checkpoint") if isinstance(t.get("load_checkpoint"), str) else None),
            eval_use_ema    = bool(ema.get("eval_use_ema", True)),
            path_save           = p.get("path_save", "."),
            training_sample_dir = p.get("training_sample_dir", os.path.join(p.get("path_save", "."), "samples")),
            log_dir             = p.get("log_dir", "./logs"),
            checkpoint_dir      = p.get("checkpoint_dir", "./checkpoints"),
            plot_initial_sample = bool(v.get("plot_initial_sample", True)),
            show_figs           = bool(v.get("show_figs", False)),
        )

# === Wiring / boundary ===
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# === Function for building and running the training pipeline ===
def build_training_run(cfg: dict):
    """ Read cfg once, construct all dependencies, return a bundle. """
    P = TrainParams.from_cfg(cfg)

    # Directories
    model_str = get_model_string(cfg)

    # Prefer explicit training_sample_dir if provided; otherwise derive from path_save
    base_samples_root = P.training_sample_dir or os.path.join(P.path_save, "samples")
    samples_dir = os.path.join(base_samples_root, model_str)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(P.log_dir, exist_ok=True)
    os.makedirs(P.checkpoint_dir, exist_ok=True)

    # Reproducibility
    set_global_seed(P.seed)

    # Device
    device = torch.device("cuda" if P.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    # Build dependencies using existing helpers
    model, _, _ = get_model(cfg)
    model = model.to(device)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer) if "learning_rate" in cfg.get("training", {}) else None
    train_dataloader, val_dataloader, gen_dataloader = get_dataloader(cfg)

    # Trainer/pipeline
    pipeline = TrainingPipeline_general(
        model=model,
        marginal_prob_std_fn=marginal_prob_std_fn,
        diffusion_coeff_fn=diffusion_coeff_fn,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        cfg=cfg
    )

    # NOTE: Below this return boundary: DO NOT ACCESS RAW CFG AGAIN!
    return P, pipeline, train_dataloader, val_dataloader, gen_dataloader, samples_dir





def train_main(cfg: dict):
    """
    Thin training entrypoint that respects the config boundary:
        - Parse cfg once (handled in build_training_run)
        - No raw cfg access below the boundary
        - Uses TrainingPipeline_general which already parsed & cached settings
    """

    logger.info("\n\n=== Starting SBGM_SD Training Pipeline ===")

    # Build everything once
    P, pipeline, train_dataloader, val_dataloader, gen_dataloader, samples_dir = build_training_run(cfg)
    
    logger.info(f"          Experiment name: {P.experiment_name}")
    if torch.cuda.is_available() and P.device.startswith("cuda"):
        try:
            logger.info(f"          ▸ Using CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception:
            logger.info("          ▸ Using CUDA device")
    else:
        logger.info("          ▸ Using CPU device")

    # Peek at the first dataset element (debug-friendly, no cfg usage)
    sample0 = None
    try:
        sample0 = train_dataloader.dataset[0]
        for key, value in sample0.items():
            try:
                logger.info(f"          {key}: {getattr(value, 'shape', None)}")
            except Exception:
                logger.info(f"          {key}: {type(value)}")

        # Optional quick-look plot of the initial sample (best-effort, no crash on failure)
        if P.plot_initial_sample:
            try:
                fig, _ = plot_sample(sample0, cfg)  # plotting_utils currently expects cfg; safe to pass here only
                save_path = os.path.join(samples_dir, 'Figures')
                os.makedirs(save_path, exist_ok=True)
                fig.savefig(os.path.join(save_path, 'Initial_sample_plot.png'), bbox_inches='tight', dpi=300)
                if P.show_figs:
                    plt.show()
                else:
                    plt.close(fig)
                logger.info(f"\n\n          ▸ Saved initial sample plot to {os.path.join(save_path, 'Initial_sample_plot.png')}")
            except Exception as e:
                logger.warning(f"          ▸ Initial sample plotting failed (continuing): {e}")
    except Exception as e:
        logger.warning(f"          ▸ Could not inspect first training sample: {e}")

    # Start training (no raw cfg below)
    logger.info(f"\n\n          === STARTING TRAINING MAIN LOOP ===\n")
    pipeline.train(
        train_dataloader,
        val_dataloader,
        gen_dataloader,
        epochs=P.epochs,
        verbose=P.verbose,
        use_mixed_precision=P.mixed_precision,
    )
    logger.info(f"\n\n          === TRAINING COMPLETE ===\n")


















