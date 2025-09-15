""" 
    UNIFIED CLI INTERFACE FOR SBGM_SD

    main_app.py
    This script serves as the main control point for the full SBGM_SD application.
    Tasks implemented:
        - Running the training process
        - Running the generation process on a trained model
        - Running the evaluation process from generated samples
        - Full model pipeline: training --> generation --> evaluation
        - 

    Tasks to be implemented:
        - Data structuring (train/test/eval splits)
        - Running full Dataset statistics based on config
"""
import argparse
import os
import logging



from sbgm.utils import get_model_string, load_config
from sbgm.logging_utils import (
    cfg_hash, make_run_name, ensure_run_dir,
    setup_logging, write_run_manifest, log_banner
)


def check_model_exists(cfg):
    model_name = get_model_string(cfg)
    ckpt_dir = os.path.join(cfg.paths.checkpoint_dir, model_name)
    return os.path.exists(ckpt_dir) and any(f.endswith(".pth.tar") for f in os.listdir(ckpt_dir))

def check_generated_samples_exist(cfg):
    model_name = get_model_string(cfg)
    gen_dir = os.path.join(cfg.paths.sample_dir, "generation", model_name, "generated_samples")
    return os.path.exists(gen_dir) and any(f.startswith("gen_samples") for f in os.listdir(gen_dir))



# Use setup_logger and write_run_manifest in main
def main():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="SBGM full pipeline launcher")
    parser.add_argument("--config_path", required=True, help="Path to the yaml config")
    parser.add_argument("--mode", choices=["train", "generate", "evaluate", "full_pipeline", "data_splits"], default="full_pipeline")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="If set, no actual training/generation/evaluation will be performed, only config parsing and logging setup.")
    args = parser.parse_args()


    cfg = load_config(args.config_path)

    # === Build run context ===
    model_name = get_model_string(cfg)
    h = cfg_hash(cfg)
    run_name = make_run_name(cfg.experiment.name, h)
    run_dir = ensure_run_dir(cfg.paths.log_dir, model_name)

    # === Logging + manifest ===
    from omegaconf import OmegaConf
    cfg_py = OmegaConf.to_container(cfg, resolve=True) # Convert to plain dict for logging
    file_level = getattr(cfg, "logging", {}).get("file_level", "INFO")
    console_level = getattr(cfg, "logging", {}).get("console_level", "WARNING")
    log_path = setup_logging(run_dir, run_name,
                             file_level=getattr(cfg, "logging", {}).get("file_level", "INFO"),
                             console_level=getattr(cfg, "logging", {}).get("console_level", "WARNING")
                             )
    logger.info("Unified log file: %s", log_path) # This line should appear in both log file and SLURM .out

    write_run_manifest(run_dir, run_name, cfg, model_name)
    
    logger.info("=== ENTERED SBGM_SD MAIN APP ===")
    logger.info("Experiment      : %s", cfg.experiment.name)
    logger.info("Mode            : %s", args.mode)
    logger.info("Config          : %s", args.config_path)
    logger.info("Run dir         : %s", run_dir)
    logger.info("Log file        : %s", log_path)
    logger.info("Model key       : %s", model_name)
    logger.info("Cfg hash        : %s", h)

    # Imports kept here to avoid circular imports
    from sbgm.cli import launch_sbgm, launch_generation, launch_evaluation
    from data_analysis_pipeline.cli import launch_split_creation

    # === Dispatch with banners ===
    if args.mode == "data_splits":
        log_banner("DATA SPLIT CREATION START")
        launch_split_creation.run(cfg)
        log_banner("DATA SPLIT CREATION DONE")

    if args.mode == "train":
        log_banner("TRAINING START")
        launch_sbgm.run(cfg)
        log_banner("TRAINING DONE")

    elif args.mode == "generate":
        log_banner("GENERATION START")
        if not check_model_exists(cfg):
            raise RuntimeError("Cannot generate: model checkpoint not found")
        launch_generation.run(cfg)
        log_banner("GENERATION DONE")

    elif args.mode == "evaluate":
        log_banner("EVALUATION START")
        if not check_generated_samples_exist(cfg):
            raise RuntimeError("Cannot evaluate: generated samples not found.")
        launch_evaluation.run(cfg)
        log_banner("EVALUATION DONE")

    elif args.mode == "full_pipeline":
        log_banner("TRAINING START")
        if not args.skip_train:
            launch_sbgm.run(cfg)
        elif not check_model_exists(cfg):
            raise RuntimeError("Cannot skip training: no trained model found.")
        log_banner("TRAINING DONE")

        log_banner("GENERATION START")
        if not args.skip_generation:
            launch_generation.run(cfg)
        elif not check_generated_samples_exist(cfg):
            raise RuntimeError("Cannot skip generation: no samples found.")
        log_banner("GENERATION DONE")

        log_banner("EVALUATION START")
        if not args.skip_evaluation:
            launch_evaluation.run(cfg)
        log_banner("EVALUATION DONE")

        
    logger.info("=== SBGM_SD MAIN APP DONE ===")


if __name__ == "__main__":
    main()