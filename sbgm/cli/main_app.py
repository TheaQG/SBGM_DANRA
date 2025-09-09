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
import sys

from sbgm.cli import launch_sbgm, launch_generation, launch_evaluation
from data_analysis_pipeline.cli import launch_split_creation
from sbgm.utils import get_model_string, load_config


def check_model_exists(cfg):
    model_name = get_model_string(cfg)
    ckpt_dir = os.path.join(cfg.paths.checkpoint_dir, model_name)
    return os.path.exists(ckpt_dir) and any(f.endswith(".pth.tar") for f in os.listdir(ckpt_dir))

def check_generated_samples_exist(cfg):
    model_name = get_model_string(cfg)
    gen_dir = os.path.join(cfg.paths.sample_dir, "generation", model_name, "generated_samples")
    return os.path.exists(gen_dir) and any(f.startswith("gen_samples") for f in os.listdir(gen_dir))



def main():
    parser = argparse.ArgumentParser(description="SBGM full pipeline launcher")
    parser.add_argument("--config_path", required=True, help="Path to the yaml config")
    parser.add_argument("--mode", choices=["train", "generate", "evaluate", "full_pipeline", "data_splits"], default="full_pipeline")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config_path)

    if args.mode == "split":
        launch_split_creation.run(cfg)

    if args.mode == "train":
        launch_sbgm.run(cfg)

    elif args.mode == "generate":
        if not check_model_exists(cfg):
            raise RuntimeError("Cannot generate: model checkpoint not found")
        launch_generation.run(cfg)
        
    elif args.mode == "evaluate":
        if not check_generated_samples_exist(cfg):
            raise RuntimeError("Cannot evaluate: generated samples not found.")
        launch_evaluation.run(cfg)

    elif args.mode == "full_pipeline":
        if not args.skip_train:
            launch_sbgm.run(cfg)
        elif not check_model_exists(cfg):
            raise RuntimeError("Cannot skip training: no trained model found.")

        if not args.skip_generation:
            launch_generation.run(cfg)
        elif not check_generated_samples_exist(cfg):
            raise RuntimeError("Cannot skip generation: no samples found.")

        if not args.skip_evaluation:
            launch_evaluation.run(cfg)

    print("\nPipeline finished successfully.")





# def main():
#     parser = argparse.ArgumentParser(description="SBGM_SD Unified CLI Interface")
#     parser.add_argument("task", choices=["train", "generate", "evaluate"])
#     args = parser.parse_args()

#     if args.task == "train":
#         launch_sbgm.run()
#     elif args.task == "generate":
#         launch_generation.run()
#     elif args.task == "evaluate":
#         launch_evaluation.run()
#     else:
#         raise ValueError(f"Unknown task: {args.task}. ")

if __name__ == "__main__":
    main()