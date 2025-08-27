import argparse
import logging
import os
import sys

from data_analysis_pipeline.cli.launch_split_creation import run as run_split
from data_analysis_pipeline.cli.launch_statistics import run as run_statistics
from sbgm.utils import load_config


print('>>> Entered main_data_app.py')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    logger.info("Setting up to run...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["create_splits", "run_statistics"])
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "create_splits":
        logger.info("Creating data splits...")
        run_split(cfg)
    elif args.mode == "run_statistics":
        logger.info("Running data statistics...")
        run_statistics(cfg)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print("Finished run...")

if __name__ == "__main__":
    main()