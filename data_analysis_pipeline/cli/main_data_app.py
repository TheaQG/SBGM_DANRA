import argparse
import logging
import os
import sys

from data_analysis_pipeline.cli.launch_split_creation import run as run_split
from data_analysis_pipeline.cli.launch_statistics import run as run_statistics
from data_analysis_pipeline.cli.launch_comparison import run as run_comparison
from data_analysis_pipeline.cli.launch_small_batches_creation import run as run_small_batches
from sbgm.utils import load_config


print('>>> Entered main_data_app.py')

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def main():
    logger.info("In main data app: Setting up to run...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["create_splits", "run_statistics", "run_comparison", "create_small_batches"], required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == "create_splits":
        logger.info("\t\tIn main data app: Creating data splits...")
        run_split(cfg)
    elif args.mode == "run_statistics":
        logger.info("\t\tIn main data app: Running data statistics...")
        run_statistics(cfg)
    elif args.mode == "run_comparison":
        logger.info("\t\tIn main data app: Running data comparison...")
        run_comparison(cfg)
    elif args.mode == "create_small_batches":
        logger.info("\t\tIn main data app: Creating small data batches...")
        run_small_batches(cfg)
    else: 
        raise ValueError(f"Unknown mode: {args.mode}")

    print("Finished run...")

if __name__ == "__main__":
    main()