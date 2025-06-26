""" 
    main_app.py
    This script serves as the main control point for the full SBGM_SD application.
    Tasks implemented:
        - Running the training process
        - Running data filtering process


    UNIFIED CLI INTERFACE FOR SBGM_SD
"""
import argparse
from sbgm.cli import launch_sbgm, launch_generation


def main():
    parser = argparse.ArgumentParser(description="SBGM_SD Unified CLI Interface")
    parser.add_argument("task", choices=["train", "generate"])
    args = parser.parse_args()

    if args.task == "train":
        launch_sbgm.run()
    elif args.task == "generate":
        launch_generation.run()
    else:
        raise ValueError(f"Unknown task: {args.task}. ")

if __name__ == "__main__":
    main()