""" 
    main_app.py
    This script serves as the main control point for the full SBGM_SD application.
    Tasks implemented:
        - Running the training process
        - Running data filtering process


    UNIFIED CLI INTERFACE FOR SBGM_SD
"""
import argparse
from sbgm.cli.launch_sbgm import run
from sbgm.data_scripts import data_filter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help='Task to run (e.g., train, evaluate)')
    args = parser.parse_args()

    if args.task == 'train':
        run()
    elif args.task == "filter":
        data_filter.run()
    else:
        raise ValueError(f"Unknown task: {args.task}. ")
    
if __name__ == "__main__":
    main()