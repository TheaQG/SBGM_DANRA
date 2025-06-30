'''
    This script is used to run the ERA5 download pipeline locally.
'''

import argparse
import pathlib
from pathlib import Path
import yaml
import os
import shutil

import logging
from era5_download_pipeline.pipeline import download, transfer, stream
from era5_download_pipeline.utils.logging_utils import setup_logging

cfg_path = pathlib.Path(__file__).resolve().parents[1] / "cfg/era5_pipeline.yaml"
cfg = yaml.safe_load(cfg_path.read_text())

parser = argparse.ArgumentParser(description="Run the ERA5 download pipeline locally.")
parser.add_argument("--log", default="era5_logs/era5_download.log",
                    help="Path to the log file. Default: era5_logs/era5_download.log")
parser.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging level. Default: INFO")
parser.add_argument("--mode", choices=["bulk", "stream"], default="bulk",
                    help="Mode of operation. 'bulk': download everything first, 'stream': download->rsyng->delete per file.")
parser.add_argument("--workers", type=int, default=2,
                    help="Threads for stream mode.")
args = parser.parse_args()


# Set up logging
setup_logging(args.log, args.log_level)

log = logging.getLogger(__name__)
log.debug(f"Configuration loaded from {cfg_path}")


if args.mode == "bulk":
    # Bulk mode: download all data first, then transfer
    download.pull_all(cfg)
    for var_long, vinfo in cfg['variables'].items():
        vshort = vinfo['short']
        tmp_dir = pathlib.Path(cfg['tmp_dir']) / vshort
        remote_dir = cfg['lumi']['raw_dir'].format(var=vshort)
        transfer.rsync_push(tmp_dir, remote_dir, cfg)
        shutil.rmtree(tmp_dir)
elif args.mode == "stream":
    # Streaming mode: download and transfer each file immediately
    stream.download_transfer_delete(cfg, n_workers=args.workers)
else:
    raise ValueError(f"Unknown mode: {args.mode}. Use 'bulk' or 'stream'.")