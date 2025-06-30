"""
    Download --> rsync to LUMI --> delete in a streaming fashion.
    Each worker handles exactly one <variable, year> combination.
"""

import pathlib # For file paths
import concurrent.futures # For parallel execution
from . import download, transfer # Import download and transfer functions

import logging
logger = logging.getLogger(__name__)

def _job(args):
    """
    Worker function to download and transfer data for a specific variable and year.
    """
    var_long, vshort, yr, cfg = args
    tmp_dir = pathlib.Path(cfg['tmp_dir']) / vshort
    tmp_dir.mkdir(parents=True, exist_ok=True)  # Ensure the temporary directory exists

    out_nc = tmp_dir / f"{vshort}_{yr}.nc"
    remote_dir = cfg['lumi']['raw_dir'].format(var=vshort)

    logger.info(f"Starting job for {var_long} {yr} to {out_nc}")
    # 1) Download
    download.download_year(var_long, yr, out_nc, cfg)
    logger.info(f"Downloaded {var_long} for {yr} to {out_nc}")

    # 2) Transfer and auto-delete
    logger.info(f"Transferring {out_nc} to {remote_dir}")
    try:
        transfer.rsync_push(out_nc, remote_dir, cfg)
    except Exception as e:
        logger.error(f"Failed to transfer {out_nc} to {remote_dir}: {e}")
        raise
    transfer.rsync_push(out_nc, remote_dir, cfg)
    logger.info(f"Transferred {out_nc} to {remote_dir}")

    # 3) Extra safety: Remove the local file after transfer
    if out_nc.exists():
        out_nc.unlink()

    # Remove empty directory if it exists
    if not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()


def download_transfer_delete(cfg, n_workers=2):
    """
    Download ERA5 data, transfer it to LUMI, and delete the local copy.
    
    Args:
        cfg (dict): Configuration dictionary containing settings for download and transfer.
        n_workers (int): Number of parallel workers to use for downloading and transferring data.
    """
    jobs = []
    for var_long, vinfo in cfg['variables'].items():
        vshort = vinfo['short']
        for year in range(cfg['years'][0], cfg['years'][1] + 1):
            jobs.append((var_long, vshort, year, cfg))

    # ThreadPool is fine: cdsapi and rsync is I/O bound, not CPU bound
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Every worker build its own cdsapi.Client instance, so no cross-thread sharing
        executor.map(_job, jobs)
        # Log the number of jobs

    # Log completion
    logger.info(f"All downloads and transfers completed for {len(jobs)} jobs.")
