'''
    Transfer data to a remote server using rsync.
'''

import subprocess
import pathlib
from .utils import ensure_dir

import logging
logger = logging.getLogger(__name__)

def rsync_push(local_dir:pathlib.Path,
               remote_dir:str,
               cfg,
               delete=True
               ):
    local_path = pathlib.Path(local_dir).expanduser().resolve()  # Expand user and resolve to absolute path
    remote_dir = str(remote_dir)

    # Build the ssh command
    ssh_parts = ["ssh", "-o", "IdentitiesOnly=yes"]
    if cfg.get('lumi_key'):
        ssh_parts += ["-i", cfg['lumi_key']]  # Use the specified SSH key if provided
    ssh_cmd = " ".join(ssh_parts) # Join the SSH command parts into a single string
    
    rsync_cmd = [
        "rsync", # rsync command, faster than scp for large transfers
        "-avz", # Archive mode, verbose, compress files during transfer
        "--remove-source-files" if delete else "", # Remove source files after transfer if delete is True
        "--progress", # Show progress during transfer
        "-e", ssh_cmd, # Use SSH for the transfer
        str(local_path) + ("/" if local_path.is_dir() else ""), # Local directory to sync (ensure trailing slash for directory)
        f"{cfg['lumi']['user']}@{cfg['lumi']['host']}:{remote_dir}/" # Remote directory to sync to (ensure trailing slash for directory)
    ]

    # Debug logging
    logger.debug(f"Running rsync command: {' '.join(rsync_cmd)}")
    result = subprocess.run(rsync_cmd, check=True, capture_output=True, text=True)

    # Check if the command was successful
    if result.returncode:
        logger.error(f"Rsync failed with return code {result.returncode}")
        logger.error(f"Error output: {result.stderr.strip()}")
        raise RuntimeError(result.stderr.strip())