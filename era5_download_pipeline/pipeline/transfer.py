'''
    Transfer data to a remote server using rsync.
'''

import subprocess
import pathlib

from .utils import ensure_dir

def rsync_push(local_dir:pathlib.Path,
               remote_dir:str,
               cfg):
    ensure_dir(local_dir)
    ssh = f"ssh -i {cfg['lumi_key']}"

    proc = ["rsync", # rsync command, faster than scp for large transfers
            "--progress", # Show progress during transfer
            "-avz", # Archive mode, verbose, compress files during transfer
            "--remote-source-files", # Allow remote source files to be used
            "-e", # Specify the remote shell to use
            ssh, # Use SSH for the transfer
            str(local_dir) + "/", # Local directory to sync (ensure trailing slash for directory)
            # Remote directory to sync to (ensure trailing slash for directory)
            f"{cfg['lumi']['user']}@{cfg['lumi']['host']}:{remote_dir}/"
            ]

    subprocess.run(proc, check=True)
