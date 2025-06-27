'''
    This script is used to run the ERA5 download pipeline locally.
'''

import pathlib
from pathlib import Path
import yaml
from era5_download_pipeline.pipeline import download, transfer


_this_dir = Path(__file__).resolve().parent # ../cli
cfg_path = _this_dir.parent / "cfg" / "era5_pipeline_testing.yaml"
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

# 1. download all
download.pull_all(cfg)

# 2. push each variable folder, then delete it
for v in cfg['variables'].values():
    local = pathlib.Path(cfg['tmp_dir']) / v['short']
    remote = cfg['lumi']['raw_dir'].format(var=v['short'])
    transfer.rsync_push(local, remote, cfg)