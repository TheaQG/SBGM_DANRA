#!/bin/bash

# === Activate virtual env if needed ===
# source /Users/au728490/opt/anaconda3/envs/torch
source bash_scripts/env_setup.sh

# === Print the directories to verify ===
echo "ROOT_DIR: $ROOT_DIR"
echo "DATA_DIR: $DATA_DIR"
echo "SAMPLE_DIR: $SAMPLE_DIR"
echo "CKPT_DIR: $CKPT_DIR"

# === Launch the training ===
python -m sbgm.cli.main_app train