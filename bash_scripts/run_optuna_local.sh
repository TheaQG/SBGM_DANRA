#!/bin/bash

# === Activate virtual env if needed ===
# source /Users/au728490/opt/anaconda3/bin/activate torch # or conda activate torch

# === Set the root directory for the project ===
export ROOT_DIR="/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_SD"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# === Set environment variables for config === 
export DATA_DIR="/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/Data/data_DiffMod_small"
export SAMPLE_DIR="$ROOT_DIR/models_and_samples/generated_samples"
export CKPT_DIR="$ROOT_DIR/models_and_samples/trained_models"

# === Log the directories to verify ===
echo "[INFO] ROOT_DIR      = $ROOT_DIR"
echo "[INFO] DATA_DIR      = $DATA_DIR"
echo "[INFO] SAMPLE_DIR    = $SAMPLE_DIR"
echo "[INFO] CKPT_DIR      = $CKPT_DIR"

# === Launch the optimization ===
python -m sbgm.sweep.run_optuna \
    --study-name "sbgm_optuna_test" \
    --storage "sqlite:///sbgm_optuna_test.db" \
    --n-trials 3 \
    --epochs 1 \