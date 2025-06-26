# scripts/env_setup.sh

#!/bin/bash

# === Set absolute paths ===
export ROOT_DIR="/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_SD"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# === Point to small dataset ===
export DATA_DIR="/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/Data/data_DiffMod_small"

# === Define output/save locations ===
export SAMPLE_DIR="$ROOT_DIR/models_and_samples/generated_samples"
# export FIGURE_DIR="$ROOT_DIR/models_and_samples/figures"
export CKPT_DIR="$ROOT_DIR/models_and_samples/trained_models"
