# scripts/env_setup.sh

#!/bin/bash

export ROOT_DIR=$SCRATCH/$USER/Code/SBGM_new
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

export DATA_DIR=$SCRATCH/$USER/Data/Data_DiffMod/
export CKPT_DIR=$SCRATCH/$USER/Checkpoints/Checkpoints_SBGM/
export FIGS_DIR=$SCRATCH/$USER/Samples/Data_figures/
export STATS_DIR=$SCRATCH/$USER/Samples/Data_stats

export TORCH_HOME=$SCRATCH/torch-cache
export HF_HOME=/flash/$USER/hf-cache
mkdir -p $TORCH_HOME $HF_HOME