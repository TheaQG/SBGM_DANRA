#!/bin/bash
##########################################################################################################
# run_optuna_sweep_lumi.sh - one-trial-per-GPU Optuna sweep on LUMI
# 
# - Uses a SLURM *array*: each task trains **exactly one** trial
# - All tasks share the same optuna study via a SQLite Database on $SCRATCH
# - Keeps existing singularity container and env-vars
##########################################################################################################

#SBATCH --job-name=optuna_sweep                 # Optuna sweep job name
#SBATCH --array=0-199%64                        # 200 trials, max 64 concurrent
#SBATCH --output=logs/optuna_sweep_%A_%a.log    # Output log file
#SBATCH --error=logs/optuna_sweep_%A_%a.err     # Error log file
#SBATCH --account=project_465001695             # Project account
#SBATCH --partition=standard-g                  # Standard GPU partition
#SBATCH --nodes=1                               # Use one node
#SBATCH --gpus-per-node=1                       # One GPU per trial
#SBATCH --cpus-per-task=7                       # 7 CPU cores per GPU is a good rule
#SBATCH --mem-per-gpu=60G                       # Memory per GPU
#SBATCH --time=12:00:00                         # Maximum time limit


# --- Environment setup ------------------------------------------------
module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_465001695/containers/images/my_torch_container_with_plotting.sif

SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
USER_DIR=$SCRATCH/$USER
export ROOT_DIR="$USER_DIR/Code/SBGM_SD"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# --- Data and output paths --------------------------------------------
export DATA_DIR=$USER_DIR/Data/Data_DiffMod # Data_DiffMod_small
export SAMPLE_DIR="$ROOT_DIR/models_and_samples/generated_samples"
export CKPT_DIR="$ROOT_DIR/models_and_samples/trained_models"

mkdir -p logs

# --- Optuna shared storage (SQLite over Lustre/Scratch) ---------------
OPTUNA_DB_DIR=$SCRATCH/optuna_db
mkdir -p $OPTUNA_DB_DIR
STUDY_NAME=sbgm_optuna_v1
STORAGE="sqlite:///$OPTUNA_DB_DIR/$STUDY_NAME.db"

echo "[INFO] Using Optuna storage: $STORAGE"
echo "[INFO] Starting trial for array-id: $SLURM_ARRAY_TASK_ID"

# --- Launch exactly ONE trial in this task ----------------------------
srun singularity exec $CONTAINER \
    python -m sbgm.sweep.run_optuna \
    --n-trials 1 \
    --study-name $STUDY_NAME \
    --storage "$STORAGE" \
    --ebable-medium \
    --epochs 3 
