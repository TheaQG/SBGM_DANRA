#!/bin/bash
################################################################################
# run_optuna_sweep_lumi.sh - one-trial-per-GPU Optuna sweep on LUMI
################################################################################

#SBATCH --job-name=sbgm_optuna_sweep
#SBATCH --account=project_465001695
#SBATCH --partition=standard-g
#SBATCH --array=0-2%2           # change range/concurrency as needed
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/optuna_%A_%a.out
#SBATCH --error=logs/optuna_%A_%a.err

# --- 0. Modules -────────────────────────────────────────────
module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# --- 1. Host-side paths -────────────────────────────────────
CONTAINER=/scratch/project_465001695/containers/images/my_torch_container_with_plotting.sif
OVERLAY=/scratch/project_465001695/containers/overlays/hpo_overlay.img
MAMBA_PREFIX=/scratch/project_465001695/micromamba      # same as in build script
HOST_CODE=/scratch/project_465001695/quistgaa/Code/SBGM_SD

SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
OPTUNA_DB_DIR=$SCRATCH/optuna_db
STUDY_NAME=sbgm_optuna_v1
STORAGE="sqlite:///$OPTUNA_DB_DIR/$STUDY_NAME.db"

mkdir -p "$OPTUNA_DB_DIR" logs

# (optional) data / model dirs visible in container
export DATA_DIR=$SCRATCH/${USER}/Data/Data_DiffMod
export SAMPLE_DIR=$HOST_CODE/models_and_samples/generated_samples
export CKPT_DIR=$HOST_CODE/models_and_samples/trained_models

echo "[INFO] Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID"
echo "[INFO] Optuna DB : $STORAGE"

# --- 2. Launch ONE trial inside Singularity -──────────────────────────────
srun singularity exec \
     --cleanenv \
     --overlay "$OVERLAY":ro \
     --bind "$HOST_CODE:/workspace" \
     --bind "$MAMBA_PREFIX":"$MAMBA_PREFIX" \
     --env MAMBA_ROOT_PREFIX="$MAMBA_PREFIX" \
     --env DATA_DIR="$DATA_DIR" \
     --env SAMPLE_DIR="$SAMPLE_DIR" \
     --env CKPT_DIR="$CKPT_DIR" \
     --env STORAGE="$STORAGE" \
     --env STUDY_NAME="$STUDY_NAME" \
     "$CONTAINER" \
     bash -eu <<'INNER'
# ---------------------- inside container + overlay --------------------------
export MAMBA_NO_LOCK=1
export XDG_CACHE_HOME=$MAMBA_ROOT_PREFIX/cache   # keeps cache on scratch
# 1) locate micromamba binary in scratch prefix
MMB="$MAMBA_ROOT_PREFIX/bin/micromamba"
if [[ ! -x "$MMB" ]]; then
    echo "[ERROR] Micromamba not found at $MMB" >&2
    exit 1
fi


# 2) discover the site-packages path of env `hpo``
HPO_SITE=$("$MMB" run -n hpo python - << 'PY'
import site, json, sys; print(site.getsitepackages()[0])
PY
)

# 3) Append overlay last -> container libs win
export PYTHONPATH="/workspace:${PYTHONPATH:-}:$HPO_SITE"



# 4) run one Optuna trial with env hpo
python -m sbgm.sweep.run_optuna \
       --n-trials 1 \
       --study-name "$STUDY_NAME" \
       --storage "$STORAGE" \
       --enable-medium \
       --epochs 3
INNER

echo "[INFO] Optuna trial completed for job $SLURM_JOB_ID | task $SLURM_ARRAY_TASK_ID"