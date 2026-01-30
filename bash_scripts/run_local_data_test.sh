#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Usage:
#   ./run_local_data_test.sh /path/to/config.yaml [split] [n] [start] [show_original] [no_cutouts]
#
# Examples:
#   ./run_local_data_test.sh sbgm/config/paper2_data_test_local.yaml train 3 0 true false
#   ./run_local_data_test.sh sbgm/config/ablations/B0s/B0_G_1.yaml test 1 0 false true
# -----------------------

CFG_PATH="${1:-}"                 # required
SPLIT="${2:-train}"               # train | valid | test
N="${3:-3}"                       # number of samples
START="${4:-0}"                   # start index
SHOW_ORIGINAL="${5:-false}"       # true | false
NO_CUTOUTS="${6:-false}"          # true | false

if [[ -z "${CFG_PATH}" ]]; then
  echo "ERROR: Configuration file path is required."
  echo "Usage: $0 /path/to/config.yaml [split] [n] [start] [show_original] [no_cutouts]"
  exit 1
fi

# Resolve config path to absolute path (pure bash)
CFG_PATH_ABS="$(cd "$(dirname "$CFG_PATH")" && pwd)/$(basename "$CFG_PATH")"

# --- Repo root discovery (script-location aware) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Threads/workers (configs use SLURM_CPUS_PER_TASK if available) ---
CPU_WORKERS="${CPU_WORKERS:-4}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-$CPU_WORKERS}"

# --- Experiment date used by configs
export EXP_DATE="${EXP_DATE:-$(date +%Y%m%d)}"

# --- Set default paths (same pattern as your run_local.sh) ---
export DATA_DIR="${DATA_DIR:-$ROOT_DIR/../../Data/data_DiffMod_small}"
export SAMPLE_DIR="${SAMPLE_DIR:-$ROOT_DIR/models_and_samples/generated_samples}"
export CKPT_DIR="${CKPT_DIR:-$ROOT_DIR/models_and_samples/trained_models}"
export STATS_LOAD_DIR="${STATS_LOAD_DIR:-$ROOT_DIR/data_analysis_pipeline/saved/statistics_run/stats}"
export EVAL_DIR="${EVAL_DIR:-$ROOT_DIR/evaluate_sbgm/results}"
export LOG_DIR="${LOG_DIR:-$ROOT_DIR/sbgm/logs}"
export CONFIG_DIR="${CONFIG_DIR:-$ROOT_DIR/sbgm/config}"

# --- PYTHONPATH so `python -m sbgm...` works like on LUMI ---
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Threading caps
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_WORKERS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_WORKERS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_WORKERS}"

mkdir -p "$LOG_DIR" || true

echo "[LOCAL DATA TEST]"
echo "  CFG            = ${CFG_PATH_ABS}"
echo "  SPLIT          = ${SPLIT}"
echo "  N              = ${N}"
echo "  START          = ${START}"
echo "  SHOW_ORIGINAL  = ${SHOW_ORIGINAL}"
echo "  NO_CUTOUTS     = ${NO_CUTOUTS}"
echo "  ROOT           = ${ROOT_DIR}"
echo "  DATA           = ${DATA_DIR}"
echo "  STATS_LOAD_DIR = ${STATS_LOAD_DIR}"
echo "  SLURM_CPUS_PER_TASK (num workers) = ${SLURM_CPUS_PER_TASK}"
echo "  EXP_DATE       = ${EXP_DATE}"
echo ""

# Build optional flags
EXTRA_FLAGS=()
if [[ "${SHOW_ORIGINAL}" == "true" ]]; then
  EXTRA_FLAGS+=("--show_original")
fi
if [[ "${NO_CUTOUTS}" == "true" ]]; then
  EXTRA_FLAGS+=("--no_cutouts")
fi

# Run dataset tester
python -m sbgm.data.dataset_tester \
  --cfg_path "${CFG_PATH_ABS}" \
  --split "${SPLIT}" \
  --n "${N}" \
  --start "${START}" \
  "${EXTRA_FLAGS[@]}"