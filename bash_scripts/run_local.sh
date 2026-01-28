#!/usr/bin/env bash
set -euo pipefail # Exit on error, undefined variable, or error in a pipeline

# -----------------------
# Usage:
#   ./run_local.sh /path/to/config.yaml [mode] [make_plots]
#
# Examples:
#   ./run_local.sh sbgm/config/ablations/B2.yaml full_pipeline true
#   ./run_local.sh mnt/data/paper2_data_test_local.yaml quicklook true
# -----------------------

CFG_PATH="${1:-}"   # Path to config YAML file, required
MODE="${2:-full_pipeline}" # Mode: full_pipeline | generate | evaluate | quicklook | ...
MAKE_PLOTS="${3:-true}"  # Whether to make plots: true | false

if [[ -z "${CFG_PATH}" ]]; then
  echo "ERROR: Configuration file path is required."
  echo "Usage: $0 /path/to/config.yaml [mode] [make_plots]"
  exit 1
fi

# Resolve config path to absolute path (pure bash, robust)
CFG_PATH_ABS="$(cd "$(dirname "$CFG_PATH")" && pwd)/$(basename "$CFG_PATH")"

# --- Repo root discovery (script-location aware) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Threads/workers (configs use SLURM_CPUS_PER_TASK if available) ---
CPU_WORKERS="${CPU_WORKERS:-4}" # Default to 4 if not set
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-$CPU_WORKERS}"

# --- Optional GPU selection
# export CUDA_VISIBLE_DEVICES=0 # Uncomment and set to limit to specific GPU(s)

# --- Experiment date used by configs
export EXP_DATE="${EXP_DATE:-$(date +%Y%m%d)}" # Default to today's date

# --- Set default paths
export DATA_DIR="${DATA_DIR:-$ROOT_DIR/../../Data/data_DiffMod_small}"
export SAMPLE_DIR="${SAMPLE_DIR:-$ROOT_DIR/models_and_samples/generated_samples}"
export CKPT_DIR="${CKPT_DIR:-$ROOT_DIR/models_and_samples/trained_models}"
export STATS_LOAD_DIR="${STATS_LOAD_DIR:-$ROOT_DIR/data_analysis_pipeline/saved/statistics_run/stats}"
export EVAL_DIR="${EVAL_DIR:-$ROOT_DIR/evaluate_sbgm/results}"
export LOG_DIR="${LOG_DIR:-$ROOT_DIR/sbgm/logs}"
export CONFIG_DIR="${CONFIG_DIR:-$ROOT_DIR/sbgm/config}"

# --- PYTHONPATH so `python -m sbgm...` workks like on LUMI
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Threading caps (helps avoid oversubscription on laptop/desktop)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_WORKERS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_WORKERS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_WORKERS}"

mkdir -p "$LOG_DIR" "$SAMPLE_DIR" "$CKPT_DIR" "$EVAL_DIR" || true # Create dirs if not exist

echo "[LOCAL RUN]"
echo "  CFG     = ${CFG_PATH_ABS}"
echo "  MODE    = ${MODE}"
echo "  PLOTS   = ${MAKE_PLOTS}"
echo "  ROOT    = ${ROOT_DIR}"
echo "  DATA    = ${DATA_DIR}"
echo "  SLURM_CPUS_PER_TASK (num workers) = ${SLURM_CPUS_PER_TASK}"
echo "  EXP_DATE= ${EXP_DATE}"
echo ""

# Run
if [[ "${MAKE_PLOTS}" == "true" ]]; then
  python -m sbgm.cli.main_app --mode "${MODE}" --config_path "${CFG_PATH_ABS}" --make_plots
else
  python -m sbgm.cli.main_app --mode "${MODE}" --config_path "${CFG_PATH_ABS}"
fi