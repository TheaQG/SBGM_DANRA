#!/bin/bash

#SBATCH -A project_xxx
#SBATCH -J era5_proc
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

module load cdo netcdf-python
srun python -m era5_pipeline.cli.run_lumi