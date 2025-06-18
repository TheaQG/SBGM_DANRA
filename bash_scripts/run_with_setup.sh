# scripts/run_data_filter.sh

#!/bin/bash
#SBATCH --job-name=data_filter
# [Other Slurm options]

module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_465001695/containers/images/my_torch_container_with_plotting.sif

source scripts/env_setup.sh

srun singularity exec $CONTAINER \
    python sbgm/data_scripts/data_filter.py