#!/bin/bash
#SBATCH --array=0-4
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

module purge
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0
source venv/bin/activate

python -u NN_test.py $SLURM_ARRAY_TASK_ID