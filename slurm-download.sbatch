#!/bin/bash
#SBATCH --job-name=download-worker
#SBATCH --output=logs/download_%A_%a.out
#SBATCH --array=1-14  # Change this to use more workers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-00:00:00

# Load necessary modules (if any)
# module load python/3.x

# Activate your virtual environment if needed
source env/bin/activate

# Run the script with the current task ID
python worker-single-download.py --index ${SLURM_ARRAY_TASK_ID} --max 14
