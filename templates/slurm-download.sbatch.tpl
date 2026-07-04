#!/bin/bash
#SBATCH --job-name=comma-download
#SBATCH --output=logs/download_%A_%a.out
#SBATCH --array=1-${workers}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time}${extra_sbatch}

# Load necessary modules (if any)
# module load python/3.x

# Activate your virtual environment if needed
source env/bin/activate

# Run the script with the current task ID
python worker_single_download.py --index ${SLURM_ARRAY_TASK_ID} --max ${workers} --files "not_contains_bsb.csv=;" "biblissima_arca_gallica_addenda_20251021.csv=$" "oxford.csv=$"
