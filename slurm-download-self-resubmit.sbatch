#!/bin/bash
#SBATCH --job-name=comma-download
#SBATCH --output=logs/single_download_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00

# Load necessary modules (if any)
# module load python/3.x

# Activate your virtual environment if needed
source env/bin/activate

# Run the script with the current task ID
python worker-single-download.py --index 1 --max 1 --files "contains_bsb.csv=;" --max_download 5000 --queue 1000000

# ---- Schedule rerun ----
echo "[$(date)] Job finished, resubmitting for tomorrow..."
sbatch --begin=now+24hours "$0"