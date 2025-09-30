#!/bin/bash
#SBATCH --job-name=comma-tei-txt-convert
#SBATCH --cpus-per-task=15
#SBATCH --mem=5G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/convert_and_count_%j.out
#SBATCH --error=logs/convert_and_count_%j.err

# Activate virtual environment
source env/bin/activate

# Run your scripts

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

python worker-convert-tei-txt.py
python stats-count.py

