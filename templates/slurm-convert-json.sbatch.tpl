#!/bin/bash
#SBATCH --job-name=comma-json-conversion
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --output=logs/convert_and_count_%j.out
#SBATCH --error=logs/convert_and_count_%j.err${extra_sbatch}

# Activate virtual environment
source env/bin/activate

# Run your scripts

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt"

CONVERT_WORKERS=${workers} python worker_convert_json.py
