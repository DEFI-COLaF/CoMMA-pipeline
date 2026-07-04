#!/bin/bash
#SBATCH --job-name=comma-processing
#SBATCH --time=${time}
#SBATCH --output=./log/process_log_%j.log${extra_sbatch}


BATCHING=$((CPUS - 4))

echo "[PARAMS] CPUS=$CPUS RAM=${RAM}G REVERSE=$REVERSE RESUBMIT=$RESUBMIT BATCHING=$BATCHING"
# This script runs the processing, aka. the segmentation, archiving and OCR

echo "### Running $SLURM_JOB_NAME ###"

source $HOME/.bashrc
source $HOME/.bash_profile

start=`date +%s`


source env/bin/activate
echo $start
echo "Running worker"

KRAKEN_BATCH_SIZE="${BATCHING}" REVERSE="${REVERSE}" python worker_process.py

conda deactivate

end=`date +%s`

echo $end
echo "Resubmitting"

if [[ "$RESUBMIT" == "1" ]]; then
    echo "RESUBMIT mode active"
    sbatch "$0"
fi
