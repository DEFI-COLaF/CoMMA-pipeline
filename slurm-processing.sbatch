#!/bin/bash
#
#
#SBATCH --job-name=comma-processing                        # Job name
#SBATCH --cpus-per-task=90                            # cpu-cores per task (>1 if multi-threaded tasks --cpus-per-task=80 --mem=80G --time=7-00:00:00)
#SBATCH --time=7-00:00:00                             # Time limit hrs:min:sec
#SBATCH --output=./log/process_log_%j.log                     # Standard output and error log
#SBATCH --mem=90G


# This script runs the processing, aka. the segmentation, archiving and OCR

echo "### Running $SLURM_JOB_NAME ###"

source $HOME/.bashrc
source $HOME/.bash_profile

start=`date +%s`


source env/bin/activate
echo $start 
echo "Running worker"
KRAKEN_BATCH_SIZE=80 python worker-process.py

conda deactivate

end=`date +%s`
