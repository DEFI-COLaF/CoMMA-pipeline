#!/bin/bash
# Number of CPUs to request (can be overridden by env var CPUS)
CPUS=${CPUS:-96}

# RAM in GB — original logic kept (CPUS + 10)
RAM=$((CPUS + 10))

# Optional flags (can be set as env vars REVERSE and RESUBMIT)
REVERSE=${REVERSE:-0}
RESUBMIT=${RESUBMIT:-1}


echo "Submitting job with CPUS=$CPUS, RAM=${RAM}G, REVERSE=$REVERSE, RESUBMIT=$RESUBMIT"

sbatch --export=CPUS="${CPUS}",RAM="${RAM}",REVERSE="${REVERSE}",RESUBMIT="${RESUBMIT}" \
  --cpus-per-task="${CPUS}" --mem="${RAM}G" slurm/slurm-processing.sbatch
