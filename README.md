CoMMA Downloading Pipeline
==========================

## Installation

In order to use this pipeline, you need to run `pip install requirements.txt` or create an environment at ./env

## Running

Run this pipeline by launching two parallel tasks, one running the `slurm-processing.sbatch` (Check the model used) and one the `worker-download.sbatch` (Needs a list of IIIF manifests).
 You can change the file it uses to run download, it basically needs a CSV with a `manifest_url` column.

You will need to change the models configuration

## Cite

CoMMA Paper
