#!/bin/bash

#SBATCH --time=0-00:05:00   ## days-hours:minutes:seconds
#SBATCH --mem=4GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4   ## Use greater than 1 for parallelized jobs
#SBATCH --job-name=EvaleSmall ## job name
#SBATCH --output=log/evaluate.log  ## standard out file
#SBATCH --account=slt.cl.uzh
#SBATCH --partition=standard

module load mamba
source activate ma_env2
/data/vebern/conda/envs/ma_env2/bin/python finetune_whisper/finetune.py evaluate --eval_file finetune_whisper/test_predictions_whisper_small_untrained.csv --log_file log/evaluate.log