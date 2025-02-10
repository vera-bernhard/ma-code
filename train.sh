#!/bin/bash

#SBATCH --time=0-01:00:00   ## days-hours:minutes:seconds
#SBATCH --mem=32GB          
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=4   ## Use greater than 1 for parallelized jobs
#SBATCH --job-name=PreProcessData ## job name
#SBATCH --output=log/preprocess.out  ## standard out file
#SBATCH --account=slt.cl.uzh
#SBATCH --partition=standard

python finetune_whisper/finetune.py
