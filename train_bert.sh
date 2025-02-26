#!/bin/bash

#SBATCH --time=0-00:30:00   ## days-hours:minutes:seconds
#SBATCH --mem=32GB         
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4   ## Use greater than 1 for parallelized jobs
#SBATCH --job-name=FinetuneSwissBert ## job name
#SBATCH --output=log/train_bert.log  ## standard out file
#SBATCH --account=slt.cl.uzh
#SBATCH --partition=standard

module load mamba
source activate ma_env2
/data/vebern/conda/envs/ma_env2/bin/python finetune_bert/finetune.py 
