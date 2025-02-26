#!/bin/bash

#SBATCH --time=0-06:00:00   ## days-hours:minutes:seconds
#SBATCH --mem=32GB
#SBATCH --gpus=1          
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4   ## Use greater than 1 for parallelized jobs
#SBATCH --job-name=FinetuneSmall ## job name
#SBATCH --output=log/train.log  ## standard out file
#SBATCH --account=slt.cl.uzh
#SBATCH --partition=standard

module load mamba
source activate ma_env2
/data/vebern/conda/envs/ma_env2/bin/python finetune_whisper/finetune.py train --feat_dir /scratch/vebern/srf_ad_feat --model_save_dir model/whisper_small_finetuned_20250224 --log_file log/train.log --batch_size 8 --epochs 1
