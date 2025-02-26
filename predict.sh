#!/bin/bash

#SBATCH --time=0-00:30:00   ## days-hours:minutes:seconds
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4   ## Use greater than 1 for parallelized jobs
#SBATCH --job-name=FinetuneSmall ## job name
#SBATCH --output=log/predict.log  ## standard out file
#SBATCH --account=slt.cl.uzh
#SBATCH --partition=standard

module load mamba
source activate ma_env2
python finetune_whisper/finetune.py predict --feat_dir /scratch/vebern/srf_ad_feat/test --log log/predict.log --predict_file finetune_whisper/test_predictions_whisper_small_finetuned.csv --model_path model/whisper_small_finetuned_20250224/checkpoint-4250
