#!/bin/bash

#SBATCH --time=0-01:00:00   ## days-hours:minutes:seconds
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4   ## Use greater than 1 for parallelized jobs
#SBATCH --job-name=PredictLarge ## job name
#SBATCH --output=log/predict_large.log  ## standard out file
#SBATCH --account=slt.cl.uzh
#SBATCH --partition=standard

module load mamba
source activate ma_env2
python finetune_whisper/finetune.py predict --feat_dir /scratch/vebern/srf_ad_feat/test --log log/predict.log --predict_file finetune_whisper/test_predictions_whisper_large_untrained.csv --whisper_size large
