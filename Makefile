show-jobs:
	squeue -u $(USER)

sort-jobs:
	squeue -S t,Q | less

stop-all-jobs:
	scancel -u $(USER)

max-ram:
	@echo -n "Enter the job ID or job name: " && read jobid && bash -c 'if [[ $$jobid =~ ^[0-9]+$$ ]]; then sstat -a $$jobid -o Jobid,MaxRSS,AveCPU; else sstat -a --name=$$jobid -o Jobid,MaxRSS,AveCPU; fi'

stop-job:
	@echo -n "Enter the job ID or job name: " && read jobid && bash -c 'if [[ $$jobid =~ ^[0-9]+$$ ]]; then scancel $$jobid; else scancel --name=$$jobid; fi'

# 10 Minute GPU session
interactive-gpu:
	srun --pty -n 1 -c 2 --time=00:10:00 --gpus=1 --mem=8G bash -l

interactive-cpu:
	srun --pty -n 1 -c 2 --time=00:10:00 --mem=7G bash -l

sync:
	rsync -az /home/vera/Documents/Uni/Master/Master_Thesis/ma-code/data_prepared/srf_ad/ vebern@cluster.s3it.uzh.ch:/home/vebern/scratch/srf_ad

eval:
	python finetune_whisper/finetune.py evaluate --eval_file predictions_whisper_small_untrained.csv --log_file log/evaluate.log

split:
	python finetune_whisper/finetune.py split --raw_data data_prepared/srf_ad --split_dir data_prepared/split_test --log_file log/split.log --subset_ratio 0.01

preprocess:
	python finetune_whisper/finetune.py preprocess  --raw_data data_prepared/srf_ad --split_dir data_prepared/split_test --feat_dir data_prepared/feats --log_file log/preprocess.log

train:
	python finetune_whisper/finetune.py train --feat_dir data_prepared/feats --model_save_dir finetune_whisper/models --log_file log/train.log

predict:
	python finetune_whisper/finetune.py predict --feat_dir data_prepared/feats/test --log log/predict.log --predict_file test_pred.csv
