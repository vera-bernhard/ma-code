import os
import random
import csv
import argparse
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Generator

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
    WhisperTokenizer,
    WhisperFeatureExtractor
)
from jiwer import wer, cer
from tqdm import tqdm
from datasets import DatasetDict, Audio, Dataset, IterableDataset
import pandas as pd
import torch
import wandb

# Set seed for reproducibility
random.seed(42)

# based on https://huggingface.co/learn/audio-course/chapter5/fine-tuning


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        # Add attention mask to avoid warning
        attention_mask = batch["attention_mask"] if "attention_mask" in batch else None
        if attention_mask is not None:
            batch["attention_mask"] = attention_mask

        return batch


def make_split(data_dir: str, output_dir: str, train_ratio: float = 0.8, dataset_name: str = "srf_ad", subset: float = 1) -> tuple[str, str, str]:
    """Given a directory with wav files and corresponding transcript files, creates a train, validation, and test split
    i.e. it writes three text files with the paths to the audio and transcript files for each split.

    """
    files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
    files.sort()
    random.shuffle(files)
    if subset < 1:
        files = files[:int(subset * len(files))]

    val_test_ratio = (1 - train_ratio) / 2
    total = len(files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_test_ratio)

    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    def save_split(files: list[str], filename: str):
        outfile = os.path.join(output_dir, filename)
        with open(outfile, "w", encoding="utf-8") as f:
            for file in files:
                base_name = os.path.basename(file).split(".")[0]
                f.write(
                    f"{data_dir}/{base_name}.wav\t{data_dir}/{base_name}.txt\n")
        return outfile

    return (
        save_split(train_files, f"train_{dataset_name}.txt"),
        save_split(val_files, f"val_{dataset_name}.txt"),
        save_split(test_files, f"test_{dataset_name}.txt"),
    )


def load_custom_dataset(train_file: str, val_file: str, test_file: str, logger: logging.Logger = None) -> DatasetDict:
    """Loads a custom dataset from a given train, validation, and test file."""
    train_data = load_data_from_file(train_file, logger)
    val_data = load_data_from_file(val_file, logger)
    test_data = load_data_from_file(test_file, logger)

    dataset = DatasetDict({
        "train": train_data,
        "validation": val_data,
        "test": test_data
    })

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def load_data_from_file(file_path: str, logger: logging.Logger = None) -> Dataset:
    """Loads dataset from a given text file with wav-transcript pairs."""
    logger.info(f"Loading data from {file_path}") if logger else print(
        f"Loading from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        audio_file, text_file = line.strip().split("\t")
        text = open(text_file, "r", encoding="utf-8").read()
        data.append({"audio": {"path": audio_file},
                    "text": text, "audio_path": audio_file})

    data = Dataset.from_list(data)
    return data


def preprocess(dataset: DatasetDict, whisper_size: str = 'small', outdir: str = "./data_prepared/srf_ad_feat", logger: logging.Logger = None) -> str:
    logger.info('Starting data preprocessing...') if logger else print(
        'Starting data preprocessing...')
    tokenizer = WhisperTokenizer.from_pretrained(
        f"openai/whisper-{whisper_size}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        f"openai/whisper-{whisper_size}")

    os.makedirs(outdir, exist_ok=True)

    def preprocess_and_save(batch: dict, idx: int, split_name: str) -> str:
        """Processes a single batch and saves to disk immediately."""
        file_path = os.path.join(split_outdir, f"data_{idx}.pt")

        if os.path.exists(file_path):
            logger.info(f"File {file_path} already exists. Skipping...") if logger else print(
                f"File {file_path} already exists. Skipping...")
            return file_path  # Only return the file path

        audio_array = batch['audio']['array']
        if audio_array is None or audio_array.size == 0:
            logger.warning(f"Faulty audio: {batch['audio']['path']}") if logger else print(
                f"Faulty audio: {batch['audio']['path']}")
            return None

        # Process audio
        audio_input = feature_extractor(
            audio_array, sampling_rate=batch['audio']['sampling_rate'], return_tensors="pt")
        input_features = audio_input["input_features"]
        input_features = torch.nn.functional.pad(input_features, (0, max(
            0, 3000 - input_features.shape[-1])), mode='constant', value=0)
        input_features = input_features[:, :3000]

        # Process text
        token_data = tokenizer(batch["text"], padding="max_length",
                               truncation=True, max_length=448, return_tensors="pt")

        processed_data = {
            "input_features": input_features[0],
            "labels": token_data.input_ids[0],
            "attention_mask": token_data.attention_mask[0],
            "audio_path": batch["audio_path"]
        }

        torch.save(processed_data, file_path)
        return file_path

    for split_name, split_dataset in dataset.items():
        logger.info(f"Processing split: {split_name}") if logger else print(
            f"Processing split: {split_name}")

        split_outdir = os.path.join(outdir, split_name)
        os.makedirs(split_outdir, exist_ok=True)

        for idx, example in enumerate(tqdm(split_dataset, desc=f"Preprocessing {split_name}")):
            try:
                preprocess_and_save(example, idx, split_outdir)
            except Exception as e:
                logger.error(
                    f"Failed to process {split_name} index {idx}: {e}") if logger else print(
                    f"Failed to process {split_name} index {idx}: {e}")

    logger.info(f"Preprocessed dataset saved to {outdir}") if logger else print(
        f"Preprocessed dataset saved to {outdir}")
    return outdir


def load_data_generator(split_dir: str, split: str = None, subset: float = 1.0) -> Generator:
    if split is not None:
        file_paths = sorted(os.listdir(os.path.join(split_dir, split)))
    else:
        file_paths = sorted(os.listdir(split_dir))
    if subset < 1:
        file_paths = file_paths[:int(subset * len(file_paths))]
    for file_name in file_paths:
        if split is not None:
            file_path = os.path.join(split_dir, split, file_name)
        else:
            file_path = os.path.join(split_dir, file_name)
        yield torch.load(file_path, weights_only=True)


def load_data_split(split_dir: str, subset: float = 1.0) -> tuple[IterableDataset, int]:
    return IterableDataset.from_generator(lambda: load_data_generator(split_dir, subset=subset)), len(os.listdir(split_dir))


def load_data_splits(feat_dir: str, subset: float = 1.0):
    return DatasetDict({
        split_name: IterableDataset.from_generator(
            lambda split_dir=split_name: load_data_generator(feat_dir, split_dir, subset))
        for split_name in os.listdir(feat_dir)
    })


def fine_tune(feat_dir: str, whisper_size: str = 'small', save_path: str = "./finetune_whisper", batch_size: int = 8, epochs: int = 3, logger: logging.Logger = None, subset: float = 1.0):
    logger.info("Loading data splits...") if logger else print(
        "Loading data splits...")
    dataset = load_data_splits(feat_dir, subset=subset)

    # TODO: some redundancy as listdir is called in load_data_splits already
    num_train_samples = len(os.listdir(os.path.join(feat_dir, "train")))
    if subset < 1:
        nr_subset = int(subset * num_train_samples)
        logger.info(
            f"Using {nr_subset} samples out of {num_train_samples} for training") if logger else print(
            f"Using {nr_subset} samples out of {num_train_samples} for training")
        num_train_samples = nr_subset
        if num_train_samples == 0:
            raise ValueError(
                "The subset ratio is too low. No training samples available")
    max_steps = max(num_train_samples // batch_size, 1) * epochs

    # dataset = DatasetDict.load_from_disk(feat_dir)
    model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{whisper_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    processor = WhisperProcessor.from_pretrained(
        f"openai/whisper-{whisper_size}")

    wandb.init(project="finetune-whisper")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    date = datetime.now().strftime("%Y%m%d_%H%M")

    # check if save_path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    training_args = TrainingArguments(
        output_dir=save_path,
        eval_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        logging_dir="./logs",
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=max_steps,
        weight_decay=0.01,
        fp16=True,
        push_to_hub=False,
        report_to="wandb",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        run_name=f'finetune-whisper-{whisper_size}_{date}'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper(processor),
    )

    logger.info("Starting training...") if logger else print(
        "Starting training...")
    trainer.train()
    # check if save_path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger.info(f"Saving model to {save_path}") if logger else print(
        f"Saving model to {save_path}")
    model.save_pretrained(save_path)
    logger.info("Prediction on test set...") if logger else print(
        "Prediction on test set...")
    pred_file = os.path.join(save_path, "predictions.csv")
    predict(trainer, dataset["test"], pred_file,
            whisper_size=whisper_size, logger=logger)


def compute_metrics_wrapper(processor: WhisperProcessor):
    def compute_metrics(pred):
        logits, _ = pred.predictions # pred.predictions is a tuple with the first element being the predicted ids, second element is the logits
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_ids = logits.argmax(axis=-1)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer_score = wer(label_str, pred_str)
        cer_score = cer(label_str, pred_str)
        return {"wer": 100*wer_score, "cer": 100*cer_score}

    return compute_metrics


def predict(trainer: Trainer, dataset: Union[Dataset, IterableDataset], outfile: str, model_path: str = None, whisper_size: str = 'small', data_size: int = None, logger: logging.Logger = None):

    # Note: iterable datasets require the data_size to be provided, as they do not have a length attribute
    if type(dataset) == IterableDataset:
        if data_size is None:
            raise ValueError(
                "If dataset is of type IterableDataset, you must provide the data_size")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Case 1: Fine-tuned model
    if model_path:
        logger.info(f"Loading fine-tuned model from {model_path}") if logger else print(
            f"Loading fine-tuned model from {model_path}")
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)

    # Case 2: Trainer object
    elif trainer:
        model = trainer.model
        processor = trainer.data_collator.processor

    # Case 3: Pre-trained model only
    else:
        logger.info(
            f"Loading pre-trained Whisper model: openai/whisper-{whisper_size}...") if logger else print(
            f"Loading pre-trained Whisper model: openai/whisper-{whisper_size}...")
        model = WhisperForConditionalGeneration.from_pretrained(
            f"openai/whisper-{whisper_size}")
        processor = WhisperProcessor.from_pretrained(
            f"openai/whisper-{whisper_size}")

    model.to(device)
    model.eval()
    if os.path.exists(outfile):
        logger.warning(f"File {outfile} already exists. Deleting...") if logger else print(
            f"File {outfile} already exists. Deleting...")
        os.remove(outfile)
    with open(outfile, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Wrtie header if file is empty
        if f.tell() == 0:
            writer.writerow(["audio_file", "true_text", "pred_text"])

        with torch.no_grad():
            dataset_size = data_size if data_size else len(dataset)
            for i, batch in tqdm(enumerate(dataset), total=dataset_size, desc="Predicting"):
                input_features = batch["input_features"].clone(
                ).detach().unsqueeze(0).to(device)

                attention_mask = batch.get("attention_mask", None)

                generated_ids = model.generate(
                    input_features, attention_mask=attention_mask)
                pred_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True)[0]
                true_text = processor.batch_decode(
                    batch["labels"].clone().detach().unsqueeze(0), skip_special_tokens=True)[0]
                audio_file = batch.get("audio_path", "Unknown")
                writer.writerow([audio_file, true_text, pred_text])
                f.flush()

    logger.info(f"Predictions saved to {outfile}") if logger else print(
        f"Predictions saved to {outfile}")


def evaluate(pred_file: str) -> dict:
    df = pd.read_csv(pred_file)
    true_texts = df["true_text"].to_list()
    pred_texts = df["pred_text"].to_list()
    wer_score = wer(true_texts, pred_texts)
    cer_score = cer(true_texts, pred_texts)
    return {"wer": 100*wer_score, "cer": 100*cer_score}


def setup_logger(log_file: str) -> logging.Logger:
    """Set up logging configuration and return the logger."""
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
        ],
        force=True
    )
    logger = logging.getLogger()
    return logger


def init_argparse():
    parser = argparse.ArgumentParser(
        description="Fine-tune the Whisper model on the SRF AD dataset")

    parser.add_argument("job", type=str, choices=[
                        "train", "predict", "preprocess", "split", "evaluate"], help="The job to perform")
    # add log file
    parser.add_argument("--log_file", type=str,
                        help="Path to log file", required=True)

    parser.add_argument("--raw_data", type=str,
                        help="Path to directory with audio and transcript files", default='')
    parser.add_argument("--split_dir", type=str,
                        help="Path to directory with split files", default='')
    parser.add_argument("--split_ratio", type=float,
                        default=0.8, help="Ratio for train split")
    parser.add_argument("--subset_ratio", type=float,
                        default=1.0, help="Ratio of data to use")

    parser.add_argument("--whisper_size", type=str, default="small",
                        choices=["small", "medium", "large"], help="Size of the Whisper model")
    parser.add_argument("--feat_dir", type=str,
                        help="Path to directory with preprocessed data", default='')

    parser.add_argument("--model_save_dir", type=str,
                        help="Path to save the fine-tuned model", default='')
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs for training")

    parser.add_argument("--model_path", type=str,
                        help="Path to model", default='')
    parser.add_argument("--predict_file", type=str,
                        help="Path to save predictions", default='')

    parser.add_argument("--eval_file", type=str,
                        help="Path to predictions file")

    return parser


def check_raw_data(raw_data_dir: str, logger: logging.Logger = None):
    files = os.listdir(raw_data_dir)
    for file in files:
        if file.endswith(".wav"):
            if not any([file.replace(".wav", ".txt") in files]):
                logger.warning(f"Missing transcript file for {file}") if logger else print(
                    f"Missing transcript file for {file}")
        elif file.endswith(".txt"):
            if not any([file.replace(".txt", ".wav") in files]):
                logger.warning(f"Missing audio file for {file}") if logger else print(
                    f"Missing audio file for {file}")
        else:
            raise ValueError(f"Unexpected file in directory: {file}")
    nr_files = len(files) // 2
    logger.info(
        f"Found {nr_files} pairs of audio and transcript files in {raw_data_dir}") if logger else print(
        f"Found {nr_files} pairs of audio and transcript files in {raw_data_dir}")


def parse_args():
    parser = init_argparse()
    args = parser.parse_args()

    # init log file
    logger = setup_logger(args.log_file)

    if args.job == "split":
        # Required: raw_data, split_dir

        if not args.raw_data:
            raise ValueError("You must provide --raw_data")
        if not os.path.exists(args.raw_data):
            raise FileExistsError(
                f"The raw data directory '{args.raw_data}' does not exist")
        else:
            check_raw_data(args.raw_data, logger)
        if not args.split_dir:
            raise ValueError(
                "You must provide --split_dir to save the split files")
        os.makedirs(args.split_dir, exist_ok=True)
        logger.info(
            f"Splitting data in '{args.raw_data}' and saving to '{args.split_dir}'")

    elif args.job == 'preprocess':
        # Required: raw_data, split_dir, feat_dir
        if not args.raw_data:
            raise ValueError("You must provide --raw_data")
        else:
            check_raw_data(args.raw_data, logger)

        if not os.path.exists(args.split_dir):
            raise FileExistsError("The split directory does not exist")
        else:
            files = os.listdir(args.split_dir)
            if not any([f.startswith("test") for f in files]) or not any([f.startswith("train") for f in files]) or not any([f.startswith("val") for f in files]):
                raise FileExistsError(
                    "The split directory does not contain the necessary files")

        if not args.feat_dir:
            raise ValueError(
                "You must provide the path --feat_dir to save the preprocessed data")
        os.makedirs(args.feat_dir, exist_ok=True)
        logger.info(
            f"Preprocessing data from '{args.raw_data}' and saving to '{args.feat_dir}'")

    elif args.job == 'train':
        # required feat_dir, model_save_dir
        if not args.feat_dir:
            raise ValueError(
                "You must provide --feat_dir to load the preprocessed data")
        else:
            # check if there is test, train, validation directories
            if os.path.exists(os.path.join(args.feat_dir, "test")) and os.path.exists(os.path.join(args.feat_dir, "train")) and os.path.exists(os.path.join(args.feat_dir, "validation")):
                pass
            else:
                raise ValueError(
                    f"The preprocessed data directory '{args.feat_dir}' does not contain the necessary files")
        logger.info(
            f"Fine-tuning whisper {args.whisper_size} on {args.feat_dir} and saving to {args.model_save_dir}")

    elif args.job == 'predict':
        # required feat_dir, model_path, predict_file
        if not args.feat_dir:
            raise ValueError(
                "You must provide --feat_dir to load the preprocessed data")
        else:
            if not os.path.exists(args.feat_dir):
                raise FileExistsError(
                    f"The preprocessed data directory '{args.feat_dir}' does not exist")
        if not args.model_path:
            logger.info(
                f'No model found, assuming pretrained whisper-{args.whisper_size}')
        else:
            if not os.path.exists(args.model_path):
                raise FileExistsError(
                    f"The model path '{args.model_path}' does not exist")

        if not args.predict_file:
            raise ValueError(
                "You must provide --predict_file to save the predictions")
        logger.info(
            f"Predicting with model from {args.model_path} on {args.feat_dir} and saving to {args.predict_file}")

    elif args.job == 'evaluate':
        if not args.eval_file:
            raise ValueError(
                "You must provide --eval_file to evaluate the predictions")
        if not os.path.exists(args.eval_file):
            raise ValueError(
                f"The predictions file '{args.eval_file}' does not exist")
        else:
            df = pd.read_csv(args.eval_file, nrows=1)
            if not all([col in df.columns for col in ["audio_file", "true_text", "pred_text"]]):
                raise ValueError(
                    "The predictions file does not contain the necessary columns (audio_file, true_text, pred_text)")
        logger.info(f"Evaluating predictions in {args.eval_file}")
    return args, logger


if __name__ == "__main__":
    args, logger = parse_args()

    if args.job == 'split':
        train_file, val_file, test_file = make_split(
            args.raw_data, args.split_dir, train_ratio=args.split_ratio, subset=args.subset_ratio)
        logger.info(f"Split files saved to {args.split_dir}")

    elif args.job == 'preprocess':
        for file in os.listdir(args.split_dir):
            if file.startswith("train"):
                train_file = os.path.join(args.split_dir, file)
            elif file.startswith("val"):
                val_file = os.path.join(args.split_dir, file)
            elif file.startswith("test"):
                test_file = os.path.join(args.split_dir, file)
        dataset = load_custom_dataset(
            train_file, val_file, test_file, logger=logger)
        preprocess(dataset, args.whisper_size, args.feat_dir, logger=logger)

    elif args.job == 'train':
        fine_tune(args.feat_dir, args.whisper_size, args.model_save_dir,
                  batch_size=args.batch_size, epochs=args.epochs, subset=args.subset_ratio, logger=logger)

    elif args.job == 'predict':
        dataset, size = load_data_split(args.feat_dir, args.subset_ratio)
        # Case 1: Fine-tuned model
        if args.model_path:
            predict(trainer=None, dataset=dataset, outfile=args.predict_file,
                    model_path=args.model_path, whisper_size=args.whisper_size, data_size=size, logger=logger)
        # Case 2: Pre-trained model
        else:
            predict(trainer=None, dataset=dataset, outfile=args.predict_file,
                    model_path=None, whisper_size=args.whisper_size, data_size=size, logger=logger)

    elif args.job == 'evaluate':
        wer_score = evaluate(args.eval_file)
        logger.info(f"Word Error Rate: {wer_score}")
