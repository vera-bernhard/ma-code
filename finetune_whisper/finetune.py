import os
import random
import torch
import pandas as pd
import wandb
from datasets import DatasetDict, Audio, Dataset, IterableDataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
    WhisperTokenizer,
    WhisperFeatureExtractor
)
from jiwer import wer
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Generator
from tqdm import tqdm
import csv
import logging
import sys

# Set seed for reproducibility
random.seed(42)

# based on https://huggingface.co/learn/audio-course/chapter5/fine-tuning


# add logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/preprocess.out"),
    ],
    force=True
)

# logger for console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger()
logger.addHandler(console_handler)


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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for key in batch:
            batch[key] = batch[key].to(device)

        return batch


def make_split(data_dir: str, output_dir: str, train_ratio: float = 0.8, dataset_name: str = "srf_ad", subset: float = 1):
    """Given a directory with wav files and corresponding transcript files, creates a train, validation, and test split
    i.e. it writes three text files with the paths to the audio and transcript files for each split.

    """
    logger.info(f"Making the split in {output_dir}")
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


def load_custom_dataset(train_file: str, val_file: str, test_file: str):
    """Loads a custom dataset from a given train, validation, and test file."""
    logger.info(f"Loading splits")
    train_data = load_data_from_file(train_file)
    val_data = load_data_from_file(val_file)
    test_data = load_data_from_file(test_file)

    dataset = DatasetDict({
        "train": train_data,
        "validation": val_data,
        "test": test_data
    })

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def load_data_from_file(file_path: str):
    """Loads dataset from a given text file with wav-transcript pairs."""
    logging.info(f"Loading from {file_path}")
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


def preprocess(dataset: DatasetDict, whisper_size: str = 'small', outdir: str = "./data_prepared/srf_ad_feat") -> str:
    logger.info('Starting data preprocessing...')
    tokenizer = WhisperTokenizer.from_pretrained(
        f"openai/whisper-{whisper_size}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        f"openai/whisper-{whisper_size}")

    os.makedirs(outdir, exist_ok=True)

    def preprocess_and_save(batch: dict, idx: int, split_name: str) -> str:
        """Processes a single batch and saves to disk immediately."""
        file_path = os.path.join(split_outdir, f"data_{idx}.pt")

        if os.path.exists(file_path):
            logger.info(f"File {file_path} already exists. Skipping...")
            return file_path  # Only return the file path

        audio_array = batch['audio']['array']
        if audio_array is None or audio_array.size == 0:
            logger.warning(f"Faulty audio: {batch['audio']['path']}")
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
        logger.info(f"Processing split: {split_name}")

        split_outdir = os.path.join(outdir, split_name)
        os.makedirs(split_outdir, exist_ok=True)

        for idx, example in enumerate(tqdm(split_dataset, desc=f"Preprocessing {split_name}")):
            try:
                preprocess_and_save(example, idx, split_outdir)
            except Exception as e:
                logger.error(
                    f"Failed to process {split_name} index {idx}: {e}")

    logger.info(f"Preprocessed dataset saved to {outdir}")
    return outdir


def load_data_generator(split_dir: str, split: str, subset: float = 1.0) -> Generator:
    file_paths = sorted(os.listdir(os.path.join(split_dir, split)))
    if subset < 1:
        file_paths = file_paths[:int(subset * len(file_paths))]
    for file_name in file_paths:
        file_path = os.path.join(split_dir, split, file_name)
        yield torch.load(file_path, weights_only=True)


def load_data_split(split_dir: str, split: str, subset: float = 1.0) -> tuple[IterableDataset, int]:
    files = os.listdir(os.path.join(split_dir, split))
    if subset < 1:
        files = files[:int(subset * len(files))]
    return IterableDataset.from_generator(
        lambda: load_data_generator(split_dir, split, subset)), len(files)


def load_data_splits(feat_dir: str, subset: float = 1.0):
    return DatasetDict({
        split_name: IterableDataset.from_generator(
            lambda split_dir=split_name: load_data_generator(feat_dir, split_dir, subset))
        for split_name in os.listdir(feat_dir)
    })


def fine_tune(feat_dir: str, whisper_size: str = 'small', save_path: str = "./finetune_whisper", batch_size: int = 4):
    dataset = load_data_splits(feat_dir)

    # TODO: some redundancy as listdir is called in load_data_splits already
    num_train_samples = len(os.listdir(os.path.join(feat_dir, "train")))
    max_steps = num_train_samples // batch_size

    # dataset = DatasetDict.load_from_disk(feat_dir)
    model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/whisper-{whisper_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    processor = WhisperProcessor.from_pretrained(
        f"openai/whisper-{whisper_size}")

    wandb.init(project="finetune-whisper")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = TrainingArguments(
        output_dir="./finetune_whisper",
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
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        push_to_hub=False,
        report_to="wandb",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    # check if save_path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save_pretrained(save_path)
    predict(trainer, dataset["test"], "predictions.csv")


def predict(trainer: Trainer, dataset: Union[Dataset, IterableDataset], outfile: str, model_path: str = None, whisper_size: str = 'small', data_size: int = None):

    # Note: iterable datasets require the data_size to be provided, as they do not have a length attribute
    if type(dataset) == IterableDataset:
        if data_size is None:
            raise ValueError(
                "If dataset is of type IterableDataset, you must provide the data_size")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Case 1: Fine-tuned model
    if model_path:
        print(f"Loading model from {model_path}...")
        model = WhisperForConditionalGeneration.from_pretrained(model_path)

    # Case 2: Trainer object
    elif trainer:
        model = trainer.model

    # Case 3: Pre-trained model only
    else:
        print(
            f"Loading pre-trained Whisper model: openai/whisper-{whisper_size}...")
        model = WhisperForConditionalGeneration.from_pretrained(
            f"openai/whisper-{whisper_size}")

    model.to(device)

    processor = WhisperProcessor.from_pretrained(
        f"openai/whisper-{whisper_size}")
    model.eval()
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

    print(f"Predictions saved to {outfile}")


def compute_metrics(pred, model_size: str = 'small'):
    processor = WhisperProcessor.from_pretrained(
        f"openai/whisper-{model_size}")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_score = wer(label_texts, pred_texts)

    return {"wer": wer_score}


if __name__ == "__main__":
    data_path = "/home/vebern/scratch/srf_ad"
    output_path = "/home/vebern/data/ma-code/finetune_whisper/finetune_whisper_small"
    output_feat_path = "/home/vebern/scratch/srf_ad_feat"

    model_size = 'small'
    sizes = ['small', 'medium', 'large']

    # train_file, val_file, test_file = make_split(
    #     data_path, output_path)

    # dataset = load_custom_dataset(train_file, val_file, test_file)
    # output_feat_path = preprocess(dataset, model_size, output_feat_path)

    # load preprocessed datasez

    # # let not fine-tuned model predict on test set
    # test_set, nr_test_set = load_data_split(output_feat_path, "test", 0.01)
    # predict(None, test_set, f"predictions_whisper_{model_size}_untrained.csv",
    # model_path = None, whisper_size = model_size, data_size = nr_test_set)

    fine_tune(output_feat_path, model_size,
              save_path=f"./fine_tuned_whisper_{model_size}")
