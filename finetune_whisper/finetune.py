import os
import random
import torch
import pandas as pd
import wandb
from datasets import DatasetDict, Audio, Dataset
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
from typing import Any, Dict, List, Union
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
        logging.StreamHandler(sys.stdout)
    ],
    force=True)
logger = logging.getLogger()

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
        data.append({"audio": {"path": audio_file}, "text": text})

    data = Dataset.from_list(data)
    return data


def preprocess(dataset: DatasetDict, whisper_size: str = 'small', outdir: str = "./data_prepared/srf_ad_feat"):
    logger.info('Starting data preprocessing...')
    tokenizer = WhisperTokenizer.from_pretrained(
        f"openai/whisper-{whisper_size}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        f"openai/whisper-{whisper_size}")

    os.makedirs(outdir, exist_ok=True)

    def preprocess_function(batch):
        audio_array = batch["audio"]["array"]
        if audio_array is None or len(audio_array) == 0:
            logger.warning(f"Faulty audio: {batch['audio']['path']}")
            return None

        audio_input = feature_extractor(
            audio_array,
            sampling_rate=batch["audio"]["sampling_rate"],
            return_tensors="pt"
        )

        # Manually pad the input features to 3000 (cause just setting padding="max_length" doesn't work)
        input_features = audio_input["input_features"]
        if input_features.shape[-1] < 3000:
            pad_width = 3000 - input_features.shape[-1]
            input_features = torch.nn.functional.pad(input_features, (0, pad_width), mode='constant', value=0)
        else:
            input_features = input_features[:, :3000]
        
        if input_features.shape[-1] != 3000:
            logger.warning(f"Warning: Expected 3000 mel features, got {input_features.shape[-1]}")
        
        logger.info(f"Processed {batch['audio']['path']} successfully.")
        token_data = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=448, return_tensors="pt")
        return {
            "input_features": input_features[0],  # Take the first element for batch processing
            "labels": token_data.input_ids[0],
            "attention_mask": token_data.attention_mask[0],
        }

    dataset = dataset.map(preprocess_function, remove_columns=[
                          "audio", "text"], num_proc=4)
    logger.info(f'Saving data set to {outdir}')
    dataset.save_to_disk(outdir)
    return dataset


def fine_tune(feat_dir: str, whisper_size: str = 'small', save_path: str = "./finetune_whisper"):
    dataset = DatasetDict.load_from_disk(feat_dir)
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
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_total_limit=2,
        save_steps=500,
        logging_dir="./logs",
        learning_rate=1e-5,
        warmup_steps=500,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        push_to_hub=False,
        report_to="wandb"
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


def predict(trainer: Trainer, dataset: Dataset, outfile: str, model_path: str = None, whisper_size: str = 'small'):
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
        print(f"Loading pre-trained Whisper model: openai/whisper-{whisper_size}...")
        model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{whisper_size}")

    model.to(device)

    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{whisper_size}")
    model.eval()

    with open(outfile, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if f.tell() == 0:
            writer.writerow(["audio_file", "true_text", "pred_text"])

        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="Predicting"):
                batch = dataset[i]
                input_features = torch.tensor(batch["input_features"]).unsqueeze(0).to(device)
                attention_mask = batch.get("attention_mask", None)

                generated_ids = model.generate(input_features, attention_mask=attention_mask)
                pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                true_text = processor.batch_decode(torch.tensor(batch["labels"]).unsqueeze(0), skip_special_tokens=True)[0]
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


    train_file, val_file, test_file = make_split(
        data_path, output_path)
    model_size = 'small'
    sizes = ['small', 'medium', 'large']

    dataset = load_custom_dataset(train_file, val_file, test_file)
    dataset = preprocess(dataset, model_size, output_feat_path)

    # load preprocessed dataset
    # dataset = DatasetDict.load_from_disk(output_feat_path)

    # let not fine-tuned model predict on test set
    # predict(None, dataset["test"], f"predictions_whisper_{model_size}_untrained.csv",
    #         model_path=None, whisper_size=model_size)

    # fine_tune(output_feat_path, model_size, save_path=f"./fine_tuned_whisper_{model_size}_{subset}")
