import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import Dataset, DatasetDict
import torch


class BertFineTuneDataset:
    def __init__(self, data_dir, tokenizer_name="bert-base-german-cased", test_size=0.1, max_length=512):
        self.data_dir = data_dir
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.test_size = test_size
        self.max_length = max_length

    def _load_sentences(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _concatenate_and_tokenize(self, sentences):
        tokenized_texts = []
        current_tokens = []

        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)  # Tokenize sentence
            # +2 for [CLS] & [SEP]
            if len(current_tokens) + len(tokens) + 2 <= self.max_length:
                current_tokens.extend(tokens)
            else:
                tokenized_texts.append(
                    self.tokenizer.convert_tokens_to_string(current_tokens))
                current_tokens = tokens  # Start a new sequence

        if current_tokens:
            tokenized_texts.append(
                self.tokenizer.convert_tokens_to_string(current_tokens))

        return tokenized_texts

    def prepare_dataset(self):
        """Processes each file separately, splits, and tokenizes."""
        all_texts = []

        for file in os.listdir(self.data_dir):  # Process each file separately
            sentences = self._load_sentences(file)
            concatenated_texts = self._concatenate_and_tokenize(sentences)
            all_texts.extend(concatenated_texts)  # Keep separate per file

        # Split into train & validation
        train_texts, val_texts = train_test_split(
            all_texts, test_size=self.test_size, random_state=42)

        # Create Hugging Face dataset
        dataset = DatasetDict({
            "train": Dataset.from_dict({"text": train_texts}),
            "validation": Dataset.from_dict({"text": val_texts})
        })

        # Tokenize dataset
        dataset = dataset.map(lambda x: self.tokenizer(
            x["text"], truncation=True, max_length=self.max_length, padding="max_length"), batched=True)

        return dataset


def finetune(data_dir: str, save_dir: str):
    model_name = "google-bert/bert-base-german-cased"
    model = BertForMaskedLM.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    dataset_builder = BertFineTuneDataset(data_dir)
    dataset = dataset_builder.prepare_dataset()

    date = datetime.now().strftime("%Y%m%d_%H%M")
    training_args = TrainingArguments(
        output_dir=save_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=500,
        save_total_limit=2,
        push_to_hub=False,
        report_to="wandb",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        run_name=f"finetune-swissbert_{date}"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def get_token_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state, inputs["attention_mask"]


def compute_bertscore(hypothesis: str, reference, model: str, tokenizer: BertTokenizer):
    hyp_embeddings, cand_mask = get_token_embeddings(
        hypothesis, model, tokenizer)
    ref_embeddings, ref_mask = get_token_embeddings(
        reference, model, tokenizer)
    cos = torch.nn.CosineSimilarity()
    similarities = cos(hyp_embeddings, ref_embeddings)

    valid_tokens = torch.min(cand_mask, ref_mask).float()
    # Aggregate: Mean similarity over valid (non-padding) tokens
    score = (similarities * valid_tokens.unsqueeze(-1)
             ).sum() / valid_tokens.sum()

    return score.item()
