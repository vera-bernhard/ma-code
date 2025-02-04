
# https://medium.com/@prashanth.ramanathan/fine-tuning-a-pre-trained-gpt-2-model-and-performing-inference-a-hands-on-guide-57c097a3b810

from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenizer.pad_token = tokenizer.eos_token

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

# Set the EOS token as the padding token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    inputs =  tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='/mnt/disks/disk1/results',
    evaluation_strategy='epoch',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='/mnt/disks/disk1/logs'
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# Train the model
trainer.train(resume_from_checkpoint='/mnt/disks/disk1/results/checkpoint-5000')

# save the model and tokenizer explicitly
model_output_dir = '/mnt/disks/disk1/results/model'

model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def main(input_text):
    # Load the tokenizer and model from the saved directory
    model_path = '/mnt/disks/disk1/results/model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Calculate the Number of Parameters in the model being used for inference
    total_params = get_model_parameters(model)
    print(f"Total number of paramerers: {total_params}")

    # Prepare the input text you want to generate predictions for
    inputs = tokenizer(input_text, return_tensors='pt')

    # Generate Text
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained model.")
    parser.add_argument('input_text', type=str, help="The input text to generate from.")
    args = parser.parse_args()
    main(args.input_text)