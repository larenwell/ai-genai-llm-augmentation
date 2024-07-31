import os
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

# Load the pre-trained FLAN-T5 model and tokenizer
model_name = 'google/flan-t5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define function to load and preprocess the books data
def load_books_data(data_dir):
    """
    Load books data from a directory.
    
    Args:
        data_dir (str): Path to the directory containing book files.
    
    Returns:
        Dataset: Hugging Face Dataset object with the loaded books data.
    """
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                data.append({"text": text})  # List of lists
    return Dataset.from_pandas(pd.DataFrame(data))

# Prepare the dataset for T5 model
def preprocess_function_gutenberg(examples):
    """
    Preprocess Gutenberg books data.
    
    Args:
        examples (dict): Dictionary containing book text data.
    
    Returns:
        dict: Dictionary with tokenized inputs and labels for the model.
    """
    inputs = [text for text in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Use the same inputs as targets for self-supervised learning
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(inputs, max_length=150, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Preprocess the booksum dataset
def preprocess_function_booksum(examples):
    """
    Preprocess BookSum data.
    
    Args:
        examples (dict): Dictionary containing chapter text and summary data.
    
    Returns:
        dict: Dictionary with tokenized inputs and labels for the model.
    """
    inputs = [text for text in examples["chapter"]]
    targets = [summary for summary in examples["summary_text"]]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Load the cleaned books data
books_data = load_books_data('cleaned_gutenberg')

# Apply the preprocessing function to the Gutenberg dataset
books_data = books_data.map(preprocess_function_gutenberg, batched=True, remove_columns=["text"])

# Load the train split of the booksum dataset
booksum_train = load_dataset('kmfoda/booksum', split='train')

# Preprocess the booksum train dataset
booksum_train = booksum_train.map(preprocess_function_booksum, batched=True)

# Combine the datasets
combined_train_dataset = concatenate_datasets([books_data, booksum_train])

# Load the evaluation dataset
booksum_test = load_dataset('kmfoda/booksum', split='test')

# Preprocess the evaluation dataset
booksum_test = booksum_test.map(preprocess_function_booksum, batched=True)

# Assign the combined training dataset to the train_dataset variable
train_dataset = combined_train_dataset

# Assign the BookSum test dataset to the eval_dataset variable
eval_dataset = booksum_test

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Adjusted to avoid deprecation warning
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=3,
    save_steps=10,
    logging_dir="./logs",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()
trainer.push_to_hub()

# Evaluate the model
results = trainer.evaluate()

# Print evaluation results
print(results)

# Save the model
save_directory = "./models/flan_t5_small_finetuned_kmfodabooksum_75books"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
