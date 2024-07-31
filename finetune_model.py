import os
import pandas as pd
import gradio as gr
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, LlamaTokenizer, LlamaForCausalLM,
    PegasusTokenizer, PegasusForConditionalGeneration, BartTokenizer, BartForConditionalGeneration, 
    T5Tokenizer, T5ForConditionalGeneration, GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling)

# Load environment variable for Hugging Face API token
token = os.getenv("HUGGING_FACE_API_TOKEN")

# Define the model name for the Llama 2 model
model_name = "meta-llama/Llama-2-7b-hf"

try:
    print(f"Attempting to load tokenizer for {model_name}")
    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=token)
    print("Tokenizer loaded successfully")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Attempting to load model for {model_name}")
    model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=token)
    print("Model loaded successfully")
    model.config.pad_token_id = tokenizer.pad_token_id
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please check your HUGGING_FACE_API_TOKEN and ensure you have access to the Llama 2 model.")


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
                data.append({"text": text})
    return Dataset.from_pandas(pd.DataFrame(data))


def preprocess_function_gutenberg(examples):
    """
    Preprocess Gutenberg books data.
    
    Args:
        examples (dict): Dictionary containing book text data.
    
    Returns:
        dict: Dictionary with tokenized inputs and labels for the model.
    """
    inputs = [text for text in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", pad_to_max_length=True) #pad_to_max_length=True
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


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
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", pad_to_max_length=True) #pad_to_max_length=True
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length", pad_to_max_length=True) #pad_to_max_length=True
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def fine_tune_model():
    """
    Fine-tune the model.
    
    The function performs the following steps:
    - Loads and preprocesses Gutenberg books data.
    - Loads and preprocesses BookSum training data.
    - Combines the datasets for training.
    - Loads and preprocesses BookSum test data.
    - Configures and runs the Trainer for fine-tuning the model.
    """

    # Load the cleaned Gutenberg books data from the specified directory
    books_data = load_books_data('data/train_cleaned_gutenberg')
    
    # Map the preprocessing function to the Gutenberg books data, batching the operation and removing the original text column
    books_data = books_data.map(preprocess_function_gutenberg, batched=True, remove_columns=["text"])

    # Load the BookSum training dataset from the Hugging Face dataset repository and map the preprocessing function to the BookSum training data, batching the operation
    booksum_train = load_dataset('kmfoda/booksum', split='train')
    booksum_train = booksum_train.map(preprocess_function_booksum, batched=True)

    # Combine the Gutenberg books data and the BookSum training data into a single training dataset
    combined_train_dataset = concatenate_datasets([books_data, booksum_train])

    # Load the BookSum test dataset from the Hugging Face dataset repository and map the preprocessing function to the BookSum test data, batching the operation
    booksum_test = load_dataset('kmfoda/booksum', split='test')
    booksum_test = booksum_test.map(preprocess_function_booksum, batched=True)

    # Assign the combined training dataset to the train_dataset variable
    train_dataset = combined_train_dataset

    # Assign the BookSum test dataset to the eval_dataset variable
    eval_dataset = booksum_test

    # Set the number of gradient accumulation steps and define the effective batch size
    gradient_accumulation_steps = 4
    effective_batch_size = 16 

    # Calculate the per-device training batch size by dividing the effective batch size by the gradient accumulation steps
    per_device_train_batch_size = effective_batch_size // gradient_accumulation_steps

    # Set up the training arguments for the Trainer
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=3,
        save_steps=10,
        logging_dir="./logs",
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True
    )

    # Create a data collator for language modeling, without using masked language modeling to handle padding and token masking.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize the Trainer with the model, training arguments, datasets, tokenizer, and data collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Start the training process
    trainer.train()

    # Push the trained model to the Hugging Face Hub
    trainer.push_to_hub("llama2_finetuned_kmfodabooksum")

    # Evaluate the trained model and print the results
    results = trainer.evaluate()
    print(results)

    # Save the fine-tuned model and tokenizer to the specified directory
    save_directory = "./models/llama2_finetuned_kmfodabooksum"
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

if __name__ == "__main__":
    fine_tune_model()