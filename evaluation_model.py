import os
import json
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer, PegasusForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration, 
    GPTNeoForCausalLM, GPT2Tokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
)
from datasets import load_dataset
import evaluate
import torch

def preprocess_function_booksum(examples):
    """
    Preprocess the BookSum dataset.

    Args:
        examples (dict): A dictionary containing 'chapter' and 'summary_text' keys.

    Returns:
        dict: A dictionary with tokenized inputs and labels for the model.
    """
    inputs = [text for text in examples["chapter"]]
    targets = [summary for summary in examples["summary_text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(text_target=targets, max_length=150, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_model_and_tokenizer(model_directory, model_type):
    """
    Load a model and tokenizer based on the model type.

    Args:
        model_directory (str): The directory where the model is stored.
        model_type (str): The type of model to load (e.g., "pegasus", "bart").

    Returns:
        tuple: A tuple containing the tokenizer and the model.
    """
    if model_type == "pegasus":
        tokenizer = PegasusTokenizer.from_pretrained(model_directory)
        model = PegasusForConditionalGeneration.from_pretrained(model_directory)
    elif model_type == "bart":
        tokenizer = BartTokenizer.from_pretrained(model_directory)
        model = BartForConditionalGeneration.from_pretrained(model_directory)
    elif model_type == "flan-t5":
        tokenizer = T5Tokenizer.from_pretrained(model_directory)
        model = T5ForConditionalGeneration.from_pretrained(model_directory)
    elif model_type == "gpt-neo":
        tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
        model = GPTNeoForCausalLM.from_pretrained(model_directory)
    elif model_type == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(model_directory)
        model = LlamaForCausalLM.from_pretrained(model_directory)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        if "causal" in model_type:
            model = AutoModelForCausalLM.from_pretrained(model_directory)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_directory)
    return tokenizer, model

def generate_summary(texts):
    """
    Generate summaries from input texts.

    Args:
        texts (list): A list of input texts to summarize.

    Returns:
        list: A list of generated summaries.
    """
    inputs = tokenizer(texts, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    summaries = [tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids]
    return summaries

def evaluate_summaries(dataset):
    """
    Evaluate summaries using the ROUGE metric.

    Args:
        dataset (Dataset): The dataset containing input texts and reference summaries.

    Returns:
        dict: A dictionary containing ROUGE scores.
    """
    batch_size = 16
    summaries = []
    references = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        batch_texts = [example['chapter'] for example in batch]
        batch_summaries = generate_summary(batch_texts)
        summaries.extend(batch_summaries)
        references.extend([example['summary_text'] for example in batch])
    return rouge.compute(predictions=summaries, references=references)

# Dictionary mapping model directories to their respective model types
model_directories = {
    "./models/flan_t5_small_finetuned_kmfodabooksum_13books": "flan-t5",  # train_batch_size: 2, eval_batch_size: 2, num_epochs: 3
    "./models/flan_t5_small_finetuned_kmfodabooksum_77books": "flan-t5",  # train_batch_size: 2, eval_batch_size: 2, num_epochs: 3
    "./models/flan_t5_small_finetuned_kmfodabooksum_75books": "flan-t5"   # train_batch_size: 4, eval_batch_size: 4, num_epochs: 5
}

# Load the ROUGE evaluation metric
rouge = evaluate.load('rouge')

# Dictionary to store evaluation results for each model
results = {}

# Loop through each model directory and evaluate the model
for directory, model_type in model_directories.items():
    # Load the model and tokenizer
    tokenizer, model = load_model_and_tokenizer(directory, model_type)

    # Load the BookSum test dataset
    booksum_test = load_dataset('kmfoda/booksum', split='test')
    
    # Preprocess the BookSum test dataset
    booksum_test = booksum_test.map(preprocess_function_booksum, batched=True)

    # Evaluate the model on the BookSum test dataset
    rouge_results = evaluate_summaries(booksum_test)
    
    # Store the ROUGE results for the current model
    results[directory] = rouge_results

# Print the evaluation results for each model
for model_dir, result in results.items():
    print(f"ROUGE Results for {model_dir}: ", json.dumps(result, indent=4))
