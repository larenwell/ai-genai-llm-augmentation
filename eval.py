from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, PegasusTokenizer, PegasusForConditionalGeneration, BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration, GPTNeoForCausalLM, GPT2Tokenizer
from datasets import load_metric,load_dataset

# Preprocess the booksum dataset
def preprocess_function_booksum(examples):
    inputs = [text for text in examples["chapter"]]
    targets = [summary for summary in examples["summary_text"]]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Function to load a model and tokenizer based on the model type
def load_model_and_tokenizer(model_directory, model_type):
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
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        if "causal" in model_type:
            model = AutoModelForCausalLM.from_pretrained(model_directory)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_directory)
    return tokenizer, model

# Directories of saved models and their types
model_directories = {
    "./models/fine_tuned_model_flan_t5_small_13books": "flan-t5",
    "./models/fine_tuned_model_flan_t5_small_75books": "flan-t5",
    "./models/fine_tuned_model_pegasus_75books": "pegasus",
    "./models/fine_tuned_model_bart_large_75books": "bart",
    "./models/fine_tuned_model_gpt_neo_1_3B_75books": "gpt-neo"
}

# Evaluate each model
rouge = load_metric('rouge')
results = {}

for directory, model_type in model_directories.items():
    tokenizer, model = load_model_and_tokenizer(directory, model_type)
    
    def generate_summary(text):
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def evaluate_summaries(dataset):
        summaries = []
        references = []
        for example in dataset:
            summary = generate_summary(example['chapter'])
            summaries.append(summary)
            references.append(example['summary_text'])
        return rouge.compute(predictions=summaries, references=references)

    # Load the test dataset
    booksum_test = load_dataset('kmfoda/booksum', split='test')
    booksum_test = booksum_test.map(preprocess_function_booksum, batched=True)

    rouge_results = evaluate_summaries(booksum_test)
    results[directory] = rouge_results

# Print evaluation results
for model_dir, result in results.items():
    print(f"ROUGE Results for {model_dir}: ", result)