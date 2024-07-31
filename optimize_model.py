import os
import pandas as pd
import optuna
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments)

# Load environment variable for Hugging Face API token
token = os.getenv("HUGGING_FACE_API_TOKEN")

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
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", pad_to_max_length=True)
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
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", pad_to_max_length=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length", pad_to_max_length=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def fine_tune_model():
    """
    Fine-tune the model using Optuna for hyperparameter optimization.
    """
    global model, tokenizer, train_dataset, eval_dataset

    # Define model name and load tokenizer and model
    model_name = 'google/flan-t5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load and preprocess Gutenberg books data
    books_data = load_books_data('data/train_cleaned_gutenberg')
    books_data = books_data.map(preprocess_function_gutenberg, batched=True, remove_columns=["text"])

    # Load and preprocess BookSum training data
    booksum_train = load_dataset('kmfoda/booksum', split='train')
    booksum_train = booksum_train.map(preprocess_function_booksum, batched=True)

    # Combine Gutenberg and BookSum training data
    combined_train_dataset = concatenate_datasets([books_data, booksum_train])

    # Load and preprocess BookSum test data
    booksum_test = load_dataset('kmfoda/booksum', split='test')
    booksum_test = booksum_test.map(preprocess_function_booksum, batched=True)

    # Assign combined training dataset and test dataset
    train_dataset = combined_train_dataset
    eval_dataset = booksum_test

    def objective(trial):
        """
        Objective function for Optuna hyperparameter optimization.

        Args:
            trial (Trial): An Optuna trial object.

        Returns:
            float: The evaluation loss.
        """
        # Suggest hyperparameters for the trial
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
        batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
        num_train_epochs = trial.suggest_categorical('num_train_epochs', [3, 5, 7])

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            save_total_limit=3,
            save_steps=10,
            logging_dir="./logs",
            gradient_accumulation_steps=4,
            gradient_checkpointing=True 
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        eval_results = trainer.evaluate()
        return eval_results['eval_loss']

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    # Print the best trial results
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    # Call the fine-tune model function when the script is executed
    fine_tune_model()
