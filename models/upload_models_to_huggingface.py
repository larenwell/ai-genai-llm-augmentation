import os
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import HfApi, create_repo, CommitOperationAdd, create_commit
import time

# Load environment variables from .env file in the root directory
load_dotenv(find_dotenv())

def upload_model_to_hub(model_name, model_path, token):
    """
    Upload a model to the Hugging Face Hub.

    Args:
        model_name (str): The name of the model to upload.
        model_path (str): The local path to the model files.
        token (str): The Hugging Face API token.

    Raises:
        Exception: If the upload fails after all retries.
    """
    api = HfApi()

    # Check if the repository exists, create it if it doesn't
    try:
        api.repo_info(repo_id=f"larenwell/{model_name}", token=token)
        print(f"Repository {model_name} already exists on Hugging Face Hub.")
    except Exception as e:
        create_repo(repo_id=f"larenwell/{model_name}", token=token)
        print(f"Created repository {model_name} on Hugging Face Hub.")

    # Prepare files for upload
    files_to_upload = []
    for root, dirs, files in os.walk(model_path):
        for file in files:
            file_path = os.path.join(root, file)
            repo_path = os.path.relpath(file_path, model_path)
            files_to_upload.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=file_path))

    # Retry logic for uploading files
    retries = 5
    for attempt in range(retries):
        try:
            create_commit(
                repo_id=f"larenwell/{model_name}",
                operations=files_to_upload,
                commit_message=f"Add {model_name}",
                token=token,
            )
            print(f"Model {model_name} has been uploaded to the Hugging Face Hub")
            break
        except Exception as e:
            print(f"Upload failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

def main():
    """
    Main function to upload models to the Hugging Face Hub.
    
    Raises:
        ValueError: If the Hugging Face API token is not found.
    """
    # Get the Hugging Face token from environment variables
    token = os.getenv("HUGGING_FACE_API_TOKEN")

    if not token:
        raise ValueError("Hugging Face API token not found. Please set it in the .env file.")

    # Define model paths
    models = {
        "flan_t5_small_finetuned_kmfodabooksum_13books": "./models/flan_t5_small_finetuned_kmfodabooksum_13books",
        "flan_t5_small_finetuned_kmfodabooksum_75books": "./models/flan_t5_small_finetuned_kmfodabooksum_75books",
        "flan_t5_small_finetuned_kmfodabooksum_77books": "./models/flan_t5_small_finetuned_kmfodabooksum_77books"
    }

    # Upload models
    for model_name, model_path in models.items():
        upload_model_to_hub(model_name, model_path, token)

if __name__ == "__main__":
    main()
