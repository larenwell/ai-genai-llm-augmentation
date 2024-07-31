import os
import re
import requests
import json
from nltk.tokenize import sent_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

# Define the best model and load the tokenizer and model from Hugging Face
best_model_finetuned_hf = "larenwell/flan_t5_small_finetuned_kmfodabooksum_75books"
tokenizer = T5Tokenizer.from_pretrained(best_model_finetuned_hf)
model = T5ForConditionalGeneration.from_pretrained(best_model_finetuned_hf)

# Load the sentence transformer model for generating embeddings
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize the Qdrant client
client = QdrantClient(url="http://localhost:6333")

# Define the name of the collection in Qdrant
collection_name = 'gutenberg_book_embeddings_cleaned'

# Create a new collection in Qdrant if it doesn't already exist
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=384, distance=rest.Distance.COSINE)
    )

def save_to_file(text, filename):
    """
    Save text to a file.

    Args:
        text (str): The text to save.
        filename (str): The name of the file to save the text in.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

def download_book(book_id, directory):
    """
    Download a book from Project Gutenberg.

    Args:
        book_id (int): The ID of the book to download.
        directory (str): The directory to save the downloaded book.

    Raises:
        Exception: If the download fails.
    """
    try:
        url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        text = response.text
        filename = os.path.join(directory, f"book_id_{book_id}.txt")
        save_to_file(text, filename)
        print(f"Downloaded and saved: book_{book_id}")
    except Exception as e:
        print(f"Failed to download book {book_id}: {e}")

# Define the regex pattern to remove introductory text from Project Gutenberg books
intro_text_pattern = re.compile(
    r"The Project Gutenberg eBook of .*?\n"
    r"This ebook is for the use of anyone anywhere in the United States and\n"
    r"most other parts of the world at no cost and with almost no restrictions\n"
    r"whatsoever. You may copy it, give it away or re-use it under the terms\n"
    r"of the Project Gutenberg License included with this ebook or online\n"
    r"at www\.gutenberg\.org. If you are not located in the United States,\n"
    r"you will have to check the laws of the country where you are located\n"
    r"before using this eBook\.\n", 
    re.DOTALL
)

def clean_text(text):
    """
    Clean text by removing introductory text and unwanted characters.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(intro_text_pattern, '', text)
    text = re.sub(r'@\w+|https?://\S+', '', text)
    sentences = sent_tokenize(text)
    cleaned_text = ' '.join(sentences)
    return cleaned_text

def process_and_save(filename, source_dir, target_dir):
    """
    Process and clean text from a file and save it to a new file.

    Args:
        filename (str): The name of the file to process.
        source_dir (str): The directory containing the source file.
        target_dir (str): The directory to save the cleaned file.
    """
    with open(os.path.join(source_dir, filename), 'r', encoding='utf-8') as file:
        text = file.read()
        cleaned_text = clean_text(text)
        with open(os.path.join(target_dir, filename), 'w', encoding='utf-8') as cleaned_file:
            cleaned_file.write(cleaned_text)

def chunk_text(text, chunk_size=512):
    """
    Chunk text into smaller pieces based on a specified chunk size.

    Args:
        text (str): The text to chunk.
        chunk_size (int): The maximum size of each chunk.

    Returns:
        list: A list of text chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))
        if current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_and_save(filename, source_dir, target_dir, chunk_size=512):
    """
    Chunk text from a file and save each chunk as a separate file.

    Args:
        filename (str): The name of the file to chunk.
        source_dir (str): The directory containing the source file.
        target_dir (str): The directory to save the chunked files.
        chunk_size (int): The maximum size of each chunk.
    """
    with open(os.path.join(source_dir, filename), 'r', encoding='utf-8') as file:
        text = file.read()
        chunks = chunk_text(text, chunk_size)
        for i, chunk in enumerate(chunks):
            chunk_filename = f"{filename[:-4]}_chunk_{i}.txt"
            with open(os.path.join(target_dir, chunk_filename), 'w', encoding='utf-8') as chunk_file:
                chunk_file.write(chunk)

def generate_embeddings(text):
    """
    Generate embeddings for a given text using the sentence transformer model.

    Args:
        text (str): The text to generate embeddings for.

    Returns:
        numpy.ndarray: The generated embeddings.
    """
    return embedding_model.encode(text)

def process_and_upload_embeddings_from_chunked(chunked_dir, collection_name, book_details):
    """
    Process chunked text files, generate embeddings, and upload them to Qdrant.

    Args:
        chunked_dir (str): The directory containing chunked text files.
        collection_name (str): The name of the Qdrant collection.
        book_details (list): A list of book details with metadata.
    """
    batch_size = 100
    points = []
    unique_id = 0

    for filename in os.listdir(chunked_dir):
        if filename.endswith('.txt'):
            with open(chunked_dir + "/" + filename, 'r', encoding='utf-8') as file:
                text = file.read()
                embeddings = generate_embeddings(text)
                book_id = int(filename.split('_')[2])
                book_detail = next((book for book in book_details if book["id"] == book_id), None)
                if book_detail:
                    points.append(
                        rest.PointStruct(
                            id=unique_id,
                            vector=embeddings.tolist(),
                            payload={
                                "filename": filename,
                                "title": book_detail["title"],
                                "author": book_detail["author"],
                                "context": book_detail["context"],
                                "text": text
                            }
                        )
                    )
                    unique_id += 1 

                    if len(points) >= batch_size:
                        client.upsert(collection_name=collection_name, points=points)
                        points = []
                        print(f"Uploaded a batch of embeddings")

    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"Uploaded the final batch of embeddings")

def main():
    """
    Main function to download, process, chunk, and upload books to Qdrant.
    """
    # STEP 1. Load additional Project Gutenberg books
    with open('data/books_external.json', 'r', encoding='utf-8') as file:
        book_details = json.load(file)["external_books_id"]

    source_dir = 'data/external_data'
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
    
    for book in book_details:
        book_id = book["id"]
        download_book(book_id, source_dir)

    # STEP 2. Data Cleaning
    target_dir = 'data/external_cleaned_data'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith('.txt'):
            process_and_save(filename, source_dir, target_dir)
            print(f"Processed and saved: {filename}")

    # STEP 3: Chunking
    chunked_dir = 'data/external_chunked_data'
    if not os.path.exists(chunked_dir):
        os.makedirs(chunked_dir)

    for filename in os.listdir(target_dir):
        if filename.endswith('.txt'):
            chunk_and_save(filename, target_dir, chunked_dir)
            print(f"Chunked and saved: {filename}")
    
    # STEP 4 and 5: Generating embeddings and uploading them to Qdrant
    process_and_upload_embeddings_from_chunked(chunked_dir, collection_name, book_details)

if __name__ == "__main__":
    main()
