import os
import requests
import json
import re
from nltk.tokenize import sent_tokenize

# Common introductory text patterns to remove
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

# Function to save text to a file
def save_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

# Function to download a book from Project Gutenberg
def download_book(book_id, directory):
    try:
        url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        response = requests.get(url)
        response.raise_for_status()
        text = response.text
        filename = os.path.join(directory, f"book_{book_id}.txt")
        save_to_file(text, filename)
        print(f"Downloaded and saved: book_{book_id}")
    except Exception as e:
        print(f"Failed to download book {book_id}: {e}")

# Function to clean text
def clean_text(text):
    text = re.sub(intro_text_pattern, '', text)
    text = re.sub(r'@\w+|https?://\S+', '', text)
    sentences = sent_tokenize(text)
    cleaned_text = ' '.join(sentences)
    return cleaned_text

# Function to process and save cleaned text to a file
def process_and_save(filename, source_dir, target_dir):
    with open(os.path.join(source_dir, filename), 'r', encoding='utf-8') as file:
        text = file.read()
        cleaned_text = clean_text(text)
        with open(os.path.join(target_dir, filename), 'w', encoding='utf-8') as cleaned_file:
            cleaned_file.write(cleaned_text)

# Main function to download and process books
def main():
    download_directory = 'data/train_gutenberg'
    clean_directory = 'data/train_cleaned_gutenberg'

    if not os.path.exists(download_directory):
        os.makedirs(download_directory)
    if not os.path.exists(clean_directory):
        os.makedirs(clean_directory)

    # Load book details from JSON
    with open('data/books_training.json', 'r', encoding='utf-8') as file:
        book_details = json.load(file)["training_books_id"]
    
    # Download books
    for book in book_details:
        book_id = book["id"]
        download_book(book_id, download_directory)

    # Clean and process downloaded books
    for filename in os.listdir(download_directory):
        if filename.endswith('.txt'):
            process_and_save(filename, download_directory, clean_directory)
            print(f"Processed and saved: {filename}")

if __name__ == "__main__":
    main()
