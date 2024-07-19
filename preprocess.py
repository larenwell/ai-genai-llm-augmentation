import os
import re
from nltk.tokenize import sent_tokenize

# Function to clean text
def clean_text(text):
    # Eliminate handles and URLs
    text = re.sub(r'@\w+|https?://\S+', '', text)
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    # Join sentences to form cleaned text
    cleaned_text = ' '.join(sentences)
    return cleaned_text

# Function to process and save cleaned text to a file
def process_and_save(filename, source_dir, target_dir):
    with open(os.path.join(source_dir, filename), 'r', encoding='utf-8') as file:
        text = file.read()
        cleaned_text = clean_text(text)
        with open(os.path.join(target_dir, filename), 'w', encoding='utf-8') as cleaned_file:
            cleaned_file.write(cleaned_text)

# Main function to process all files in the source directory
def main():
    source_dir = 'gutenberg'
    target_dir = 'cleaned_gutenberg'

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Process each file in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.txt'):
            process_and_save(filename, source_dir, target_dir)
            print(f"Processed and saved: {filename}")

if __name__ == "__main__":
    main()
