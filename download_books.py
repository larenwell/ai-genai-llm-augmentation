import os
import requests

# Function to save text to a file
def save_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

# Function to download a book by its ID
def download_book(book_id, directory):
    try:
        url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        text = response.text
        # Create a valid filename
        filename = os.path.join(directory, f"book_{book_id}.txt")
        # Save the text to a file
        save_to_file(text, filename)
        print(f"Downloaded and saved: book_{book_id}")
    except Exception as e:
        print(f"Failed to download book {book_id}: {e}")

# Main function to download a selection of books
def main():
    # Create a directory to save the books
    directory = 'gutenberg'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # A selection of book IDs from different genres and styles
    book_ids = [1342, 2701, 2600, 1400, 2554, 64317, 4300, 21728, 120, 730,
                132, 1232, 23, 1228, 1497, 1322, 1065, 8800, 20, 6130,
                84, 36, 35, 5230, 62,11, 16, 55, 120, 113,
                82, 1257, 135, 98, 1240, 1661, 345, 2852, 155, 583,
                100, 289, 236, 709, 1341, 42, 3296, 2397, 2376, 20203,
                2680, 4363, 4280, 1998, 3600, 1524, 1533, 2542, 3825, 1491,
                768, 1260, 161, 105, 158, 215, 910, 521, 103, 164,
                43, 174, 10007, 209, 5120]

    for book_id in book_ids:
        download_book(book_id, directory)

if __name__ == "__main__":
    main()