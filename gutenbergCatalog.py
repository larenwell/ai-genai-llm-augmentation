import rdflib
import glob

# Create an RDF graph
g = rdflib.Graph()

# Parse all the RDF/XML files in the extracted folder
rdf_files = glob.glob("rdf-files/cache/epub/*/*.rdf")

print(f"Found {len(rdf_files)} RDF files to parse.")  # Debug statement

for rdf_file in rdf_files:
    try:
        print(f"Parsing {rdf_file}")  # Debug statement
        g.parse(rdf_file, format="xml")
    except Exception as e:
        print(f"Error parsing {rdf_file}: {e}")  # Debug statement

# Helper function to get the value of a node, handling BNodes
def get_node_value(node):
    if isinstance(node, rdflib.term.BNode):
        for _, _, obj in g.triples((node, None, None)):
            return str(obj)
    return str(node)

# Extract book metadata
books_metadata = []
for book in g.subjects(rdflib.RDF.type, rdflib.URIRef("http://www.gutenberg.org/2009/pgterms/ebook")):
    title = g.value(book, rdflib.URIRef("http://purl.org/dc/terms/title"))
    author_node = g.value(book, rdflib.URIRef("http://purl.org/dc/terms/creator"))
    subjects = list(g.objects(book, rdflib.URIRef("http://purl.org/dc/terms/subject")))
    formats = list(g.objects(book, rdflib.URIRef("http://www.gutenberg.org/2009/pgterms/file")))
    
    author = get_node_value(author_node)
    subjects_list = [get_node_value(subject) for subject in subjects]
    
    # Extract URL from the formats
    text_url = None
    for format_url in formats:
        format_url_str = get_node_value(format_url)
        if format_url_str.endswith('.txt.utf-8') or format_url_str.endswith('.txt'):
            text_url = format_url_str
            break
    
    if title and text_url:
        books_metadata.append({
            "title": str(title),
            "author": author if author else "Unknown",
            "subjects": subjects_list,
            "url": str(text_url)
        })

# Print the number of books found
print(f"Extracted metadata for {len(books_metadata)} books.")  # Debug statement

# Filter books for diverse genres and styles
# For simplicity, here we just select a few books manually
# You may want to implement a more sophisticated selection based on actual genres
selected_books_metadata = [
    book for book in books_metadata
    if any(genre in book["subjects"] for genre in ["Fiction", "Science Fiction", "Mystery", "Biography"])
]

# Print selected book titles and URLs for verification
for book in selected_books_metadata[:5]:  # Limit to first 5 for display
    print(f"Title: {book['title']}, URL: {book['url']}")



