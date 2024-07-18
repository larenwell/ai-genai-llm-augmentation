import requests
import zipfile
import os
import tarfile

# Download the catalog
catalog_url = "http://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.zip"
catalog_path = "rdf-files.tar.zip"

# Download the file
response = requests.get(catalog_url)
with open(catalog_path, 'wb') as file:
    file.write(response.content)

# Extract the zip file
with zipfile.ZipFile(catalog_path, 'r') as zip_ref:
    zip_ref.extractall("rdf-files")

# Extract the tar file inside the extracted folder
tar_path = os.path.join("rdf-files", "rdf-files.tar")
with tarfile.open(tar_path, "r:") as tar:
    tar.extractall(path="rdf-files")
