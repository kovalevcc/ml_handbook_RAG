import os
from langchain_community.document_loaders import BSHTMLLoader


def parse_html_with_langchain(data_dir):
    file_paths = [os.path.join(data_dir, file) for file in os.listdir(
        data_dir) if file.endswith(".html")]
    documents = []
    for file_path in file_paths:
        loader = BSHTMLLoader(file_path)
        docs = loader.load()
        documents.extend(docs)
    return documents
