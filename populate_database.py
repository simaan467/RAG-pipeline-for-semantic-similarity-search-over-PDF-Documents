import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database")
    args = parser.parse_args()
    if args.reset:
        print("Resetting the database...")
        clear_database()
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents() -> list[Document]:
    documentloader = PyPDFDirectoryLoader(DATA_PATH)
    print("Loading documents from PDF files...")
    return documentloader.load()


def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())
    chunk_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items['ids'])
    print(f"Existing IDs in database: {len(existing_ids)}")
    new_chunks = []
    for chunk in chunk_with_ids:
        if chunk.metadata['id'] not in existing_ids:
            new_chunks.append(chunk)
    if len(new_chunks):
        print(f"adding {len(new_chunks)} new chunks to the database...")
        db.add_documents(new_chunks)
        db.persist()
    else:
        print("No new chunks to add to the database.")


def get_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get('source')
        page = chunk.metadata.get('page')
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        last_page_id =  current_page_id
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        chunk.metadata['id'] = chunk_id
    return chunks
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()

