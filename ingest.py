from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
import os

# CONFIGURATION

PDF_FOLDER = "data/pdfs"
CHROMA_DB_DIR = "data/chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

CHUNK_SIZE = 800 # i choose an 800 chunk size with an overlap of 10-30 % to maintain context and improve retrieval accuracy.
CHUNK_OVERLAP = 150

COLLECTION_NAME = "academic_documents"

# LOAD ENV VARIABLES

load_dotenv()

# LOAD EMBEDDING MODEL

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# CREATE / LOAD CHROMA DATABASE

print("Initializing ChromaDB...")

# If the ChromaDB directory doesn't exist, it will be created and a new collection will be initialized.
db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR
)

# TEXT SPLITTER

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# INGESTION FUNCTION

def ingest_pdfs():

    all_chunks = []

    # Check if folder exists
    if not os.path.exists(PDF_FOLDER):
        print(f"Folder not found: {PDF_FOLDER}")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"Found {len(pdf_files)} PDF(s)\n")

    for pdf_file in pdf_files:

        pdf_path = os.path.join(PDF_FOLDER, pdf_file)

        print(f"Processing: {pdf_file}")

        # LOAD PDF
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        print(f"Loaded {len(documents)} pages")
        
        # SPLIT DOCUMENTS

        chunks = splitter.split_documents(documents)

        print(f"Created {len(chunks)} chunks")
        
        # ADD METADATA
        
        for chunk in chunks:

            chunk.metadata["source"] = pdf_file

        all_chunks.extend(chunks)

        print("Done\n")

    # STORE IN CHROMADB

    if all_chunks:

        print("Storing embeddings into ChromaDB...")

        db.add_documents(all_chunks)

        print("\nIngestion completed successfully.")
        print(f"Total chunks stored: {len(all_chunks)}")

    else:

        print("No chunks to store.")

# RUN INGESTION

if __name__ == "__main__":

    ingest_pdfs()