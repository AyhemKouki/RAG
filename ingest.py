from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

PDF_FOLDER = "data/pdfs"
CHROMA_DB_DIR = "data/chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)


def get_collection_name(pdf_file: str):
    # remove .pdf and clean name
    return pdf_file.replace(".pdf", "").replace(" ", "_").lower()


def ingest_pdfs():

    if not os.path.exists(PDF_FOLDER):
        print("PDF folder not found")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found")
        return

    print(f"Found {len(pdf_files)} PDFs\n")

    for pdf_file in pdf_files:

        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        collection_name = get_collection_name(pdf_file)

        print(f"\nProcessing: {pdf_file}")
        print(f"Collection: {collection_name}")

        # LOAD PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # SPLIT
        chunks = splitter.split_documents(docs)

        # ADD SOURCE
        for chunk in chunks:
            chunk.metadata["source"] = pdf_file

        # CREATE A SEPARATE VECTOR DB FOR THIS PDF
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR
        )

        # STORE
        db.add_documents(chunks)

        print(f"Stored {len(chunks)} chunks in {collection_name}")

    print("\nINGESTION DONE")

if __name__ == "__main__":
    ingest_pdfs()