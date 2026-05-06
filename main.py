from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

GROQ_MODEL = "llama-3.3-70b-versatile"

# step 1 : load the PDF file and read it as a list of documents
# create instance of PyPDFLoader class
loader = PyPDFLoader("data/file.pdf")
# Open and read PDF file : returns a list of documents (metadata , source ...)
documents = loader.load()

print(f"Number of documents: {len(documents)}")

# step 2 : split the documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# split the documents into smaller chunks
chunks = splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")

#step 3 : Embed the chunks using HuggingFaceEmbeddings
# specify the model name for HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# step 4 : create a vector store using FAISS and add the embedded chunks to it
db = FAISS.from_documents(chunks, embeddings)

print(f"Number of vectors in the vector store: {db.index.ntotal}")
print(f"Number of documents in the vector store: {len(db.docstore._dict)}")

# step 5 : Retrieve relevant chunks from the vector store based on a query
query = "La technologie Ethernet"
docs = db.similarity_search(query, k=3)

# Load variables from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model=GROQ_MODEL, api_key=groq_api_key)

template = """
    Tu es un assistant pedagogique specialise en IA generative.
    Tu dois repondre uniquement a partir du contexte fourni.

    Consignes importantes :
    - Reponds en francais.
    - Si l’information n’est pas presente dans le contexte, dis clairement :
        "Je ne trouve pas cette information dans les documents fournis."
    - Donne une reponse claire, structuree et concise.
    - Termine par une ligne "Sources :" avec les fichiers utilises.

    Contexte :
    {context}

    Question :
    {question}
 """

context = "\n".join([doc.page_content for doc in docs])

prompt = PromptTemplate.from_template(template)
final_prompt = prompt.format(context=context, question=query)

response = llm.invoke(final_prompt)
print(response.content)