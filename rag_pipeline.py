from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()

# CONFIG
CHROMA_DB_DIR = "data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 3

GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------------------------
# EMBEDDINGS
# ---------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ---------------------------
# RERANKER (optional but powerful)
# ---------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------------
# LLM
# ---------------------------
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=os.getenv("GROQ_API_KEY")
)

# ---------------------------
# PROMPT
# ---------------------------
template = """
Tu es un assistant pédagogique.

Réponds uniquement avec le contexte fourni.

Règles:
- Réponds en français
- Si l'information n'existe pas dans le contexte, dis:
  "Je ne trouve pas cette information dans ce document."
- Sois clair et structuré
- Cite les sources à la fin

Contexte:
{context}

Question:
{question}

Réponse:
"""

prompt = PromptTemplate.from_template(template)

# ---------------------------
# LOAD COLLECTION PER PDF
# ---------------------------
def load_db(collection_name: str):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

# ---------------------------
# RETRIEVAL
# ---------------------------
def retrieve_documents(query, collection_name):

    db = load_db(collection_name)

    docs = db.similarity_search(query, k=10)

    return docs

# ---------------------------
# RERANKING
# ---------------------------
def rerank(query, docs):

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked[:TOP_K]]

# ---------------------------
# CONTEXT BUILDER
# ---------------------------
def build_context(docs):

    context = ""

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")

        context += f"\n[Source: {source}]\n"
        context += doc.page_content
        context += "\n"

    return context

# ---------------------------
# SOURCE EXTRACTOR
# ---------------------------
def extract_sources(docs):

    return list(set(
        doc.metadata.get("source", "Unknown")
        for doc in docs
    ))

# ---------------------------
# MAIN ASK FUNCTION
# ---------------------------
def ask_question(question, collection_name):

    print("\nSearching documents...")

    # 1. retrieve
    docs = retrieve_documents(question, collection_name)

    if not docs:
        return "Aucun document trouvé."

    # 2. rerank
    docs = rerank(question, docs)

    # 3. build context
    context = build_context(docs)

    # 4. prompt
    final_prompt = prompt.format(
        context=context,
        question=question
    )

    # 5. LLM call
    response = llm.invoke(final_prompt)

    # 6. sources
    sources = extract_sources(docs)

    final_answer = response.content
    final_answer += "\n\nSources:\n"

    for s in sources:
        final_answer += f"- {s}\n"

    return final_answer