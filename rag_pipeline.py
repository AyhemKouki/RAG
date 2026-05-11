from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os

from ingest import db , embeddings

# CONFIGURATION

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "llama-3.3-70b-versatile"

TOP_K = 3

# LOAD LLM

print("Loading LLM...")

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY
)

# PROMPT TEMPLATE

template = """
Tu es un assistant pedagogique.

Tu dois repondre UNIQUEMENT a partir du contexte fourni.

Consignes importantes :
- Reponds en francais.
- Si l'information n'est pas presente dans le contexte, dis clairement :
  "Je ne trouve pas cette information dans les documents fournis."
- Donne une reponse claire, structuree et concise.
- Cite les sources utilisees a la fin.

Contexte :
{context}

Question :
{question}

Reponse :
"""

prompt = PromptTemplate.from_template(template)

# RETRIEVAL FUNCTION

def retrieve_documents(query):

    docs = db.similarity_search(
        query,
        k=TOP_K
    )

    return docs

# BUILD CONTEXT

def build_context(docs):

    context = ""

    for doc in docs:

        source = doc.metadata.get("source", "Unknown")

        context += f"\n[Source: {source}]\n"
        context += doc.page_content
        context += "\n"

    return context

# EXTRACT SOURCES

def extract_sources(docs):

    sources = []

    for doc in docs:

        source = doc.metadata.get("source", "Unknown")

        if source not in sources:
            sources.append(source)

    return sources

# ASK FUNCTION

def ask_question(question):

    print("\nSearching relevant documents...\n")

    docs = retrieve_documents(question)

    if not docs:

        return "Aucun document pertinent trouve."

    # BUILD CONTEXT

    context = build_context(docs)

    # BUILD FINAL PROMPT

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    # GENERATE RESPONSE

    response = llm.invoke(final_prompt)

    # SOURCES

    sources = extract_sources(docs)

    final_response = response.content

    final_response += "\n\nSources :\n"

    for source in sources:

        final_response += f"- {source}\n"

    return final_response

# MAIN LOOP

if __name__ == "__main__":

    print("\n===== AI Academic Assistant =====\n")

    while True:

        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        answer = ask_question(question)

        print("\n=================================\n")
        print(answer)
        print("\n=================================\n")