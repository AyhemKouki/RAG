import streamlit as st
import os
from rag_pipeline import ask_question

st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")

PDF_FOLDER = "data/pdfs"

# Load available PDFs
pdfs = [
    f.replace(".pdf", "")
    for f in os.listdir(PDF_FOLDER)
    if f.endswith(".pdf")
]

if not pdfs:
    st.error("No PDFs found in data/pdfs")
    st.stop()

# Sidebar
st.sidebar.header("Settings")

selected_pdf = st.sidebar.selectbox(
    "Select PDF",
    pdfs
)

st.sidebar.markdown(f"**Selected Collection:** `{selected_pdf}`")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input("Ask a question about the PDF...")

if question:

    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                answer = ask_question(question, selected_pdf)

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                error_message = f"Error: {str(e)}"

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })