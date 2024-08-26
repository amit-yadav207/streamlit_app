import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub

VECTOR_STORE_PATH = "vector_store.faiss"

def load_vector_store():
    """Loads a vector store from a local file if it exists."""
    if os.path.exists(VECTOR_STORE_PATH):
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings=embeddings)
    return None

def main():
    """Main function to run the Streamlit app."""
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDFs :books:")

    vector_store = load_vector_store()
    if vector_store:
        st.write(vector_store.similarity_search("what is section for murder"))
        st.write("Vector store loaded successfully!")
    else:
        st.write("Vector store not found. Please upload it.")

if __name__ == '__main__':
    main()
