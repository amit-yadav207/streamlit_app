# import os
# import torch
# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import bot_template, user_template
# from langchain.llms import HuggingFaceHub

# VECTOR_STORE_PATH = "vector_store.faiss"

# def extract_text_from_pdfs(pdf_docs):
#     """Extracts text from a list of PDF documents."""
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def split_text_into_chunks(text):
#     """Splits the provided text into chunks for processing."""
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     return text_splitter.split_text(text)

# def create_vector_store(text_chunks):
#     """Creates a vector store from text chunks using embeddings."""
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={'device': device})
#     vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     vector_store.save_local(VECTOR_STORE_PATH)
#     return vector_store

# def load_vector_store(vector_store_path):
#     """Loads a vector store from a local file if it exists."""
#     if os.path.exists(vector_store_path):
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={'device': device})
#         return FAISS.load_local(vector_store_path, embeddings=embeddings)
#     return None

# def initialize_conversation_chain(vector_store, model_repo_id):
#     """Initializes the conversational chain using the vector store and selected model."""
#     huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
#     llm = HuggingFaceHub(
#         repo_id=model_repo_id,
#         huggingfacehub_api_token=huggingfacehub_api_token,
#         model_kwargs={"temperature": 0.5, "max_length": 512}
#     )
#     llm.client.api_url = f'https://api-inference.huggingface.co/models/{model_repo_id}'
    
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vector_store.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

# def handle_user_input(user_question):
#     """Handles user input and updates the chat history."""
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(reversed(st.session_state.chat_history)):
#         template = bot_template if i % 2 == 0 else user_template
#         st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# def clear_chat():
#     """Clears the chat history."""
#     st.session_state.chat_history = None
#     st.session_state.conversation = None
#     st.session_state.vector_store = None
#     st.write("Chat cleared.")

# def main():
#     """Main function to run the Streamlit app."""
#     load_dotenv()
#     st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None
#     if "vector_store" not in st.session_state:
#         st.session_state.vector_store = None

#     st.header("Chat with Multiple PDFs :books:")

#     if st.button("Clear Chat"):
#         clear_chat()

#     user_question = st.text_input("Ask a question about your documents:")

#     if user_question:
#         handle_user_input(user_question)
        

#     with st.sidebar:
#         st.subheader("Model Selection")
#         model_repo_id = st.selectbox(
#             "Select a model:",
#             ["mistralai/Mixtral-8x7B-Instruct-v0.1", "google/flan-t5-xxl", "mistralai/Mistral-7B-Instruct-v0.2"]
#         )


#         st.subheader("Vector Store Selection")
#         vector_store_option = st.selectbox(
#             "Select a vector store:",
#             ["vector_store_ipc.faiss", "vector_store_git.faiss", "vector_store_custom.faiss"]
#         )

#         load_vector_button = st.button("Load Vector Store")
        
#         if load_vector_button:
#             with st.spinner("Loading Vector Store..."):
#                 st.session_state.vector_store = load_vector_store(vector_store_option)
#                 if st.session_state.vector_store:
#                     st.session_state.conversation = initialize_conversation_chain(st.session_state.vector_store, model_repo_id)
#                     st.success(f"Vector store '{vector_store_option}' loaded successfully!")
#                 else:
#                     st.warning("Failed to load vector store. Please ensure the file exists.")


#         st.subheader("Your Documents")
#         pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        
#         if st.button("Process", disabled=not pdf_docs):
#             if not st.session_state.vector_store and pdf_docs:
#                 with st.spinner("Processing..."):
#                     raw_text = extract_text_from_pdfs(pdf_docs)
#                     text_chunks = split_text_into_chunks(raw_text)
#                     st.session_state.vector_store = create_vector_store(text_chunks)
#                     st.session_state.conversation = initialize_conversation_chain(st.session_state.vector_store, model_repo_id)
#                     st.success("Embeddings created successfully and vector store is ready!")
#             elif not pdf_docs:
#                 st.warning("Please upload at least one PDF document.")
#             else:
#                 st.session_state.conversation = initialize_conversation_chain(st.session_state.vector_store, model_repo_id)
#                 st.success(f"Vector store '{VECTOR_STORE_PATH}' loaded successfully!")
        

# if __name__ == '__main__':
#     main()



import os
import torch
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import bot_template, user_template
from langchain.llms import HuggingFaceHub

VECTOR_STORE_PATH = "vector_store.faiss"

def extract_text_from_pdfs(pdf_docs):
    """Extracts text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text):
    """Splits the provided text into chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def create_vector_store(text_chunks):
    """Creates a vector store from text chunks using embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={'device': device})
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store

def load_vector_store(vector_store_path):
    """Loads a vector store from a local file if it exists."""
    if os.path.exists(vector_store_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={'device': device})
        return FAISS.load_local(vector_store_path, embeddings=embeddings)
    return None

def initialize_conversation_chain(vector_store, model_repo_id):
    """Initializes the conversational chain using the vector store and selected model."""
    huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    llm = HuggingFaceHub(
        repo_id=model_repo_id,
        huggingfacehub_api_token=huggingfacehub_api_token,
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    llm.client.api_url = f'https://api-inference.huggingface.co/models/{model_repo_id}'
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    """Handles user input and updates the chat history."""
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(reversed(st.session_state.chat_history)):
        template = bot_template if i % 2 == 0 else user_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def clear_text():
    """Clears the text input without affecting the chat or other states."""
    st.session_state["user_question"] = ""

def clear_chat():
    """Clears the chat history and resets conversation state."""
    st.session_state.chat_history = None
    st.session_state.conversation = None
    st.session_state.vector_store = None
    st.write("Chat cleared.")

def main():
    """Main function to run the Streamlit app."""
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    st.header("Chat with Multiple PDFs :books:")

    if st.button("Clear Chat"):
        clear_chat()

    user_question = st.text_input("Ask a question about your documents:", key="user_question")

    if user_question:
        handle_user_input(user_question)

    st.button("Clear Text Input", on_click=clear_text)

    with st.sidebar:
        st.subheader("Model Selection")
        model_repo_id = st.selectbox(
            "Select a model:",
            ["mistralai/Mixtral-8x7B-Instruct-v0.1", "google/flan-t5-xxl", "mistralai/Mistral-7B-Instruct-v0.2"]
        )

        st.subheader("Vector Store Selection")
        vector_store_option = st.selectbox(
            "Select a vector store:",
            ["vector_store_ipc.faiss", "vector_store_git.faiss", "vector_store_custom.faiss"]
        )

        load_vector_button = st.button("Load Vector Store")

        if load_vector_button:
            with st.spinner("Loading Vector Store..."):
                st.session_state.vector_store = load_vector_store(vector_store_option)
                if st.session_state.vector_store:
                    st.session_state.conversation = initialize_conversation_chain(st.session_state.vector_store, model_repo_id)
                    st.success(f"Vector store '{vector_store_option}' loaded successfully!")
                else:
                    st.warning("Failed to load vector store. Please ensure the file exists.")

        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)

        if st.button("Process", disabled=not pdf_docs):
            if not st.session_state.vector_store and pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = extract_text_from_pdfs(pdf_docs)
                    text_chunks = split_text_into_chunks(raw_text)
                    st.session_state.vector_store = create_vector_store(text_chunks)
                    st.session_state.conversation = initialize_conversation_chain(st.session_state.vector_store, model_repo_id)
                    st.success("Embeddings created successfully and vector store is ready!")
            elif not pdf_docs:
                st.warning("Please upload at least one PDF document.")
            else:
                st.session_state.conversation = initialize_conversation_chain(st.session_state.vector_store, model_repo_id)
                st.success(f"Vector store '{VECTOR_STORE_PATH}' loaded successfully!")

if __name__ == '__main__':
    main()
