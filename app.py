import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
def get_pdf_text(pdf_docs):
    text="" #to hold all the raw text from each pdf page
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, #1000 characters chunk size
        chunk_overlap=200,
        length_function=len
    )
    
    chunks=text_splitter.split_text(raw_text)
    return chunks
    

def main():
    load_dotenv()
    
    st.set_page_config(page_title="Chat with multiple PDFS",page_icon=':books:')

    st.header("Chat with multiple pdfs :books:")
    st.text_input("Ask about your documents:")
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs=st.file_uploader("Upload your pdfs here and click on process button",accept_multiple_files=True)
        if st.button("Process"):
            #write code inside the spinner
            with st.spinner("Loading"):
                
                #get pdf text
                raw_text=get_pdf_text(pdf_docs)
                
                
                #break text into chunks
                
                text_chunks=get_text_chunks(raw_text)
                st.write(text_chunks)
                #convert into vectors/embeddings
                
                #store into vector store
                pass
        
if __name__ == '__main__':
    main()