import streamlit as st
import os
import tempfile
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Set up page configuration
st.set_page_config(page_title="PDF QA System", layout="wide")

# Initialize session state for persistent data
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def process_pdfs(uploaded_files):
    """Process uploaded PDF files and create vector store"""
    processed_files = []
    progress_bar = st.progress(0)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        # Load and process PDF
        loader = PyPDFLoader(temp_file_path)
        data = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        all_splits = text_splitter.split_documents(data)
        
        # Create vector store
        persist_directory = f"data/pdf_{idx}"
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            persist_directory=persist_directory,
        )
        
        processed_files.append({
            'name': uploaded_file.name,
            'path': temp_file_path,
            'vectorstore': vectorstore
        })
        
        # Update progress bar
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    return processed_files

def setup_qa_chain(vectorstore):
    """Set up QA chain for a given vector store"""
    # LLM setup
    llm = Ollama(
        base_url="http://localhost:11434",
        model="llama3",
        verbose=True
    )
    
    # Retriever setup
    retriever = vectorstore.as_retriever()
    
    # Prompt template
    template = """ You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
    Context: {context}
    History: {history}
    User: {question}
    Chatbot:
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )
    
    # Memory setup
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )
    
    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": memory,
        }
    )
    
    return qa_chain

def landing_page():
    """Landing page for PDF upload"""
    st.title("PDF Question Answering System")
    
    # PDF Upload
    uploaded_files = st.file_uploader(
        "Upload PDFs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process PDFs"):
            # Process PDFs
            st.session_state.processed_pdfs = process_pdfs(uploaded_files)
            
            # Change page state
            st.session_state.page = 'pdf_view'
            # st.experimental_rerun()

def pdf_view_page():
    """PDF View and Chat Page"""
    st.title("PDF Viewer and Q&A")
    
    # Check if PDFs are processed
    if not st.session_state.processed_pdfs:
        st.warning("No PDFs processed. Please upload PDFs first.")
        if st.button("Go Back"):
            st.session_state.page = 'landing'
            st.experimental_rerun()
        return
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # PDF Viewer (using streamlit-pdf)
        st.header("PDF Viewer")
        pdf_file = st.selectbox(
            "Select PDF", 
            [pdf['name'] for pdf in st.session_state.processed_pdfs]
        )
        
        # In a real implementation, you'd use a PDF viewer library
        # For now, just show the filename
        st.write(f"Viewing: {pdf_file}")
    
    with col2:
        # Chat Interface
        st.header("Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['sender'] == 'user':
                st.write(f"**User**: {message['text']}")
            else:
                st.write(f"**Assistant**: {message['text']}")
        
        # Chat input
        if "user_query" not in st.session_state:
            st.session_state.user_query = ""

        user_query = st.text_area("Ask a question about the PDF", value=st.session_state.user_query, key="chat_input")

        if user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({
                'sender': 'user',
                'text': user_query
            })
             
            # Find the current selected PDF's vectorstore
            current_pdf = next(
                pdf for pdf in st.session_state.processed_pdfs 
                if pdf['name'] == pdf_file
            )
            
            # Setup QA chain
            qa_chain = setup_qa_chain(current_pdf['vectorstore'])
            
            # Get response
            with st.spinner("Processing..."):
                response = qa_chain.invoke({"query": user_query})
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                'sender': 'assistant',
                'text': response.get('result', 'Unable to process the query.')
            })


            
            # Rerun to display new messages and clear the input
            st.session_state.user_query = ""
            st.experimental_rerun()

# Main app logic
def main():
    if st.session_state.page == 'landing':
        landing_page()
    elif st.session_state.page == 'pdf_view':
        pdf_view_page()

# Run the main app
if __name__ == "__main__":
    main()