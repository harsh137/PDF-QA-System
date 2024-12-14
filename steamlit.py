import streamlit as st
import os
import tempfile
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Set up page configuration
st.set_page_config(page_title="PDF QA System", layout="wide", page_icon="📄")

# Set background color to white
st.markdown("""
    <style>
        body {
            background-color: white;
        }
        .download-button {
            padding: 10px;
            background-color: #f1f1f1;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

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
            st.rerun()

def pdf_view_page():
    """PDF View and Chat Page"""
    st.title("PDF Viewer and Q&A")
    
    # Check if PDFs are processed
    if not st.session_state.processed_pdfs:
        st.warning("No PDFs processed. Please upload PDFs first.")
        if st.button("Go Back"):
            st.session_state.page = 'landing'
            st.rerun()
        return
    
    # Layout - 80% for chat and 20% for download section
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Chat Interface (taking up 80% of the space)
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
            # Add user message to chat history (original query shown to the user)
            st.session_state.chat_history.append({
                'sender': 'user',
                'text': user_query
            })
             
            # Append additional instructions to the query internally
            modified_query = f"{user_query}. Only from this pdf. Keep it short"
            
            # Get the current selected PDF's vectorstore
            pdf_file = st.selectbox(  
                "Select PDF for Chat", 
                [pdf['name'] for pdf in st.session_state.processed_pdfs],
                key="chat_pdf_select"
            )
            
            current_pdf = next(
                pdf for pdf in st.session_state.processed_pdfs 
                if pdf['name'] == pdf_file
            )
            
            # Setup QA chain
            qa_chain = setup_qa_chain(current_pdf['vectorstore'])
            
            # Get response using the modified query
            with st.spinner("Processing..."):
                response = qa_chain.invoke({"query": modified_query})
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                'sender': 'assistant',
                'text': response.get('result', 'Unable to process the query.')
            })

            # Rerun to display new messages and clear the input
            st.session_state.user_query = ""
            st.rerun()


    with col2:
        # Download section (taking up 20% of the space)
        st.header("Download PDF")
        
        # PDF download section
        pdf_file = st.selectbox(  # Ensure this is assigned before using
            "Select PDF", 
            [pdf['name'] for pdf in st.session_state.processed_pdfs],
            key="download_pdf_select"
        )
        
        # Find the selected PDF
        current_pdf = next(
            pdf for pdf in st.session_state.processed_pdfs 
            if pdf['name'] == pdf_file
        )
        
        # Download button for the selected PDF
        with open(current_pdf['path'], "rb") as pdf:
            st.download_button(
                label="Download PDF",
                data=pdf,
                file_name=current_pdf['name'],
                mime="application/pdf",
                use_container_width=True
            )


# Main app logic
def main():
    if st.session_state.page == 'landing':
        landing_page()
    elif st.session_state.page == 'pdf_view':
        pdf_view_page()

# Run the main app
if __name__ == "__main__":
    main()
