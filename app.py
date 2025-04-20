import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

#API KEY
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("🚨 Please set your Google Gemini API key in a .env file.")
    st.stop()

# Custom CSS styling
def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
try:
    load_css("assets/styles.css")
except FileNotFoundError:
    st.warning("styles.css file not found. Using default styling.")

# Title
st.markdown("""<div class="centered-title">
    <h1 class="gradient-text">MediSynth AI</h1>
</div>
""", unsafe_allow_html=True)
st.markdown("""<div class="subheading-text">Context-Aware Medical Intelligence Platform</div>""", unsafe_allow_html=True)
st.markdown("""<div class="caption-text">✨ AI-Powered Patient History Analysis & Diagnostic Support</div>""", unsafe_allow_html=True)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("assets/bg.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["gemini-2.0-flash","llama3.2", "deepseek-r1", "deepseek-r1:1.5b", "qwen:1.8b"],
        index=0
    )

    st.markdown("<div style='margin-top: 50px'></div>", unsafe_allow_html=True)

    embedding_source = st.radio(
    "Embedding Provider",
    ["Gemini embedding/001", "nomic-embed-text"],
    index=0  # Default to Gemini for speed
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🏥 Medical History Analysis  
    - 🔍 AI-Powered Insights  
    - 📑 Summarization & Recommendations  
    - 💡 Context Awareness
    - 📄 Patient Document Processing
    -🧪  Lab Report Analysis
    """)
    st.divider()
    


if embedding_source == "nomic-embed-text":
    embedding_model = "nomic-embed-text"
else:
    embedding_model = "models/embedding-001"

# Initialize session state for vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks



# ---- Document Processing Functions ----
def process_document(uploaded_file, patient_name):
    # Create a temporary file to save the uploaded content
    # with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
    #     tmp_file.write(uploaded_file.getvalue())
    #     file_path = tmp_file.name
    
    # # Select the appropriate loader based on file type
    # file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if uploaded_file:
            raw_text = get_pdf_text(uploaded_file)
        # elif file_extension == 'txt':
        #     loader = TextLoader(file_path)
        # elif file_extension in ['docx', 'doc']:
        #     loader = Docx2txtLoader(file_path)
        # elif file_extension in ['md', 'markdown']:
        #     loader = UnstructuredMarkdownLoader(file_path)
        else:
            raw_text = get_pdf_text(uploaded_file)
            # os.unlink(file_path)  # Clean up the temp file
            return None, f"Unsupported file type"

        # documents = loader.load()
        
        # # Add patient name as metadata to all documents
        # for doc in documents:
        #     doc.metadata["patient_name"] = patient_name

        # Split documents into chunks
        text_chunks = get_text_chunks(raw_text)
        
        # Create embeddings and vector store
        if embedding_source == "Google Gemini":
            embeddings = GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=GOOGLE_API_KEY,
                )
           
        else:
            embeddings = OllamaEmbeddings(
            model=embedding_model
        )
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        
        # Clean up the temp file
        #
        
        return vector_store, None
    except Exception as e:
        # Clean up the temp file in case of error
        #os.unlink(file_path)
        return None, f"Error processing document: {str(e)}"

# ---- Gemini LLM Setup ----
def setup_llm():
    if selected_model == "gemini-2.0-flash":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
    else:
        return ChatOllama(
            model=selected_model,
            base_url="http://localhost:11434",
            temperature=0.3
        )

# ---- System Prompts ----
patient_context_system_template = """You are an expert AI medical assistant for physicians.
Use the provided patient medical history context to answer questions accurately.
If the information isn't in the patient's history, say so clearly.
Always respond professionally, as you're supporting a healthcare provider.
Patient information: {patient_name}

Context from the patient's medical records:
{context}"""

standard_system_template = """You are an expert AI medical history summarizing assistant. 
Provide concise, correct solutions with strategic diagnoses. Always respond in English."""

# ---- Session State Chat History ----
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hello! I'm MediSynth. How can I assist you today? 🏥"}]

# ---- Document Upload Section ----
st.markdown("### 📄 Patient Document Uploader")
with st.expander("Upload Patient Medical Records", expanded=not st.session_state.document_processed):
    patient_name = st.text_input("Patient Name", value=st.session_state.patient_name)
    uploaded_file = st.file_uploader("Upload medical history document", accept_multiple_files=True)
    
    if st.button("Process Document") and uploaded_file is not None and patient_name:
        with st.spinner("Processing document... This may take a moment."):
            vector_store, error = process_document(uploaded_file, patient_name)
            
            if error:
                st.error(error)
            else:
                st.session_state.vector_store = vector_store
                st.session_state.document_processed = True
                st.session_state.patient_name = patient_name
                st.success(f"✅ Document for patient '{patient_name}' processed successfully!")

# Display patient information when document is processed
if st.session_state.document_processed:
    st.markdown(f"### 👤 Active Patient: **{st.session_state.patient_name}**")
    if st.button("Reset Patient Data"):
        st.session_state.vector_store = None
        st.session_state.document_processed = False
        st.session_state.patient_name = ""
        st.session_state.message_log = [{"role": "ai", "content": "Patient data cleared. I'm ready to help with a new patient. 🏥"}]
        st.rerun()

st.markdown("### 💬 Chat Interface")
# ---- Chat Display ----
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# ---- Chat Input ----
user_query = st.chat_input("Ask about patient medical history..." if st.session_state.document_processed else "Ask general medical questions or upload a patient document...")

def generate_ai_response(user_query):
    llm = setup_llm()
    
    # Use RAG if vector store is available, otherwise use standard chat
    if st.session_state.vector_store:
        # Create a prompt that includes patient context
        prompt = ChatPromptTemplate.from_messages([
            ("system", patient_context_system_template),
            ("human", "{input}")
        ])
        
        # Create a chain that combines the retrieved documents with the prompt
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create a retrieval chain that uses the vector store retriever
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity",  # Maximum Marginal Relevance
            search_kwargs={"k": 5, "fetch_k": 10}  # Retrieve 5 documents, from a fetch of 10
        )
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Execute the chain with the user query
        response = retrieval_chain.invoke({
            "input": user_query,
            "patient_name": st.session_state.patient_name
        })
        
        return response["answer"]
    else:
        # Build standard prompt sequence
        prompt_sequence = [SystemMessagePromptTemplate.from_template(standard_system_template)]
        for msg in st.session_state.message_log:
            if msg["role"] == "user":
                prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
            elif msg["role"] == "ai":
                prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
        
        # Add the current user query
        prompt_sequence.append(HumanMessagePromptTemplate.from_template(user_query))
        
        prompt_chain = ChatPromptTemplate.from_messages(prompt_sequence)
        processing_pipeline = prompt_chain | llm | StrOutputParser()
        
        return processing_pipeline.invoke({})

# ---- Processing User Input ----
if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    with st.spinner("🧠 Processing..."):
        with st.chat_message("ai"):
            ai_response_placeholder = st.empty()
            full_response = ""
            for chunk in generate_ai_response(user_query):
                full_response += chunk
                ai_response_placeholder.markdown(full_response + "▌")
            ai_response_placeholder.markdown(full_response)
    
    st.session_state.message_log.append({"role": "ai", "content": full_response})

    st.rerun()