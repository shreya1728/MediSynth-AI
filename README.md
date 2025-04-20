🩺 MediSynth AI
MediSynth AI is a Context-Aware Medical Intelligence Platform designed to assist healthcare professionals in analyzing patient medical history, interpreting lab reports, and generating diagnostic insights. It integrates document processing, vector embeddings, and powerful LLMs (local and cloud-based) to offer reliable, responsive, and secure AI-assisted medical support.

🚀 Key Features

🏥 Medical History Assistant
📄 Upload and process medical records (PDF)
🔍 Embedding-based contextual retrieval using FAISS
💬 Chat interface with memory for seamless interactions
🤖 Multi-model support: Gemini (Google) & Local Ollama LLMs (LLaMA, DeepSeek, Qwen, etc.)
🧠 Context-aware response generation
🛠 Configurable vector embedding backends: Gemini or Nomic


🧪 Lab Report Analyzer

📥 PDF Upload: Upload any lab report in PDF format.
🔍 Text Extraction & Parsing: Extracts test name, value, unit, and reference ranges using regex.
📊 Interactive Visualization: Color-coded bar chart showing elevated, deficient, and normal values.
🧾 Summary Tables: Tables for excess and deficiency markers.
🤖 Medical Assistant Chatbot: Powered by Gemini, explains abnormal values and offers insights.



🧭 How It Works
Medical Record Analysis
1) Upload Document: Upload a patient’s medical record in PDF format.

2) Chunking & Embedding: The document is chunked and embedded using the selected embedding model (Gemini or Nomic).

3) Vector Store: FAISS stores and retrieves chunks using semantic similarity.

4) LLM Interaction:
With context → Retrieval-Augmented Generation (RAG)
Without context → Chat interaction

5) UI: Responsive Streamlit interface with sidebar and chat.


Lab Report Analyzer
1) Upload Lab Report (PDF): Extracts test results using regex.

2) Parse & Visualize: Parses test names, values, units, and reference ranges.

3) Bar Chart Visualization: Uses color to indicate elevated, deficient, or normal values.

4) Abnormal Value Summary: View separate tables for high and low results.

5) RAG LLM Model