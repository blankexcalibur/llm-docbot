HackRx LLM-Powered Intelligent Query-Retrieval System

This project is an LLM-powered system designed to process large documents and answer complex, domain-specific natural language queries. It leverages a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, explainable, and structured responses from various document types, including PDFs, DOCX files, and emails.
The system is optimized for use in domains such as legal, insurance, HR, and compliance, where contextual understanding and clause-level accuracy are critical.



🚀 Features
Multi-Document Support: Ingests and processes PDF, DOCX, and TXT files.

Dynamic RAG Pipeline: Uses a FAISS vector store to perform semantic search on documents, ensuring the LLM's answers are grounded in the provided context.

Configurable LLM Backends: Easily switch between Google Gemini, OpenAI, or Ollama models by changing a single setting.

Structured Output: All responses are returned in a clean, parsable JSON format, including the answer, extracted information, and source document titles for traceability.

Scalable API: A FastAPI backend provides a robust and production-ready API for document ingestion and query handling.



🛠️ Prerequisites
Python 3.10 or higher
pip (Python package installer)
Git



⚙️ Setup and Installation
Follow these steps to set up the project on your local machine.

1. Clone the Repository
git clone https://github.com/your-username/hackrx-llm-docbot.git
cd hackrx-llm-docbot

2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

3. Install Dependencies
Install all the required Python packages from the requirements.txt file.
pip install -r requirements.txt

4. Configure Environment Variables
Create a .env file in the project's root directory. This file will store your API keys and configuration settings.
# Example .env file content
# --- LLM Provider Settings ---
LLM_PROVIDER="google"  # Options: "google", "openai", "ollama"
# --- Google Gemini Configuration ---
GEMINI_API_KEY="your_google_gemini_api_key"
GEMINI_LLM_MODEL="gemini-1.5-flash"
GEMINI_LLM_TEMPERATURE=0.2
# --- Ollama Configuration (if using a local model) ---
# OLLAMA_BASE_URL="http://localhost:11434"
# OLLAMA_LLM_MODEL="phi3:mini"
# OLLAMA_LLM_TEMPERATURE=0.2
# --- OpenAI Configuration ---
# OPENAI_API_KEY="your_openai_api_key"
# OPENAI_LLM_MODEL="gpt-4o-mini"
# OPENAI_LLM_TEMPERATURE=0.2
# --- Retrieval Settings ---
RETRIEVAL_K=2
# --- API Authentication Token ---
API_TOKEN="66a420132d7ce6f47ba61b022577f789f1dc8dee0ee1cb1a3daf9a89c2505a65"



🚀 Running the Application
Start the FastAPI server using uvicorn. The --reload flag is useful for development as it automatically restarts the server on code changes.

uvicorn app:app --reload

The application will be accessible at http://127.0.0.1:8000. You can view the interactive API documentation (Swagger UI) at http://127.0.0.1:8000/docs.

📋 API Usage
The primary endpoint for interacting with the system is /hackrx/run.

Endpoint: POST /hackrx/run
This endpoint accepts a JSON payload containing a document URL and a list of questions. It downloads the document, indexes it, and then provides structured answers for each question.

Request Body

{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...",
    "questions": [
        "What is the grace period for premium payment?",
        "Are the medical expenses for an organ donor covered?"
    ]
}

Request Headers

Content-Type: application/json

Authorization: Bearer 66a420132d7ce6f47ba61b022577f789f1dc8dee0ee1cb1a3daf9a89c2505a65 (Use the token from your .env file)

Sample Response

{
    "answers": [
        {
            "answer": "A grace period of thirty days is provided for premium payment...",
            "extracted_information": {},
            "confidence_score": 0.95,
            "relevance_score": 0.9,
            "source_documents": ["policy.pdf"],
            "error": null
        },
        {
            "answer": "Yes, the policy indemnifies the medical expenses for the organ donor...",
            "extracted_information": {},
            "confidence_score": 0.88,
            "relevance_score": 0.92,
            "source_documents": ["policy.pdf"],
            "error": null
        }
    ]
}




🧠 System Architecture
The system operates as a classic RAG pipeline, which is orchestrated by the app.py script. The workflow is as follows:

Document Ingestion: A document URL is received. The DocumentLoader downloads and parses the file content.

Indexing: The DocumentEmbedder splits the document into smaller, semantically meaningful chunks. These chunks are then embedded and stored in a FAISS vector store.

Query Handling: The LLMInterface receives a list of natural language questions.

Retrieval: For each question, the DocumentEmbedder performs a semantic search on the FAISS index to retrieve the k most relevant document chunks.

Generation: The LLMInterface combines the original question with the retrieved chunks and sends this augmented prompt to the LLM (e.g., Gemini, OpenAI, or Ollama).

Structured Output: The LLM's response is parsed and validated against a Pydantic model, ensuring a consistent and structured JSON output.