import os
from dotenv import load_dotenv

# --- Load Environment Variables ---
# This line loads variables from a .env file into the script's environment.
# This is a best practice for managing secrets and configuration.
load_dotenv()

# --- Core Application Settings ---
API_TOKEN = os.getenv("API_TOKEN", "66a420132d7ce6f47ba61b022577f789f1dc8dee0ee1cb1a3daf9a89c2505a65")
API_PREFIX = "/hackrx"

# Dynamically determine the base project directory.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- Document Processing and Storage ---
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploaded_documents')
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# --- Vector Store (FAISS) Settings ---
FAISS_INDEX_DIR = os.path.join(BASE_DIR, 'data', 'faiss_index')
FAISS_INDEX_NAME = 'policy_index'
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, FAISS_INDEX_NAME)


# --- Retrieval-Augmented Generation (RAG) Settings ---
# Number of top relevant document chunks to retrieve for the LLM.
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "2"))


# --- LLM Provider Selection ---
# The value for LLM_PROVIDER determines which LLM and embeddings models are used.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google").lower() # Default to 'google'


# --- Google Gemini Settings ---
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_LLM_MODEL: str = os.getenv("GEMINI_LLM_MODEL", "gemini-1.5-flash")
GEMINI_LLM_TEMPERATURE = float(os.getenv("GEMINI_LLM_TEMPERATURE", "0.2"))
GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")


# --- Ollama Settings (for local models) ---
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "phi3:mini")
OLLAMA_LLM_TEMPERATURE = float(os.getenv("OLLAMA_LLM_TEMPERATURE", "0.2"))
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "all-minilm")


# --- OpenAI Settings ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_LLM_TEMPERATURE = float(os.getenv("OPENAI_LLM_TEMPERATURE", "0.2"))
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


# --- Database Settings (for chat history, if used) ---
DATABASE_URL = os.path.join(BASE_DIR, 'chat_history.db')

# --- Directory Initialization ---
# Ensure that all necessary directories are created when the application starts.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)