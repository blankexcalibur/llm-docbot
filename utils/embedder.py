import os
import sys
import logging
from typing import List, Dict, Optional, Union

# --- Unconditional imports ---
# These are the core libraries for text splitting and vector storage.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LC_Document

# --- Project Path Setup ---
# Dynamically add the project root and config directory to the Python path.
# This ensures that modules can be imported correctly regardless of where the
# script is executed from.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'config'))
# --- End Project Path Setup ---

# --- Import Application Settings and Utilities ---
try:
    import config.settings as settings
    from utils.document_loader import DocumentLoader
except ImportError as e:
    # Handle missing dependencies gracefully.
    print(f"ERROR: {os.path.basename(__file__)}: Failed to import modules. Check virtual environment and requirements: {e}")
    sys.exit(1)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

# --- DocumentEmbedder Class ---
class DocumentEmbedder:
    """
    A class to manage the document embedding and retrieval process.
    It handles initializing the embedding model, loading or creating the
    FAISS vector store, and processing documents to be added to the index.
    """
    def __init__(self):
        """
        Initializes the document loader, vector store, and the embedding model.
        It also attempts to load an existing FAISS index.
        """
        self.document_loader = DocumentLoader()
        self.vector_store: Optional[FAISS] = None
        self._initialize_embedder()
        self._load_or_create_faiss_index()

    def _initialize_embedder(self):
        """
        Initializes the embedding model based on the provider specified in settings.
        The function dynamically imports the correct embedding class to keep imports clean.
        
        Raises:
            ValueError: If an unsupported LLM provider is specified in the settings.
        """
        if settings.LLM_PROVIDER == "ollama":
            from langchain_community.embeddings import OllamaEmbeddings
            self.embedder = OllamaEmbeddings(
                model=settings.OLLAMA_EMBEDDING_MODEL,
                base_url=settings.OLLAMA_BASE_URL
            )
            logger.info(f"Initialized Ollama embedding model: {settings.OLLAMA_EMBEDDING_MODEL}")
        elif settings.LLM_PROVIDER == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self.embedder = GoogleGenerativeAIEmbeddings(
                model=settings.GEMINI_EMBEDDING_MODEL,
                google_api_key=settings.GEMINI_API_KEY
            )
            logger.info(f"Initialized Google Generative AI embedding model: {settings.GEMINI_EMBEDDING_MODEL}")
        elif settings.LLM_PROVIDER == "openai":
            from langchain_openai import OpenAIEmbeddings
            self.embedder = OpenAIEmbeddings(
                model=settings.OPENAI_EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY
            )
            logger.info(f"Initialized OpenAI embedding model: {settings.OPENAI_EMBEDDING_MODEL}")
        else:
            raise ValueError(f"Unsupported LLM provider for embeddings: {settings.LLM_PROVIDER}")

    def _load_or_create_faiss_index(self):
        """
        Loads an existing FAISS index from the disk if it exists.
        If no index is found, it initializes an empty one and logs the event.
        """
        index_path_faiss = f"{settings.FAISS_INDEX_PATH}.faiss"
        index_path_pkl = f"{settings.FAISS_INDEX_PATH}.pkl"
        
        if os.path.exists(index_path_faiss) and os.path.exists(index_path_pkl):
            logger.info(f"Loading existing FAISS index from {settings.FAISS_INDEX_PATH}...")
            self.vector_store = FAISS.load_local(
                settings.FAISS_INDEX_PATH,
                self.embedder,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS index loaded successfully.")
        else:
            logger.info("No existing FAISS index found. An empty index will be initialized.")
            # Create a temporary single-entry index to enable adding documents later.
            self.vector_store = FAISS.from_texts([""], self.embedder)
            # Remove the temporary entry.
            self.vector_store.delete([self.vector_store.index_to_docstore_id[0]])
            logger.info("Empty FAISS index initialized.")

    def process_and_add_document(self, file_path: str) -> None:
        """
        Loads a document, splits it into chunks, and adds it to the FAISS index.
        This method orchestrates the entire ingestion pipeline for a single file.

        Args:
            file_path: The full path to the document file on the local file system.
        """
        logger.info(f"Loading document: {os.path.basename(file_path)}")
        pages = self.document_loader.load_document(file_path)
        
        if not pages:
            logger.warning(f"No content loaded from {file_path}. Skipping chunking and indexing.")
            return

        logger.info(f"Loaded {len(pages)} pages/sections from {os.path.basename(file_path)}")

        # Initialize the text splitter.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(f"Splitting document into chunks (size: {settings.CHUNK_SIZE}, overlap: {settings.CHUNK_OVERLAP})....")
        chunks = text_splitter.split_documents(pages)
        logger.info(f"Created {len(chunks)} chunks from {os.path.basename(file_path)}")
        
        # Add metadata to each chunk for better source tracking.
        for chunk in chunks:
            chunk.metadata['source'] = os.path.basename(file_path)
            chunk.metadata['title'] = os.path.basename(file_path).replace('.pdf', '').replace('.docx', '').replace('.txt', '')

        if self.vector_store:
            logger.info(f"Adding {len(chunks)} chunks to existing FAISS index.")
            self.vector_store.add_documents(chunks)
        else:
            logger.info(f"Creating new FAISS index with {len(chunks)} chunks.")
            self.vector_store = FAISS.from_documents(chunks, self.embedder)
        
        logger.info("Documents added to FAISS index.")
        self.vector_store.save_local(settings.FAISS_INDEX_PATH)
        logger.info("FAISS index saved.")

    def get_relevant_documents(self, query: str) -> List[LC_Document]:
        """
        Retrieves the most relevant document chunks from the FAISS index based on a query.
        
        Args:
            query: The natural language query from the user.
        
        Returns:
            A list of `LangChain_Document` objects representing the most relevant document chunks.
        """
        if not self.vector_store:
            logger.warning("FAISS index not initialized. Cannot retrieve documents.")
            return []
        
        # Ensure the embedder is initialized before performing a search.
        if not hasattr(self, 'embedder') or self.embedder is None:
            self._initialize_embedder()

        logger.info(f"Retrieving top {settings.RETRIEVAL_K} documents for query: '{query}'")
        docs = self.vector_store.similarity_search(query, k=settings.RETRIEVAL_K)
        logger.info(f"Retrieved {len(docs)} documents.")
        return docs