import os
import sys
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredEmailLoader
from langchain_core.documents import Document

# --- Project Path Setup ---
# Dynamically add the project root and config directory to the Python path.
# This ensures that modules can be imported correctly regardless of where the
# script is executed from.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'config'))
# --- End Project Path Setup ---

# --- Import Application Settings ---
try:
    import config.settings as settings
except ImportError as e:
    print(f"ERROR: {os.path.basename(__file__)}: Failed to import settings module: {e}")
    sys.exit(1)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

# --- DocumentLoader Class ---
class DocumentLoader:
    """
    A class responsible for loading various document types using LangChain's loaders.
    It supports PDFs, DOCX files, TXT files, and EML (email) files.
    """
    def __init__(self):
        """
        Initializes the DocumentLoader instance.
        No special setup is required in the constructor for this class.
        """
        pass

    def load_document(self, file_path: str) -> List[Document]:
        """
        Loads a document from a given file path based on its extension.

        This method uses a file's extension to select the appropriate LangChain
        document loader, which parses the file and returns its content as a
        list of LangChain Document objects.

        Args:
            file_path: The full path to the document file to be loaded.

        Returns:
            A list of `langchain_core.documents.Document` objects, where each
            object typically represents a page or a logical section of the document.
            Returns an empty list if the file type is unsupported or an error occurs.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        loader = None

        # Select the correct LangChain loader based on the file extension.
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_extension == ".eml":
            loader = UnstructuredEmailLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_extension} for {file_path}")
            return []

        # Load the document using the selected loader and handle potential errors.
        if loader:
            logger.info(f"Loading document: {os.path.basename(file_path)}")
            try:
                documents = loader.load()
                logger.info(f"Successfully loaded {len(documents)} pages/sections from {os.path.basename(file_path)}")
                return documents
            except Exception as e:
                logger.error(f"Error loading document {file_path}: {e}", exc_info=True)
                return []
        
        return []