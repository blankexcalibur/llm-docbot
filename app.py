import os
import sys
import logging
import httpx
import tempfile
import shutil
import asyncio
import google.api_core.exceptions
import time
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks, Depends, status
from pydantic import BaseModel, Field, HttpUrl
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# --- Project Path Setup ---
# Dynamically add the project root and config directory to the Python path.
# This ensures that modules can be imported correctly regardless of where the
# script is executed from.
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'config'))
# --- End Project Path Setup ---

# Import application settings and utility modules.
# The try-except block gracefully handles cases where a dependency is missing.
try:
    import config.settings as settings
    from utils.document_loader import DocumentLoader
    from utils.embedder import DocumentEmbedder
    from utils.llm_interface import LLMInterface, StructuredAnswer
except ImportError as e:
    # If a core module or dependency is missing, log the error and exit.
    print(f"ERROR: Failed to import local modules. Check your virtual environment and requirements.txt: {e}")
    sys.exit(1)

# --- Logging Configuration ---
# Configure basic logging to output INFO level messages to the console.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

# --- FastAPI Application Setup ---
# Main FastAPI application instance with metadata for API documentation.
app = FastAPI(
    title="HackRx LLM Document Analyzer",
    description="API for intelligent document query and retrieval for insurance, legal, HR, and compliance domains.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
# --- End FastAPI Application Setup ---

# --- API Security ---
# Define an HTTP Bearer token scheme for API key authentication.
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verifies the provided API token against the configured secret.
    
    Args:
        credentials: The Bearer token from the Authorization header.
    
    Raises:
        HTTPException: If the token is invalid or missing.
    """
    if credentials.credentials != settings.API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API token")
    return credentials.credentials
# --- End API Security ---

# --- Global Service Instances ---
# Instantiate core service classes once to be reused across all requests.
# This is a key FastAPI design pattern for dependency injection and performance.
document_embedder_instance = DocumentEmbedder()
llm_interface_instance = LLMInterface(document_embedder=document_embedder_instance)
# --- End Global Service Instances ---

# --- API Endpoints ---

@app.post("/upload/document", summary="Upload and process a document for RAG pipeline")
async def upload_document(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """
    Accepts a document file, saves it, and processes its contents for the RAG pipeline.
    The file is loaded, split into chunks, and its embeddings are added to a FAISS index.

    Args:
        file: The document file to upload (e.g., PDF, DOCX).
        token: The API token for authentication.

    Raises:
        HTTPException: If the file is missing, the type is not allowed, or an error occurs during processing.
    
    Returns:
        A JSON message confirming the successful processing of the document.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded.")

    file_extension = os.path.splitext(file.filename)[1].lstrip('.').lower()
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        allowed_types = ', '.join(settings.ALLOWED_EXTENSIONS)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"File type '{file_extension}' not allowed. Allowed types: {allowed_types}")

    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_FOLDER, file.filename)

    try:
        # Save the uploaded file to disk.
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved to: {file_path}")

        # Process the document by loading, chunking, and indexing its content.
        document_embedder_instance.process_and_add_document(file_path)

        return {"message": f"Document '{file.filename}' processed and indexed successfully."}
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process document: {e}")
    finally:
        await file.close()

@app.post("/chat", summary="Ask a question about the uploaded documents")
async def chat_with_docs(
    query: Dict[str, str],
    token: str = Depends(verify_token)
):
    """
    Answers a natural language query based on the content of the indexed documents.
    This endpoint does not maintain chat history and performs a fresh retrieval for each query.

    Args:
        query: A JSON object containing the user's question, e.g., `{"query": "What are the policy benefits?"}`.
        token: The API token for authentication.

    Raises:
        HTTPException: If the query is missing or an error occurs during retrieval or LLM generation.

    Returns:
        A structured JSON response with the answer and supporting details.
    """
    question = query.get("query")
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query parameter is missing.")

    try:
        # Retrieve relevant document chunks and get the LLM's answer.
        retrieved_chunks = document_embedder_instance.get_relevant_documents(question)
        answer_dict = await llm_interface_instance.get_answer_from_llm(question, retrieved_chunks)
        return answer_dict
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting answer for query '{question}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {e}")

# Pydantic models for the /run endpoint's request and response bodies.
class RunRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL of the document to process.")
    questions: List[str] = Field(..., description="A list of natural language questions.")

class SubmissionRunResponse(BaseModel):
    answers: List[str]


@app.post(
    settings.API_PREFIX + "/run",
    response_model=SubmissionRunResponse,
    summary="Process document from URL and answer a list of questions"
)
async def run_query_retrieval(
    request_body: RunRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Performs a full RAG pipeline run: downloads a document from a URL,
    processes it, and answers a list of questions in a single request.
    The temporary file storage is cleaned up automatically after the request.

    Args:
        request_body: A JSON object containing the document URL and a list of questions.
        background_tasks: FastAPI's dependency for running cleanup tasks in the background.
        token: The API token for authentication.

    Raises:
        HTTPException: If an error occurs during downloading, processing, or LLM generation.

    Returns:
        A JSON object containing a list of answers for each question.
    """
    start_total_time = time.time()
    document_url = str(request_body.documents)
    questions = request_body.questions
    temp_dir = tempfile.mkdtemp()
    
    # Schedule the temporary directory to be removed after the response is sent.
    background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)

    try:
        # --- Stage 1: Document Download ---
        logger.info(f"Attempting to download document from URL: {document_url}")
        download_start_time = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.get(document_url, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
        logger.info(f"Document download took: {time.time() - download_start_time:.2f} seconds.")

        # Extract filename from URL or guess based on content type.
        filename = os.path.basename(document_url.split('?')[0])
        if not filename or '.' not in filename:
            content_type = response.headers.get("Content-Type", "")
            if "pdf" in content_type:
                filename = "downloaded_document.pdf"
            elif "word" in content_type or "officedocument" in content_type:
                filename = "downloaded_document.docx"
            else:
                filename = "downloaded_document.bin"
            logger.warning(f"Could not extract filename from URL, using default: {filename}")

        temp_file_path = os.path.join(temp_dir, filename)
        with open(temp_file_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Document downloaded to: {temp_file_path}")

        # --- Stage 2: Document Indexing ---
        # Process the downloaded document and add its chunks to the vector store.
        indexing_start_time = time.time()
        document_embedder_instance.process_and_add_document(temp_file_path)
        logger.info(f"Document indexing took: {time.time() - indexing_start_time:.2f} seconds.")
        logger.info("Document indexed successfully.")
        
        # --- Stage 3: Answering Questions Concurrently ---
        # Create an asynchronous task for each question to be answered by the LLM.
        # This speeds up the process significantly by making parallel API calls.
        answering_start_time = time.time()
        tasks = []
        for q in questions:
            # First, retrieve the most relevant chunks for the question.
            retrieved_chunks = document_embedder_instance.get_relevant_documents(q)
            # Then, create a task to get the answer using the retrieved context.
            tasks.append(llm_interface_instance.get_answer_from_llm(q, retrieved_chunks))
        
        # Run all the LLM tasks concurrently and wait for them to complete.
        all_answers_dicts = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Answering {len(questions)} questions took: {time.time() - answering_start_time:.2f} seconds.")

        # Process the results, handling potential exceptions from the LLM calls.
        final_answers_list = []
        for result in all_answers_dicts:
            if isinstance(result, Exception):
                logger.error(f"Error in LLM call: {result}")
                final_answers_list.append("Could not retrieve an answer due to an API error.")
            else:
                final_answers_list.append(result.get('answer', "Could not parse answer."))

        total_time = time.time() - start_total_time
        logger.info(f"Total processing time for all questions (including download, indexing): {total_time:.2f} seconds.")
        
        return {"answers": final_answers_list}

    except google.api_core.exceptions.ResourceExhausted as e:
        # Handle API quota errors specifically.
        logger.error(f"API quota exceeded: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="The daily API quota has been exceeded. Please try again tomorrow or contact support for a plan upgrade."
        )
    except httpx.HTTPStatusError as e:
        # Handle HTTP errors during the download phase.
        logger.error(f"HTTP error downloading document: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise HTTPException(status_code=e.response.status_code, detail=f"Download error: {e.response.text[:200]}...")
    except httpx.RequestError as e:
        # Handle network-related errors.
        logger.error(f"Network error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Network error: {e}")
    except Exception as e:
        # Catch any other unexpected errors.
        logger.error(f"Unexpected error during processing: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error: {e}")