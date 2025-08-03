import os
import sys
import logging
import json
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

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
    from utils.embedder import DocumentEmbedder

    # Import specific LLM classes based on the configured provider.
    if settings.LLM_PROVIDER == "ollama":
        from langchain_community.llms import Ollama
    elif settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
    elif settings.LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
    else:
        raise ValueError(f"Unsupported LLM provider specified in settings: {settings.LLM_PROVIDER}")
except ImportError as e:
    # Handle missing dependencies gracefully.
    print(f"ERROR: {os.path.basename(__file__)}: Failed to import modules. Check virtual environment and requirements: {e}")
    sys.exit(1)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

# --- Pydantic Model for Structured Answer ---
class StructuredAnswer(BaseModel):
    """
    Represents a structured answer with various details, including confidence and relevance.
    This model enforces a strict JSON schema for the LLM's output.
    """
    answer: str = Field(..., description="The direct answer to the question.")
    extracted_information: Dict[str, Any] = Field(
        default_factory=dict, description="Key-value pairs of extracted entities or facts relevant to the answer."
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="A numerical score (0.0-1.0) indicating the confidence in the answer."
    )
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="A numerical score (0.0-1.0) indicating the relevance of the answer to the question."
    )
    source_documents: List[str] = Field(
        default_factory=list, description="List of source document titles or identifiers from which the answer was derived."
    )
    error: Optional[str] = Field(None, description="Error message if processing failed for this question.")


# --- LLM Interface Class ---
class LLMInterface:
    """
    A class to encapsulate all logic related to interacting with the LLM.
    It manages LLM initialization, prompt engineering, and output parsing.
    """
    def __init__(self, document_embedder: DocumentEmbedder):
        """
        Initializes the LLM and the RAG pipeline components.
        
        Args:
            document_embedder: An instance of DocumentEmbedder for retrieving context.
        """
        self.document_embedder = document_embedder
        
        # Initialize the LLM based on settings.
        if settings.LLM_PROVIDER == "ollama":
            self.llm = Ollama(model=settings.OLLAMA_LLM_MODEL, temperature=settings.OLLAMA_LLM_TEMPERATURE)
            logger.info(f"Initialized Ollama LLM with model: {settings.OLLAMA_LLM_MODEL}")
        elif settings.LLM_PROVIDER == "openai":
            self.llm = ChatOpenAI(model=settings.OPENAI_LLM_MODEL, api_key=settings.OPENAI_API_KEY, temperature=settings.OPENAI_LLM_TEMPERATURE)
            logger.info(f"Initialized OpenAI LLM with model: {settings.OPENAI_LLM_MODEL}")
        elif settings.LLM_PROVIDER == "google":
            self.llm = ChatGoogleGenerativeAI(model=settings.GEMINI_LLM_MODEL, google_api_key=settings.GEMINI_API_KEY, temperature=settings.GEMINI_LLM_TEMPERATURE)
            logger.info(f"Initialized Google Gemini LLM with model: {settings.GEMINI_LLM_MODEL}")
        
        # Define a JSON output parser using the Pydantic model.
        self.parser = JsonOutputParser(pydantic_object=StructuredAnswer)

        # Define the prompt template for the LLM.
        # It includes a system message to guide the LLM's behavior and enforce the JSON schema.
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly skilled AI assistant specializing in extracting and structuring information from legal, insurance, HR, and compliance documents. Your goal is to provide concise answers to user questions and, most importantly, to extract relevant details into a structured JSON format according to the provided Pydantic schema.

            Always ensure your output strictly adheres to the JSON schema. If a piece of information for a field in 'extracted_information' is not found, omit that specific field, do not put 'N/A' or 'None'. Provide only what is directly available.
            
            Strictly follow this JSON format:
            {format_instructions}

            Provide a confidence score (0.0-1.0) based on how certain you are of the answer, a relevance score (0.0-1.0) based on how relevant the source documents were, and list the source document names that directly contributed to the answer.
            If you cannot find an answer or the question is out of context, return an empty answer string (""), 0.0 for both scores, an empty dictionary for 'extracted_information', an empty list for 'source_documents', and set the 'error' field to "No relevant information found or question out of context.".

            Context:
            {context}"""),
            ("user", "Question: {question}")
        ])

        # Create the LLM chain using LangChain Expression Language (LCEL).
        # This chain connects the prompt, the LLM, and the output parser.
        self.chain = self.prompt | self.llm | self.parser
        logger.info("LLM chain (prompt | LLM | parser) initialized.")

    async def get_answer_from_llm(self, question: str, retrieved_chunks: List[Document]) -> Dict[str, Any]:
        """
        Retrieves an answer from the LLM based on the given question and pre-retrieved
        document chunks, parsing the response into a StructuredAnswer Pydantic model.
        
        Args:
            question: The natural language query from the user.
            retrieved_chunks: A list of relevant document chunks (LangChain Document objects).
        
        Returns:
            A dictionary representing the structured answer.
        
        Raises:
            ValidationError: If the LLM's output does not conform to the Pydantic schema.
        """
        try:
            # Combine the content of the retrieved chunks to form the context for the LLM.
            context_content = "\n\n".join([doc.page_content for doc in retrieved_chunks])
            
            # Extract source document titles for the final output.
            source_titles = list(set([doc.metadata.get("title", doc.metadata.get("source", "Unknown Source")) for doc in retrieved_chunks]))
            logger.info(f"Retrieved {len(retrieved_chunks)} document chunks. Source titles: {source_titles}")

            # Invoke the LangChain Expression Language (LCEL) chain.
            # This asynchronously calls the LLM with the prompt and context.
            llm_raw_output_dict = await self.chain.ainvoke({
                "context": context_content,
                "question": question,
                "format_instructions": self.parser.get_format_instructions()
            })

            # Since the parser is part of the chain, `llm_raw_output_dict` is already a dictionary.
            # We add source information here before validating with the Pydantic model.
            llm_raw_output_dict["source_documents"] = source_titles
            
            # Validate the LLM's output against the Pydantic model.
            structured_answer = StructuredAnswer(**llm_raw_output_dict)
            logger.info(f"Structured answer generated for question '{question}'.")
            
            # Return the Pydantic model as a dictionary.
            return structured_answer.model_dump()

        except ValidationError as e:
            # Handle cases where the LLM fails to produce valid JSON.
            logger.error(f"Pydantic validation error for question '{question}'. Raw LLM output did not match schema: {e}", exc_info=True)
            return StructuredAnswer(
                answer="An internal error occurred during answer formatting. The LLM did not provide a valid JSON response.",
                confidence_score=0.0,
                relevance_score=0.0,
                source_documents=[],
                error=f"Pydantic validation failed: {e}"
            ).model_dump()
        except Exception as e:
            # Catch all other unexpected errors during the process.
            logger.error(f"Error in get_answer_from_llm for question '{question}': {e}", exc_info=True)
            return StructuredAnswer(
                answer="An unexpected internal error occurred.",
                confidence_score=0.0,
                relevance_score=0.0,
                source_documents=[],
                error=f"An unexpected error occurred: {e}"
            ).model_dump()