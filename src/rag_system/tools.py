import os
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from crewai.tools import tool
from typing import Dict, Union, Any


# Load environment variables to get the database URL
load_dotenv()

def warm_up_ollama(base_url: str, model_name: str) -> bool:
    """Pre-warm Ollama model to avoid cold start delays"""
    try:
        # Send a small embedding request to warm up the model
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model_name, "prompt": "test"},
            timeout=30
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Warning: Could not warm up Ollama model: {e}")
        return False

@tool("Document Retrieval Tool")  
def document_retrieval_tool(query: Union[str, Dict[str, Any]]) -> str:
    """Retrieves relevant context from a collection of policy and standards documents. Use this tool to search for information in policy documents, manuals, and standards."""
    try:
        # Debug: Log what we actually receive
        print(f"DEBUG: Tool received query parameter: {repr(query)}")
        
        # Handle CrewAI's parameter passing - extract actual query from different formats
        search_query = None
        
        if isinstance(query, str):
            search_query = query
        elif isinstance(query, dict):
            # CrewAI passes: {"description": "actual query", "type": "str"}
            if "description" in query:
                search_query = query["description"]
            elif "query" in query:
                search_query = query["query"]
            else:
                # Fallback: convert dict to string
                search_query = str(query)
        else:
            search_query = str(query)
        
        # Validate we have a proper query string
        if not search_query or not isinstance(search_query, str):
            return "Error: No valid search query provided."
        
        # Check if we got a placeholder description instead of real query
        if search_query in ["The search query to find relevant documents", ""]:
            return "Error: Tool received schema placeholder instead of actual query."
        
        search_query = search_query.strip()
        print(f"DEBUG: Extracted search query: {repr(search_query)}")
        
        DATABASE_URL = os.getenv("DATABASE_URL")
        if not DATABASE_URL:
            return "Error: DATABASE_URL environment variable not set."

        db_url_parts = urlparse(DATABASE_URL)

        # Debug print parsed connection details (password masked)
        print(f"DEBUG: Parsed Postgres connection details - "
              f"host: {db_url_parts.hostname}, "
              f"port: {db_url_parts.port}, "
              f"database: {db_url_parts.path.lstrip('/')}, "
              f"user: {db_url_parts.username}, "
              f"password: {'******' if db_url_parts.password else None}")
        
        # Initialize the vector store with connection details
        vector_store = PGVectorStore.from_params(
            host=db_url_parts.hostname,
            port=db_url_parts.port,
            database=db_url_parts.path.lstrip('/'),
            user=db_url_parts.username,
            password=db_url_parts.password,
            table_name="document_embeddings",  # Use the actual table name
            embed_dim=768, # Dimension for nomic-embed-text
        )

        # Initialize the embedding model - use environment variable for base URL to support Docker
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Pre-warm the Ollama model to avoid cold start delays
        warm_up_ollama(ollama_base_url, "nomic-embed-text:v1.5")
        
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text:v1.5",
            base_url=ollama_base_url,
            request_timeout=120.0  # Increased timeout for slower connections
        )

        # Create a LlamaIndex VectorStoreIndex object from the vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )

        # Create a retriever to query the index
        retriever = index.as_retriever(similarity_top_k=5)
        
        # Retrieve nodes (chunks) relevant to the query
        retrieved_nodes = retriever.retrieve(search_query)
        
        if not retrieved_nodes:
            return "No relevant documents found for this query."
        
        # Format the retrieved context with source metadata
        formatted_chunks = []
        for i, node in enumerate(retrieved_nodes, 1):
            content = node.get_content()
            
            # Extract source file information from metadata
            source_info = "Unknown source"
            if hasattr(node, 'metadata') and node.metadata:
                file_name = node.metadata.get('source_file', node.metadata.get('file_name', 'Unknown file'))
                file_path = node.metadata.get('file_path', '')
                if file_path:
                    # Extract just the filename from the full path
                    source_info = f"Source: {os.path.basename(file_path)}"
                else:
                    source_info = f"Source: {file_name}"
            
            formatted_chunk = f"**Document Chunk {i}**\n{source_info}\n\nContent:\n{content}"
            formatted_chunks.append(formatted_chunk)
        
        context = "\n\n" + "="*50 + "\n\n".join(formatted_chunks)
        
        return context
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"
