import os
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from crewai.tools import tool
from typing import Dict, Union, Any, Optional


class DocumentRetrievalService:
    """
    A service class for handling document retrieval operations.
    Encapsulates vector store operations and document formatting logic.
    """
    
    def __init__(self):
        """Initialize the document retrieval service."""
        # Load environment variables
        load_dotenv()
        
        # Configuration
        self.contextual_table = "doc_md_contextual_20250830"
        self.regular_table = "doc_md_20250830_015134"
        self.embed_dim = 768
        self.embedding_model_name = "nomic-embed-text:v1.5"
        self.similarity_top_k = 2
        self.sparse_top_k = 2
        self.request_timeout = 120.0
        
        # Get configuration from environment
        self.database_url = os.getenv("DATABASE_URL")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    def _warm_up_ollama(self, base_url: str, model_name: str) -> bool:
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

    def _extract_search_query(self, query: Union[str, Dict[str, Any]]) -> Optional[str]:
        """Extract and validate search query from CrewAI parameter passing."""
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
            return None
        
        # Check if we got a placeholder description instead of real query
        if search_query in ["The search query to find relevant documents", ""]:
            return None
        
        search_query = search_query.strip()
        print(f"DEBUG: Extracted search query: {repr(search_query)}")
        return search_query
    
    def _get_database_connection_params(self) -> Optional[Dict[str, Any]]:
        """Parse database URL and return connection parameters."""
        if not self.database_url:
            return None

        db_url_parts = urlparse(self.database_url)

        # Debug print parsed connection details (password masked)
        print(f"DEBUG: Parsed Postgres connection details - "
              f"host: {db_url_parts.hostname}, "
              f"port: {db_url_parts.port}, "
              f"database: {db_url_parts.path.lstrip('/')}, "
              f"user: {db_url_parts.username}, "
              f"password: {'******' if db_url_parts.password else None}")
        
        return {
            'host': db_url_parts.hostname,
            'port': db_url_parts.port,
            'database': db_url_parts.path.lstrip('/'),
            'user': db_url_parts.username,
            'password': db_url_parts.password
        }
    
    def _create_vector_store(self, db_params: Dict[str, Any]) -> Optional[PGVectorStore]:
        """Create vector store with fallback from contextual to regular table."""
        # Try contextual table first
        try:
            vector_store = PGVectorStore.from_params(
                host=db_params['host'],
                port=db_params['port'],
                database=db_params['database'],
                user=db_params['user'],
                password=db_params['password'],
                table_name=self.contextual_table,
                embed_dim=self.embed_dim,
                hybrid_search=True,
                text_search_config="english",
            )
            print(f"DEBUG: Using contextual table: {self.contextual_table}")
            return vector_store
        except Exception as e:
            print(f"DEBUG: Contextual table not available, using regular table: {e}")
            try:
                vector_store = PGVectorStore.from_params(
                    host=db_params['host'],
                    port=db_params['port'],
                    database=db_params['database'],
                    user=db_params['user'],
                    password=db_params['password'],
                    table_name=self.regular_table,
                    embed_dim=self.embed_dim,
                    hybrid_search=True,
                    text_search_config="english",
                )
                return vector_store
            except Exception as e2:
                print(f"ERROR: Failed to create vector store: {e2}")
                return None
    
    def _create_embedding_model(self) -> OllamaEmbedding:
        """Create and configure the embedding model."""
        # Pre-warm the Ollama model to avoid cold start delays
        self._warm_up_ollama(self.ollama_base_url, self.embedding_model_name)
        
        return OllamaEmbedding(
            model_name=self.embedding_model_name,
            base_url=self.ollama_base_url,
            request_timeout=self.request_timeout
        )
    
    def _format_retrieved_nodes(self, retrieved_nodes) -> str:
        """Format the retrieved context with source metadata and contextual information."""
        if not retrieved_nodes:
            return "No relevant documents found for this query."
        
        formatted_chunks = []
        for i, node in enumerate(retrieved_nodes, 1):
            content = node.get_content()
            
            # Extract source file information from metadata
            source_info = "Unknown source"
            context_info = ""
            page_info = ""
            
            if hasattr(node, 'metadata') and node.metadata:
                # File information
                file_name = node.metadata.get('source_file', node.metadata.get('file_name', 'Unknown file'))
                file_path = node.metadata.get('file_path', '')
                if file_path:
                    source_info = f"Source: {os.path.basename(file_path)}"
                else:
                    source_info = f"Source: {file_name}"
                
                # Contextual information (if available)
                context = node.metadata.get('context', '')
                if context:
                    context_info = f"\nContext: {context}"
                
                # Page number information (if available)
                page_num = node.metadata.get('page_number', '')
                if page_num:
                    page_info = f" (Page {page_num})"
            
            formatted_chunk = f"**Document Chunk {i}**\n{source_info}{page_info}{context_info}\n\nContent:\n{content}"
            formatted_chunks.append(formatted_chunk)
        
        return "\n\n" + "="*50 + "\n\n".join(formatted_chunks)
    
    def retrieve_documents(self, query: Union[str, Dict[str, Any]]) -> str:
        """
        Main document retrieval method.
        Retrieves relevant context from a collection of policy and standards documents.
        """
        try:
            # Extract and validate search query
            search_query = self._extract_search_query(query)
            if not search_query:
                return "Error: No valid search query provided."
            
            # Get database connection parameters
            db_params = self._get_database_connection_params()
            if not db_params:
                return "Error: DATABASE_URL environment variable not set."
            
            # Create vector store
            vector_store = self._create_vector_store(db_params)
            if not vector_store:
                return "Error: Failed to create vector store connection."
            
            # Create embedding model
            embed_model = self._create_embedding_model()
            
            # Create a LlamaIndex VectorStoreIndex object from the vector store
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed_model
            )

            # Create a query engine with hybrid search mode
            query_engine = index.as_query_engine(
                vector_store_query_mode="hybrid",
                similarity_top_k=self.similarity_top_k,
                sparse_top_k=self.sparse_top_k
            )
            
            # Query using hybrid search (combines vector + text search)
            response = query_engine.query(search_query)
            retrieved_nodes = response.source_nodes
            
            # Format and return the results
            return self._format_retrieved_nodes(retrieved_nodes)
            
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"


# Create a global instance of the service
_document_service = DocumentRetrievalService()

@tool("Document Retrieval Tool")
def document_retrieval_tool(query: Union[str, Dict[str, Any]]) -> str:
    """Retrieves relevant context from a collection of policy and standards documents. Use this tool to search for information in policy documents, manuals, and standards."""
    return _document_service.retrieve_documents(query)
