import os
from dotenv import load_dotenv
from urllib.parse import urlparse
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, validator
from typing import Type, Union, Any

# Load environment variables to get the database URL
load_dotenv()

class DocumentRetrievalInput(BaseModel):
    """Input schema for document retrieval tool."""
    query: str = Field(..., description="The search query to find relevant documents")
    
    @validator('query', pre=True)
    def extract_query(cls, v: Union[str, dict, Any]) -> str:
        """Extract query string from various input formats."""
        if isinstance(v, str):
            return v
        elif isinstance(v, dict):
            # Handle CrewAI's format: {'description': 'query text'}
            if 'description' in v:
                return v['description']
            # Handle other dict formats
            return str(v.get('query', v))
        else:
            return str(v)

class DocumentRetrievalTool(BaseTool):
    name: str = "Document Retrieval Tool"
    description: str = "Retrieves relevant context from a collection of policy and standards documents"
    args_schema: Type[BaseModel] = DocumentRetrievalInput

    def _run(self, query: str = None, description: str = None, **kwargs) -> str:
        """
        Retrieves relevant context from a collection of policy and standards documents.
        
        Args:
            query: The search query to find relevant documents
            description: Alternative parameter name that CrewAI might use
            
        Returns:
            str: Retrieved context from documents
        """
        try:
            # Handle different parameter names that CrewAI might pass
            search_query = query or description or kwargs.get('query') or kwargs.get('description')
            
            if not search_query:
                return "Error: No search query provided."
                
            DATABASE_URL = os.getenv("DATABASE_URL")
            if not DATABASE_URL:
                return "Error: DATABASE_URL environment variable not set."

            db_url_parts = urlparse(DATABASE_URL)
            
            # Initialize the vector store with connection details
            vector_store = PGVectorStore.from_params(
                host=db_url_parts.hostname,
                port=db_url_parts.port,
                database=db_url_parts.path.lstrip('/'),
                user=db_url_parts.username,
                password=db_url_parts.password,
                table_name="document_embeddings",
                embed_dim=768, # Dimension for nomic-embed-text
            )

            # Initialize the embedding model
            embed_model = OllamaEmbedding(model_name="nomic-embed-text")

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

# Create an instance of the tool for use in agents
document_retrieval_tool = DocumentRetrievalTool()