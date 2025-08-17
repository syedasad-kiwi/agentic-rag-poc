import os
from dotenv import load_dotenv
from urllib.parse import urlparse
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from crewai.tools import BaseTool

# Load environment variables to get the database URL
load_dotenv()

class DocumentRetrievalTool(BaseTool):
    name: str = "Document Retrieval Tool"
    description: str = "Retrieves relevant context from a collection of policy and standards documents"

    def _run(self, query: str) -> str:
        """
        Retrieves relevant context from a collection of policy and standards documents.
        
        Args:
            query: The search query to find relevant documents
            
        Returns:
            str: Retrieved context from documents
        """
        try:
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
            retrieved_nodes = retriever.retrieve(query)
            
            if not retrieved_nodes:
                return "No relevant documents found for this query."
            
            # Format the retrieved context for the agent
            context = "\n\n".join([node.get_content() for node in retrieved_nodes])
            
            return context
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"
