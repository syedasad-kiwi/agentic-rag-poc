#!/usr/bin/env python3
"""
Enhanced PGVector Indexing with Contextual RAG
Implements Anthropic's Contextual Retrieval approach using LlamaIndex and Ollama
"""

import os
import sys
import logging
import json
import copy
from datetime import datetime
from urllib.parse import quote_plus
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine, text
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import Document, TextNode
import asyncio
import nest_asyncio

class ContextualRAGIndexer:
    """
    A class to handle enhanced PGVector indexing with Contextual RAG.
    Implements Anthropic's Contextual Retrieval approach using LlamaIndex and Ollama.
    """
    
    def __init__(
        self,
        md_dir: str = "/Users/kiwitech/Documents/agentic-rag-poc/data/processed/md",
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "rag_db",
        db_user: str = "kiwitech",
        db_password: str = "zakirnagar",
        table_name: str = "doc_md_contextual_20250830",
        embed_dim: int = 768,
        context_llm_model: str = "gemma3:4b",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the Contextual RAG Indexer.
        
        Args:
            md_dir: Directory containing markdown documents
            db_host: PostgreSQL host
            db_port: PostgreSQL port
            db_name: Database name
            db_user: Database user
            db_password: Database password
            table_name: Table name for vector storage
            embed_dim: Embedding dimension
            context_llm_model: LLM model for context generation
            ollama_base_url: Ollama service URL
        """
        self.md_dir = md_dir
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.table_name = table_name
        self.embed_dim = embed_dim
        self.context_llm_model = context_llm_model
        self.ollama_base_url = ollama_base_url
        
        # Construct DATABASE_URL
        self.database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        # Setup logging
        self._setup_logging()
        
        # Context prompt template
        self.context_prompt_template = """You are analyzing a procurement policy document. Your task is to provide context for a specific chunk.

<document>
{WHOLE_DOCUMENT}
</document>

<chunk>
{CHUNK_CONTENT}
</chunk>

Provide a brief context (1-2 sentences) explaining:
1. Which section/topic this chunk relates to
2. How it connects to the overall procurement process
3. Its relationship to other procedures

Respond with only the context, nothing else."""
    
    def _setup_logging(self):
        """Configure logging for the indexer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'contextual_rag_index_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def check_database_connection(self) -> bool:
        """Test database connectivity and check for pgvector extension"""
        try:
            self.logger.info(f"Testing connection to database: {self.db_name} at {self.db_host}:{self.db_port}")
            
            engine = create_engine(self.database_url)
            
            with engine.connect() as conn:
                # Test basic connection
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                self.logger.info(f"✅ PostgreSQL connected: {version}")
                
                # Check pgvector extension
                result = conn.execute(text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"))
                has_pgvector = result.fetchone()[0]
                
                if has_pgvector:
                    self.logger.info("✅ pgvector extension is installed")
                else:
                    self.logger.error("❌ pgvector extension not found")
                    return False
                    
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Database connection failed: {e}")
            return False

    def test_ollama_connection(self) -> bool:
        """Test Ollama service connectivity and model availability"""
        try:
            import requests
            
            # Test Ollama service
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                self.logger.error(f"❌ Ollama service not accessible at {self.ollama_base_url}")
                return False
            
            # Check if required models are available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            required_models = [self.context_llm_model, "nomic-embed-text:v1.5"]
            missing_models = [model for model in required_models if model not in model_names]
            
            if missing_models:
                self.logger.error(f"❌ Missing Ollama models: {missing_models}")
                self.logger.info(f"Available models: {model_names}")
                return False
            
            self.logger.info(f"✅ Ollama connected with required models: {required_models}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ollama connection failed: {e}")
            return False

    def clear_existing_table(self) -> bool:
        """Clear existing table data"""
        try:
            self.logger.info(f"Clearing existing data from table: {self.table_name}")
            engine = create_engine(self.database_url)
            
            with engine.connect() as conn:
                # Check if table exists
                result = conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'data_{self.table_name}'
                    )
                """))
                table_exists = result.fetchone()[0]
                
                if table_exists:
                    # Clear the table
                    conn.execute(text(f"DELETE FROM data_{self.table_name}"))
                    conn.commit()
                    self.logger.info(f"✅ Cleared table data_{self.table_name}")
                else:
                    self.logger.info(f"Table data_{self.table_name} doesn't exist yet - will be created")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to clear table: {e}")
            return False

    def _extract_page_number_from_text(self, text: str, chunk_index: int) -> int:
        """Extract page number from text content or estimate based on chunk position"""
        # Simple heuristic: assume ~2000 characters per page
        # You can enhance this by looking for page markers in markdown
        estimated_page = max(1, (chunk_index * 800) // 2000 + 1)  # 800 chars per chunk, 2000 per page
        return estimated_page

    def create_contextual_nodes(self, nodes: List[TextNode], whole_document: str) -> List[TextNode]:
        """Create contextual nodes using Ollama LLM"""
        self.logger.info(f"Creating contextual nodes for {len(nodes)} chunks...")
        
        context_llm = Ollama(
            model=self.context_llm_model,
            base_url=self.ollama_base_url,
            request_timeout=120.0
        )
        
        enhanced_nodes = []
        
        for i, node in enumerate(nodes):
            try:
                enhanced_node = copy.deepcopy(node)
                
                # Generate context using LLM
                context_prompt = self.context_prompt_template.format(
                    WHOLE_DOCUMENT=whole_document[:8000],  # Limit document size for context
                    CHUNK_CONTENT=node.text
                )
                
                # Get context from LLM
                context_response = context_llm.complete(context_prompt)
                context = str(context_response).strip()
                
                # Add context to metadata
                enhanced_node.metadata["context"] = context
                
                # Add page number estimation
                page_num = self._extract_page_number_from_text(node.text, i)
                enhanced_node.metadata["page_number"] = page_num
                
                enhanced_nodes.append(enhanced_node)
                
                # Log progress every 50 nodes
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Generated context for {i + 1}/{len(nodes)} nodes")
                    
            except Exception as e:
                self.logger.warning(f"Failed to generate context for node {i}: {e}")
                # Fallback: use original node with page number
                fallback_node = copy.deepcopy(node)
                fallback_node.metadata["context"] = f"Part of {node.metadata.get('file_name', 'document')}"
                fallback_node.metadata["page_number"] = self._extract_page_number_from_text(node.text, i)
                enhanced_nodes.append(fallback_node)
        
        self.logger.info(f"✅ Created {len(enhanced_nodes)} contextual nodes")
        return enhanced_nodes

    def load_documents(self) -> Optional[List[Document]]:
        """Load markdown documents from the directory"""
        try:
            self.logger.info(f"Loading documents from: {self.md_dir}")
            
            if not os.path.exists(self.md_dir):
                self.logger.error(f"❌ Directory not found: {self.md_dir}")
                return None
            
            # Load documents
            reader = SimpleDirectoryReader(input_dir=self.md_dir, recursive=True)
            documents = reader.load_data()
            
            self.logger.info(f"✅ Loaded {len(documents)} documents")
            
            # Log document info
            for doc in documents:
                file_name = doc.metadata.get('file_name', 'Unknown')
                text_length = len(doc.text)
                self.logger.info(f"Document: {file_name}, Length: {text_length:,} characters")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load documents: {e}")
            return None

    def process_documents(self) -> bool:
        """Main indexing function with contextual RAG"""
        self.logger.info("="*80)
        self.logger.info("Starting Contextual RAG PGVector Indexing Process")
        self.logger.info("="*80)
        
        # Step 1: Check database connection
        if not self.check_database_connection():
            self.logger.error("Aborting: Database connection issues")
            return False
        
        # Step 2: Check Ollama
        if not self.test_ollama_connection():
            self.logger.error("Aborting: Ollama connection issues")
            return False
        
        # Step 3: Clear existing table
        if not self.clear_existing_table():
            self.logger.error("Aborting: Failed to clear existing table")
            return False
        
        # Step 4: Load documents
        documents = self.load_documents()
        if not documents:
            self.logger.error("Aborting: No documents to index")
            return False
    
        try:
            # Step 5: Configure embedding model
            self.logger.info("Configuring embedding model...")
            Settings.embed_model = OllamaEmbedding(
                model_name="nomic-embed-text:v1.5",
                base_url=self.ollama_base_url,
            )
            
            # Step 6: Create vector store
            self.logger.info("Creating vector store...")
            vector_store = PGVectorStore.from_params(
                database=self.db_name,
                host=self.db_host,
                port=self.db_port,
                user=self.db_user,
                password=self.db_password,
                table_name=self.table_name,
                embed_dim=self.embed_dim,
                hybrid_search=True,
                text_search_config="english",
                hnsw_kwargs={
                    "hnsw_m": 16,
                    "hnsw_ef_construction": 64,
                    "hnsw_ef_search": 40,
                    "hnsw_dist_method": "vector_cosine_ops"
                }
            )
            self.logger.info("Vector store created")
            
            # Step 7: Use async parallel processing for optimal performance
            self.logger.info("Setting up async parallel processing for contextual enhancement...")
            
            # Enable async processing
            nest_asyncio.apply()
            
            # Process all documents with optimized async approach
            all_enhanced_nodes = []
            
            async def process_documents_async():
                """Async function to process documents with parallel contextual enhancement"""
                
                # Create ingestion pipeline for standard processing
                pipeline = IngestionPipeline(
                    transformations=[
                        MarkdownNodeParser(include_metadata=True),
                        TokenTextSplitter(
                            chunk_size=800,
                            chunk_overlap=200,
                            separator=" "
                        ),
                        Settings.embed_model,  # Embedding generation
                    ]
                )
                
                all_nodes = []
                
                for doc_idx, document in enumerate(documents):
                    self.logger.info(f"Processing document {doc_idx + 1}/{len(documents)}: {document.metadata.get('file_name', 'Unknown')}")
                    
                    # Run async parallel processing for chunking and embedding
                    processed_nodes = await pipeline.arun(documents=[document], num_workers=4)
                    
                    # Apply contextual enhancement to processed nodes
                    enhanced_nodes = self.create_contextual_nodes(processed_nodes, document.text)
                    all_nodes.extend(enhanced_nodes)
                    
                    self.logger.info(f"Processed document {doc_idx + 1} with {len(enhanced_nodes)} enhanced chunks using async pipeline")
                
                return all_nodes
            
            # Run the async processing
            loop = asyncio.get_event_loop()
            all_enhanced_nodes = loop.run_until_complete(process_documents_async())
            
            # Step 8: Create index and add all enhanced nodes
            self.logger.info(f"Creating vector store index and adding {len(all_enhanced_nodes)} enhanced nodes...")
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=Settings.embed_model,
            )
            
            # Add nodes to the index in batches
            batch_size = 100
            for i in range(0, len(all_enhanced_nodes), batch_size):
                batch = all_enhanced_nodes[i:i + batch_size]
                index.insert_nodes(batch)
                self.logger.info(f"Progress: {min(i + batch_size, len(all_enhanced_nodes))}/{len(all_enhanced_nodes)} nodes inserted")
            
            self.logger.info(f"All {len(all_enhanced_nodes)} enhanced nodes inserted")
            
            # Step 9: Verify indexing
            self.logger.info("Verifying indexed data...")
            engine = create_engine(self.database_url)
            
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM data_{self.table_name}"))
                total_count = result.fetchone()[0]
                self.logger.info(f"Total indexed chunks: {total_count}")
                
                # Check contextual metadata
                result = conn.execute(text(f"""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN metadata_->>'context' IS NOT NULL THEN 1 END) as with_context,
                        COUNT(CASE WHEN metadata_->>'page_number' IS NOT NULL THEN 1 END) as with_page_num
                    FROM data_{self.table_name}
                """))
                stats = result.fetchone()
                self.logger.info(f"✅ Chunks with context: {stats[1]}/{stats[0]}")
                self.logger.info(f"✅ Chunks with page numbers: {stats[2]}/{stats[0]}")
            
            self.logger.info("="*80)
            self.logger.info("CONTEXTUAL RAG INDEXING COMPLETED SUCCESSFULLY!")
            self.logger.info(f"Enhanced {len(all_enhanced_nodes)} chunks with contextual information")
            self.logger.info(f"Table: data_{self.table_name}")
            self.logger.info(f"Context LLM: {self.context_llm_model}")
            self.logger.info("="*80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Indexing failed: {e}")
            return False

def main():
    """Main function to run the contextual RAG indexing process."""
    # Default configuration
    indexer = ContextualRAGIndexer()
    success = indexer.process_documents()
    
    if success:
        print("Contextual RAG indexing completed successfully!")
    else:
        print("Contextual RAG indexing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
