import os
import logging
import psycopg2
import time
from dotenv import load_dotenv
from urllib.parse import urlparse
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.readers.docling import DoclingReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore

# --- 1. Basic Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- 2. Constants and Settings ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

# --- Database Connection Details ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set. Please add it to your .env file.")

db_url_parts = urlparse(DATABASE_URL)
DB_HOST = db_url_parts.hostname
DB_PORT = db_url_parts.port
DB_NAME = db_url_parts.path.lstrip('/')
DB_USER = db_url_parts.username
DB_PASSWORD = db_url_parts.password

# --- Ollama Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = "nomic-embed-text"
EMBEDDING_DIM = 768

# --- Helper Functions for Database Setup ---
def ensure_database_exists():
    """
    Check if we can connect to the target database.
    With Docker Compose setup, the database is created automatically.
    """
    logging.info(f"Checking connection to database '{DB_NAME}' on server {DB_HOST}:{DB_PORT}...")
    try:
        # Try to connect directly to our target database
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()
        logging.info(f"✅ Successfully connected to database '{DB_NAME}'.")
        return True

    except psycopg2.OperationalError as e:
        logging.error(f"❌ Could not connect to the database '{DB_NAME}'.")
        logging.error(f"   Please ensure the Docker container is running and fully initialized.")
        logging.error(f"   Error details: {e}")
        return False
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred while connecting to the database: {e}")
        return False

def check_vector_extension():
    """
    Connects to the TARGET database and ensures the pgvector extension is enabled.
    """
    logging.info(f"Connecting to '{DB_NAME}' to verify pgvector extension...")
    try:
        # Now we connect to our target database using the specific user for that DB.
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            logging.info("✅ pgvector extension is enabled.")
            conn.close()
            return True
        else:
            logging.error("❌ Failed to create or find the pgvector extension in the target database.")
            conn.close()
            return False
    except Exception as e:
        logging.error(f"❌ An error occurred while enabling pgvector extension: {e}")
        return False

# --- Main Ingestion Pipeline ---
def main():
    """
    The main function to run the entire data ingestion pipeline.
    """
    # Give Docker a moment to initialize the database fully on first run.
    logging.info("Waiting 5 seconds for the database container to initialize...")
    time.sleep(5)


    if not ensure_database_exists():
        return
    if not check_vector_extension():
        return

    # --- 3. Document Parsing with Docling ---
    logging.info("Initializing DoclingReader for structure-aware document parsing...")
    reader = DoclingReader(keep_image=False)
    
    all_docs = []
    for filename in os.listdir(RAW_DATA_DIR):
        file_path = os.path.join(RAW_DATA_DIR, filename)
        if os.path.isfile(file_path) and (filename.lower().endswith('.pdf') or filename.lower().endswith('.docx')):
            logging.info(f"Parsing document: {filename}")
            try:
                docs = reader.load_data(file_path=[file_path])
                for doc in docs:
                    doc.metadata['source_file'] = filename
                all_docs.extend(docs)
                logging.info(f"Successfully parsed and loaded {len(docs)} semantic chunks from {filename}.")
            except Exception as e:
                logging.error(f"Failed to parse {filename}. Error: {e}")

    if not all_docs:
        logging.error("No documents were successfully parsed. Aborting.")
        return

    logging.info(f"Total semantic document chunks loaded: {len(all_docs)}")

    # --- 4. Initialize Embedding Model ---
    logging.info(f"Initializing Ollama embedding model: '{EMBEDDING_MODEL_NAME}'...")
    embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL_NAME)

    # --- 5. Set up PGVector Store and Storage Context ---
    logging.info("Setting up PGVectorStore and PostgresDocumentStore...")
    
    vector_store_table_name = "document_embeddings"
    vector_store = PGVectorStore.from_params(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        table_name=vector_store_table_name,
        embed_dim=EMBEDDING_DIM,
    )

    doc_store_table_name = "document_text_store"
    doc_store = PostgresDocumentStore.from_params(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        table_name=doc_store_table_name,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=doc_store
    )

    # --- 6. Indexing: Embedding and Storing ---
    logging.info("Starting the indexing process...")
    
    index = VectorStoreIndex.from_documents(
        all_docs,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    logging.info("✅ Ingestion pipeline completed successfully!")
    logging.info(f"Embeddings and documents are now stored in your database: '{DB_NAME}'.")
    logging.info(f"Vector Table: '{vector_store_table_name}', Document Table: '{doc_store_table_name}'")


if __name__ == "__main__":
    main()
