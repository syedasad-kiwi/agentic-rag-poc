# Agentic RAG POC

A proof-of-concept implementation of an agentic Retrieval-Augmented Generation (RAG) system using CrewAI, LlamaIndex, and PostgreSQL with pgvector for intelligent document analysis and question answering.

## 🚀 Features

- **Intelligent Document Processing**: Automatically parses and chunks PDF and DOCX documents using DoclingReader with AI-generated context
- **Vector Search**: PostgreSQL with pgvector extension for efficient similarity search
- **Agentic Workflow**: CrewAI agents that can retrieve, analyze, and synthesize information
- **Source Attribution**: Responses include source file references for transparency
- **Graceful Failure Handling**: System responds appropriately when no relevant information is found
- **Local LLM Integration**: Uses Ollama for embedding and language model inference
- **Web Interface**: OpenWebUI integration for user-friendly chat interface
- **Docker Support**: Full containerization with Docker Compose for easy deployment
- **OpenAI-Compatible API**: FastAPI server with OpenAI-compatible endpoints

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   PostgreSQL    │    │   CrewAI        │
│   Ingestion     │───▶│   + pgvector    │───▶│   Agents        │
│   (DoclingReader)│    │   Vector Store  │    │   (Ollama LLM)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Documents │    │   Embeddings    │    │   FastAPI       │
│   (PDF/DOCX)    │    │   & Metadata    │    │   Server        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OpenWebUI     │───▶│   OpenAI API    │───▶│   Intelligent   │
│   Frontend      │    │   Compatible    │    │   Responses     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **Docker & Docker Compose** (Recommended - for complete containerized setup)
- **Ollama** (Required - for LLM and embeddings)
- Python 3.8+ (for local development)
- PostgreSQL 14+ with pgvector extension (if not using Docker)

## 🛠️ Installation & Setup

### Option 1: Docker Setup (Recommended)

This is the complete containerized setup that includes the RAG API, OpenWebUI interface, and networking.

#### 1. Clone the Repository

```bash
git clone https://github.com/syedasad-kiwi/agentic-rag-poc.git
cd agentic-rag-poc
```

#### 2. Install and Setup Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull gemma2:9b          # Main LLM for responses
ollama pull nomic-embed-text   # Embedding model
```

#### 3. Setup PostgreSQL with Docker

```bash
# Start PostgreSQL with pgvector
docker-compose up -d

# Verify PostgreSQL is running
docker ps | grep postgres
```

#### 4. Configure Environment

Create environment files:

```bash
# Create .env file for local development
cat > .env << EOF
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db
OLLAMA_BASE_URL=http://localhost:11434
EOF

# Create .env.docker for container environment  
cat > .env.docker << EOF
DATABASE_URL=postgresql://postgres:password@host.docker.internal:5432/rag_db
OLLAMA_BASE_URL=http://host.docker.internal:11434
EOF
```

#### 5. Document Ingestion

```bash
# Install Python dependencies locally for ingestion
pip install -r requirements.txt

# Place your documents in data/raw/ and run ingestion
python src/data_ingestion/ingest.py
```

#### 6. Build and Deploy RAG API

```bash
# Build the RAG API Docker image
docker build -t rag-api .

# Create Docker network for service communication
docker network create rag-network

# Run RAG API container
docker run --name rag-api -d \
  --network rag-network \
  -p 8000:8000 \
  --env-file .env.docker \
  rag-api
```

#### 7. Deploy OpenWebUI

```bash
# Run OpenWebUI container
docker run --name open-webui -d \
  --network rag-network \
  -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

#### 8. Configure OpenWebUI

1. Open your browser to `http://localhost:3000`
2. Complete the initial setup (create admin account)
3. Go to **Settings** → **Connections** → **OpenAI API**
4. Set the **API Base URL** to: `http://rag-api:8000/v1`
5. Leave the **API Key** field empty
6. Click **Save**
7. The RAG models should now appear in the model dropdown

### Option 2: Local Development Setup
For development without Docker:

#### 1. Python Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Set Up PostgreSQL with pgvector

##### Option A: Using Docker (Recommended)

```bash
docker-compose up -d
```

##### Option B: Local Installation

1. Install PostgreSQL 14+
2. Install pgvector extension
3. Create database and enable extension

#### 3. Set Up Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull gemma2:9b
ollama pull nomic-embed-text
```

#### 4. Environment Configuration

Create a `.env` file in the root directory:

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db
OLLAMA_BASE_URL=http://localhost:11434
```

## 📊 Usage

### Using the Web Interface (Recommended)

1. **Access OpenWebUI**: Navigate to `http://localhost:3000`
2. **Select Model**: Choose `rag-model` or `crew-ai-rag` from the dropdown
3. **Ask Questions**: Type your questions about the ingested documents
4. **View Responses**: Get intelligent responses with source attributions

### Using the API Directly

```bash
# Test the models endpoint
curl http://localhost:8000/v1/models

# Send a chat completion request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag-model",
    "messages": [
      {"role": "user", "content": "What are the key procurement requirements?"}
    ]
  }'
```

### Command Line Usage

```bash
python main.py
```

### Programmatic Usage

```python
from src.rag_system.crew import create_rag_crew

# Create and run the crew
crew = create_rag_crew('What are the key procurement requirements?')
result = crew.kickoff()
print(result)
```

## 🔧 Document Ingestion

The system includes an enhanced ingestion pipeline with AI-generated context:

### 1. Place Documents

Place your documents (PDF/DOCX) in the `data/raw/` directory.

### 2. Run Ingestion

```bash
python src/data_ingestion/ingest.py
```

### Features:
- **DoclingReader**: Advanced document parsing
- **Sentence Splitter**: Intelligent text chunking  
- **AI Context Generation**: Each chunk gets 2-line context using Gemma LLM
- **Vector Storage**: Embeddings stored in PostgreSQL with pgvector

## � Docker Commands Reference

### Container Management

```bash
# View running containers
docker ps

# View container logs
docker logs rag-api
docker logs open-webui

# Restart containers
docker restart rag-api
docker restart open-webui

# Stop and remove containers
docker stop rag-api open-webui
docker rm rag-api open-webui

# Rebuild RAG API
docker build -t rag-api .
docker run --name rag-api -d --network rag-network -p 8000:8000 --env-file .env.docker rag-api
```

### Network Management

```bash
# Create network
docker network create rag-network

# View networks
docker network ls

# Inspect network
docker network inspect rag-network
```

### Troubleshooting

```bash
# Check if services are running on correct ports
lsof -i :8000  # RAG API
lsof -i :3000  # OpenWebUI
lsof -i :5432  # PostgreSQL

# Test API connectivity
curl http://localhost:8000/v1/models

# View detailed container logs
docker logs -f rag-api
```

## 🔍 Key Features Explained

### Enhanced Document Processing

The system now includes AI-generated context for each document chunk:

```python
# Each chunk gets contextual information
chunk_text = "Procurement standards require..."
ai_context = "This section outlines mandatory procurement compliance requirements for all departments."
```

### OpenAI-Compatible API

FastAPI server with OpenAI-compatible endpoints:

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completion endpoint

### Source Attribution

The system automatically extracts and includes source file information in responses:

```python
# Example response format
"Based on the Abu Dhabi Procurement Standards document..."

Sources:
- Abu Dhabi Procurement Standards.PDF
- Procurement Manual (Business Process).PDF
```

### CORS Support

Full CORS support for web interface integration:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📁 Project Structure

```
agentic-rag-poc/
├── data/
│   ├── raw/                 # Input documents (PDF/DOCX)
│   └── processed/           # Processed data
├── src/
│   ├── config/             # Configuration settings
│   ├── data_ingestion/     # Document parsing and ingestion
│   │   └── ingest.py       # Enhanced ingestion with AI context
│   ├── evaluation/         # Evaluation scripts and datasets
│   └── rag_system/         # Core RAG implementation
│       ├── agents.py       # CrewAI agent definitions
│       ├── crew.py         # Workflow orchestration
│       └── tools.py        # Custom tools and utilities
├── notebooks/              # Jupyter notebooks for exploration
├── api.py                  # FastAPI server with OpenAI compatibility
├── main.py                 # Main application entry point
├── Dockerfile              # Docker configuration for RAG API
├── requirements.txt        # Python dependencies
├── requirements-docker.txt # Docker-specific dependencies
├── docker-compose.yml      # PostgreSQL setup
├── .env                    # Local environment variables
└── .env.docker            # Docker environment variables
```

## 🧪 Testing & Evaluation

### API Testing

```bash
# Test models endpoint
curl http://localhost:8000/v1/models

# Test chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag-model",
    "messages": [{"role": "user", "content": "What are the procurement standards?"}]
  }'
```

### Evaluation Suite

```bash
python src/evaluation/run_ragas_eval.py
```

### Jupyter Notebooks

Explore the system using the provided notebooks:

- `notebooks/01_parsing_and_chunking.ipynb` - Document processing exploration
- `notebooks/02_agent_and_evaluation.ipynb` - Agent testing and evaluation

## ⚠️ Troubleshooting

### Common Issues

**1. Models not appearing in OpenWebUI dropdown:**
- Verify RAG API is running: `curl http://localhost:8000/v1/models`
- Check OpenWebUI connection settings: `http://rag-api:8000/v1`
- Restart OpenWebUI container: `docker restart open-webui`

**2. Docker connectivity issues:**
- Ensure all containers are on the same network: `docker network inspect rag-network`
- Check container logs: `docker logs rag-api` and `docker logs open-webui`
- Verify environment files are configured correctly

**3. Database connection errors:**
- Ensure PostgreSQL container is running: `docker ps | grep postgres`
- Check database URL in environment files
- Verify pgvector extension is installed

**4. Ollama model errors:**
- Ensure Ollama is running: `ollama list`
- Pull missing models: `ollama pull gemma2:9b` and `ollama pull nomic-embed-text`
- Check Ollama base URL in environment files

### Logs and Debugging

```bash
# View all container logs
docker logs rag-api -f
docker logs open-webui -f
docker logs postgres -f

# Check network connectivity
docker exec rag-api ping open-webui
docker exec open-webui ping rag-api

# Test database connection
docker exec postgres psql -U postgres -d rag_db -c "SELECT count(*) FROM data_document_embeddings;"
```

## 🔒 Security Considerations

For production deployment, consider:

- Remove `allow_origins=["*"]` and specify allowed origins
- Add API authentication and rate limiting
- Use environment-specific configuration files
- Enable PostgreSQL SSL/TLS
- Implement proper logging and monitoring

## 🚀 Production Deployment

### Docker Compose Production Setup

Create a `docker-compose.prod.yml`:

```yaml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg14
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - rag-network

  rag-api:
    build: .
    environment:
      DATABASE_URL: postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/rag_db
      OLLAMA_BASE_URL: ${OLLAMA_BASE_URL}
    depends_on:
      - postgres
    networks:
      - rag-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - rag-api
    networks:
      - rag-network

volumes:
  postgres_data:

networks:
  rag-network:
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CrewAI](https://github.com/joaomdmoura/crewAI) for the multi-agent framework
- [LlamaIndex](https://github.com/run-llama/llama_index) for RAG utilities and DoclingReader
- [Ollama](https://ollama.ai/) for local LLM inference
- [pgvector](https://github.com/pgvector/pgvector) for vector similarity search
- [OpenWebUI](https://github.com/open-webui/open-webui) for the web interface
- [FastAPI](https://github.com/tiangolo/fastapi) for the API framework

## 📞 Support

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is a proof-of-concept implementation. For production use, consider additional security measures, error handling, and scalability optimizations.
