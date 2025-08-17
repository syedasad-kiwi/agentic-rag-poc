# Agentic RAG POC

A proof-of-concept implementation of an agentic Retrieval-Augmented Generation (RAG) system using CrewAI, LlamaIndex, and PostgreSQL with pgvector for intelligent document analysis and question answering.

## 🚀 Features

- **Intelligent Document Processing**: Automatically parses and chunks PDF and DOCX documents using DoclingReader
- **Vector Search**: PostgreSQL with pgvector extension for efficient similarity search
- **Agentic Workflow**: CrewAI agents that can retrieve, analyze, and synthesize information
- **Source Attribution**: Responses include source file references for transparency
- **Graceful Failure Handling**: System responds appropriately when no relevant information is found
- **Local LLM Integration**: Uses Ollama for embedding and language model inference

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
│   Raw Documents │    │   Embeddings    │    │   Intelligent   │
│   (PDF/DOCX)    │    │   & Metadata    │    │   Responses     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL 14+ with pgvector extension
- Ollama with required models
- Docker (optional, for containerized PostgreSQL)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/syedasad-kiwi/agentic-rag-poc.git
cd agentic-rag-poc
```

### 2. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Up PostgreSQL with pgvector

#### Option A: Using Docker (Recommended)

```bash
docker-compose up -d
```

#### Option B: Local Installation

1. Install PostgreSQL 14+
2. Install pgvector extension
3. Create database and enable extension

### 4. Set Up Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

### 5. Environment Configuration

Create a `.env` file in the root directory:

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db
```

## 📊 Usage

### 1. Document Ingestion

Place your documents (PDF/DOCX) in the `data/raw/` directory, then run:

```bash
python src/data_ingestion/ingest.py
```

### 2. Query the System

```bash
python main.py
```

Or use the system programmatically:

```python
from src.rag_system.crew import create_rag_crew

# Create and run the crew
crew = create_rag_crew('What are the key procurement requirements?')
result = crew.kickoff()
print(result)
```

### 3. Jupyter Notebooks

Explore the system using the provided notebooks:

- `notebooks/01_parsing_and_chunking.ipynb` - Document processing exploration
- `notebooks/02_agent_and_evaluation.ipynb` - Agent testing and evaluation

## 🔧 Configuration

### Key Components

- **Document Retrieval Tool**: Custom CrewAI tool with pydantic validation
- **Agents**: Specialized agents for document research and analysis
- **Vector Store**: PostgreSQL with pgvector for similarity search
- **Embeddings**: Ollama nomic-embed-text model

### Customization

- Modify agent behavior in `src/rag_system/agents.py`
- Adjust retrieval parameters in `src/rag_system/tools.py`
- Configure workflow in `src/rag_system/crew.py`

## 📁 Project Structure

```
agentic-rag-poc/
├── data/
│   ├── raw/                 # Input documents (PDF/DOCX)
│   └── processed/           # Processed data
├── src/
│   ├── config/             # Configuration settings
│   ├── data_ingestion/     # Document parsing and ingestion
│   ├── evaluation/         # Evaluation scripts and datasets
│   └── rag_system/         # Core RAG implementation
│       ├── agents.py       # CrewAI agent definitions
│       ├── crew.py         # Workflow orchestration
│       └── tools.py        # Custom tools and utilities
├── notebooks/              # Jupyter notebooks for exploration
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # PostgreSQL setup
└── main.py                # Main application entry point
```

## 🔍 Key Features Explained

### Source Attribution

The system automatically extracts and includes source file information in responses:

```python
# Example response format
"Based on the Abu Dhabi Procurement Standards document..."

Sources:
- Abu Dhabi Procurement Standards.PDF
- Procurement Manual (Business Process).PDF
```

### Graceful Error Handling

When no relevant documents are found, the system responds appropriately:

```
"Based on the available documents, I cannot find relevant information to answer this question."
```

### Hierarchical Agent Process

Uses CrewAI's hierarchical process for reliable task coordination and context passing between agents.

## 🧪 Testing

Run the evaluation suite:

```bash
python src/evaluation/run_ragas_eval.py
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
- [LlamaIndex](https://github.com/run-llama/llama_index) for RAG utilities
- [Ollama](https://ollama.ai/) for local LLM inference
- [pgvector](https://github.com/pgvector/pgvector) for vector similarity search

## 📞 Support

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is a proof-of-concept implementation. For production use, consider additional security measures, error handling, and scalability optimizations.
