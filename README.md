# Cancer Data Sharing AI Chatbot

An intelligent AI assistant for NCI data sharing policies and guidelines, built with LlamaIndex, Qdrant, and Streamlit.

## Overview

This application helps cancer researchers, data managers, and institutional officials understand:
- NCI/NIH data sharing policies and requirements
- Data submission and access processes
- Genomic data sharing guidelines
- Available datasets and repositories
- Data management and sharing plan (DMSP) requirements

### Key Features

- **Intelligent Query Classification**: Automatically categorizes questions to retrieve the most relevant information
- **Context-Aware Routing**: Routes queries to appropriate document types based on classification
- **Metadata-Based Filtering**: Filters content by document type, category, and other metadata
- **Response Quality Evaluation**: Assesses responses on relevance, accuracy, completeness, clarity, and actionability
- **Streamlit UI**: Clean, interactive web interface with chat history and source citations
- **Dual LLM Support**: Works with both Ollama (Llama 3.2) and OpenAI (GPT-4o-mini, GPT-4o)
- **Flexible Deployment**: Supports local Qdrant and Qdrant Cloud

## Architecture

### System Components

```
Streamlit UI → LlamaIndex (Query Router + RAG) → Qdrant Vector Store
                    ↓
              Ollama/OpenAI LLM
                    ↓
         HuggingFace Embeddings (E5-Large-V2)
```

### Data Pipeline

```
Markdown Files → Parse & Chunk → LLM Classify → Enrich Metadata → Create Nodes → Vector Store
```

### Query Categories

Queries are classified into these categories for intelligent routing:

1. **guidance** - Guidelines and best practices
2. **policy** - Rules and requirements
3. **process** - Step-by-step procedures
4. **resources** - Training materials, guides, templates, tools, datasets
5. **glossary** - Definitions and terminology
6. **faq** - Common questions and answers
7. **news** - News, announcements, and updates

### Document Metadata

Each document chunk includes rich metadata for filtering:

- **document_type**: About, Data, Guidance, Process, News
- **category**: Content classification (guidance, policy, process, etc.)
- **source_file**: Origin file path
- **section_title**: Markdown section context
- **agencies**: NIH, NCI, FDA, CDC, NSF, DOD, etc.
- **repositories**: dbGaP, SRA, GEO, PDC, GDC, CDS, IDC
- **data_types**: genomic, clinical, imaging, proteomic, etc.
- **subject_types**: human, animal, cell_line, tissue
- **policy_references**: NIH_DMS_Policy, GDSP, etc.
- **requirements**: DMS Plan, Consent, IRB, Access Control
- **keywords**: Extracted significant terms

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Ollama (for local LLM) OR OpenAI API key (for cloud LLM)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd data-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials
```

### Option 1: Using Ollama (Local)

```bash
# Install Ollama from https://ollama.ai

# Pull Llama 3.2 model
ollama pull llama3.2

# Run the chatbot
streamlit run chatbot_app.py
```

### Option 2: Using OpenAI (Cloud)

```bash
# Add your OpenAI API key to .streamlit/secrets.toml
echo 'OPENAI_API_KEY = "sk-..."' > .streamlit/secrets.toml

# Run the chatbot
streamlit run chatbot_app_openAI.py
```

## Data Ingestion

### Initial Setup

Before running the chatbot, ingest your Markdown documents:

```bash
# Basic ingestion (local Qdrant)
python ingest_pipeline.py --data-dir ./data

# Using Qdrant Cloud
python ingest_pipeline.py --data-dir ./data --use-cloud
```

### Ingestion Options

```bash
python ingest_pipeline.py \
  --data-dir ./data \
  --vector-store-path ./qdrant_data \
  --collection-name cancer_data_sharing \
  --chunk-size 1024
```

### Upload to Qdrant Cloud

If you've ingested locally and want to migrate to cloud:

```bash
python upload_to_qdrant_cloud.py \
  --local-path ./qdrant_data \
  --collection-name cancer_data_sharing
```

## Configuration

### Environment Variables

Create `.streamlit/secrets.toml` for sensitive data:

```toml
# OpenAI API Key (required for chatbot_app_openAI.py)
OPENAI_API_KEY = "sk-..."

# Qdrant Cloud Configuration (optional)
QDRANT_HOST = "https://your-cluster.cloud.qdrant.io:6333"
QDRANT_API_KEY = "your-api-key"

# Ollama Configuration (for chatbot_app.py)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

# Embedding Model
EMBEDDING_MODEL = "intfloat/e5-large-v2"

# Data Configuration
DATA_DIR = "./data"
VECTOR_STORE_PATH = "./qdrant_data"
COLLECTION_NAME = "cancer_data_sharing"
```

### Application Settings

Adjustable through the Streamlit sidebar:

- **Response Creativity** (Temperature): 0.0 - 1.0
- **Number of Sources**: 1 - 10 retrieved documents
- **Show Query Classification**: Display query routing information
- **Show Response Evaluation**: Display quality metrics

## Project Structure

```
data-pipeline/
├── chatbot_app.py              # Main chatbot (Ollama/Llama 3.2)
├── chatbot_app_openAI.py       # OpenAI variant (GPT-4o-mini)
├── ingest_pipeline.py          # Data ingestion and indexing
├── upload_to_qdrant_cloud.py   # Upload local data to cloud
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── secrets.toml           # API keys and credentials
├── data/                      # Markdown corpus
│   ├── About/
│   ├── Data/
│   ├── Guidance/
│   ├── News/
│   ├── Process/
│   └── documents/
└── qdrant_data/              # Local vector database
```

### Key Components

#### Ingestion Pipeline (`ingest_pipeline.py`)
- **MarkdownCorpusParser**: Discovers and parses Markdown files
- **ChunkClassifier**: LLM-based content classification
- **MetadataEnricher**: Extracts agencies, repositories, data types, etc.
- **IngestionPipeline**: Orchestrates the complete indexing workflow

#### Chatbot Application (`chatbot_app.py`, `chatbot_app_openAI.py`)
- **QueryClassifier**: Classifies user queries for intelligent routing
- **QueryRouter**: Routes queries to appropriate document types
- **ResponseEvaluator**: Evaluates response quality on multiple dimensions
- **Streamlit UI**: Interactive chat interface with source citations

## Features

### 1. Intelligent Query Classification

The chatbot automatically classifies queries using both keyword matching and LLM analysis:

- **guidance**: "What are best practices for..."
- **policy**: "What rules apply to..."
- **process**: "How do I submit..."
- **resources**: "What tools are available..."
- **glossary**: "What does DMSP mean?"
- **faq**: "Can I share data without..."
- **news**: "What's new in..."

### 2. Context-Aware Routing

Queries are routed to relevant document types based on classification:

- **Guidance queries** → Guidance documents
- **Process queries** → Process documents
- **Resource queries** → Data/Resources
- **Definition queries** → About/Glossary

### 3. Metadata-Based Filtering

Retrieved content is filtered by:
- Document type (About, Data, Guidance, Process, News)
- Categories (guidance, policy, process, etc.)
- Agencies, repositories, data types
- Policy requirements and compliance levels

### 4. Response Quality Evaluation

Each response is evaluated on:
- **Relevance**: How well it addresses the query
- **Accuracy**: Factual correctness based on context
- **Completeness**: Thoroughness of the answer
- **Clarity**: Ease of understanding
- **Actionability**: Provides actionable information

### 5. Source Citations

Every response includes:
- Source file paths
- Section titles and document types
- Relevant metadata (repositories, data types, etc.)
- Text snippets from source documents

## Usage Examples

### Example Questions

Try asking:

1. **Policy Questions**
   - "What is the NIH Data Management and Sharing Policy?"
   - "What are the requirements for genomic data sharing?"

2. **Process Questions**
   - "How do I submit data to dbGaP?"
   - "What are the steps to access cancer datasets?"

3. **Resource Questions**
   - "What repositories are available for cancer data?"
   - "What tools can help with data sharing?"

4. **Definition Questions**
   - "What is a Data Management and Sharing Plan?"
   - "What does controlled access mean?"

5. **FAQ Questions**
   - "Can I share data from non-NIH funded studies?"
   - "What privacy protections are required?"

## Deployment

### Local Development

```bash
streamlit run chatbot_app.py
# or
streamlit run chatbot_app_openAI.py
```

### Production Deployment

#### Streamlit Cloud

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Add secrets in Streamlit Cloud dashboard
4. Deploy

#### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "chatbot_app_openAI.py"]
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check model is pulled: `ollama pull llama3.2`

2. **OpenAI API Error**
   - Verify API key in `.streamlit/secrets.toml`
   - Check API key has sufficient credits

3. **Qdrant Connection Error**
   - For local: Check `qdrant_data/` exists with data
   - For cloud: Verify `QDRANT_HOST` and `QDRANT_API_KEY`

4. **Empty Responses**
   - Ensure data ingestion completed successfully
   - Check collection name matches in all files

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Acknowledgments

- Built with [LlamaIndex](https://www.llamaindex.ai/)
- Vector database by [Qdrant](https://qdrant.tech/)
- UI powered by [Streamlit](https://streamlit.io/)
- Embeddings from [HuggingFace](https://huggingface.co/)
