# Data Ingestion Pipeline

This script implements the **High-Level Data & Indexing Design** from the AI Chatbot system architecture.

## Overview

The ingestion pipeline processes a unified Markdown corpus and creates a searchable vector database with rich metadata for intelligent retrieval.

### Key Features

- **Markdown Corpus Processing**: Parses all `.md` files from the `data/` directory
- **LLM-Assisted Classification**: Automatically categorizes chunks into 12 content types
- **Metadata Enrichment**: Infers funding sources, subject types, repositories, and policy requirements
- **Vector Indexing**: Creates searchable embeddings with metadata filtering support
- **Portkey Integration**: Optional routing through Portkey.ai gateway

## Architecture

```
Markdown Files → Parse & Chunk → LLM Classify → Enrich Metadata → Create Nodes → Vector Store
```

### Chunk Categories

The pipeline classifies content into these categories:

1. **policy** - Rules and mandates that must be followed
2. **scope** - Who/what the policy covers
3. **process** - How to submit, share, or access data
4. **technical** - Data formats, metadata standards
5. **privacy_security** - Human subject protection, access controls
6. **costs_funding** - Budget requirements and fees
7. **data_reuse** - How to use/cite shared data
8. **compliance** - Oversight and enforcement
9. **resources** - Training materials and guides
10. **dataset_access** - Finding and using datasets
11. **glossary** - Definitions and terminology
12. **faq** - Frequently asked questions

### Metadata Structure

Each chunk is enriched with:

- **category**: Content type classification
- **source_file**: Origin file path
- **section_title**: Markdown section context
- **funding_filters**: [NIH, NCI, DOD, NSF]
- **subject_filters**: [human, animal, cell_line]
- **repository_tags**: [dbGaP, SRA, GEO, PDC, GDC, CDS]
- **policy_requirements**: [DMS Plan, Consent, Access Control, etc.]
- **keywords**: Extracted significant terms

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install and start Ollama (if not already installed)
# Visit https://ollama.ai for installation instructions

# Pull Llama 3.2 model
ollama pull llama3.2

# Set up environment variables
cp .env.example .env
# Edit .env if needed (default Ollama settings should work)
```

## Usage

### Basic Usage

```bash
# Run with default settings
python ingest_pipeline.py

# Specify custom data directory
python ingest_pipeline.py --data-dir ./data

# Use custom collection name
python ingest_pipeline.py --collection-name my_collection
```

### With Portkey Gateway

```bash
# Route LLM calls through Portkey.ai
python ingest_pipeline.py \
  --use-portkey \
  --portkey-api-key YOUR_PORTKEY_KEY
```

### Advanced Options

```bash
python ingest_pipeline.py \
  --data-dir ./data \
  --vector-store-path ./qdrant_data \
  --collection-name cancer_data_sharing \
  --chunk-size 1024 \
  --use-portkey \
  --portkey-api-key YOUR_KEY
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `./data` | Path to Markdown corpus |
| `--vector-store-path` | `./qdrant_data` | Path for Qdrant storage |
| `--collection-name` | `cancer_data_sharing` | Vector collection name |
| `--chunk-size` | `1024` | Target chunk size (tokens) |
| `--use-portkey` | `False` | Route through Portkey.ai |
| `--portkey-api-key` | - | Portkey API key |

## Environment Variables

Create a `.env` file with:

```env
# OpenAI API Key (required)
OPENAI_API_KEY=sk-...

# Portkey Configuration (optional)
PORTKEY_API_KEY=pk-...
PORTKEY_BASE_URL=https://api.portkey.ai/v1

# Qdrant Configuration (optional)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## Code Structure

### Main Components

#### `MarkdownCorpusParser`
- Discovers all Markdown files
- Extracts frontmatter metadata
- Parses document structure and sections

#### `ChunkClassifier`
- Uses LLM to classify content type
- Supports batch processing
- Configurable with Portkey routing

#### `MetadataEnricher`
- Keyword-based metadata inference
- Extracts funding sources, subjects, repositories
- Identifies policy requirements

#### `IngestionPipeline`
- Orchestrates the complete workflow
- Creates LlamaIndex nodes with metadata
- Builds vector index with Qdrant

## Example Integration

```python
from ingest_pipeline import IngestionPipeline

# Create pipeline
pipeline = IngestionPipeline(
    data_dir="./data",
    collection_name="cancer_data_sharing",
    chunk_size=1024,
    use_portkey=True,
    portkey_api_key="pk-..."
)

# Run ingestion
index = pipeline.run_pipeline()

# Index is now ready for querying
```

## Querying with Metadata Filters

After ingestion, query with metadata filters:

```python
from llama_index.core import VectorStoreIndex

# Load existing index
index = VectorStoreIndex.from_vector_store(vector_store)

# Query with filters
query_engine = index.as_query_engine(
    filters={
        "category": "policy",
        "funding_filters": ["NIH", "NCI"],
        "subject_filters": ["human"]
    }
)

response = query_engine.query(
    "What are the requirements for sharing human genomic data?"
)
```

## Output

The pipeline logs:

- Number of files processed
- Total chunks created
- Classification statistics by category
- Metadata enrichment counts
- Vector indexing progress

Example output:

```
INFO - Found 47 Markdown files
INFO - Processing: data/Guidance/genomic-data-sharing.md
INFO - Created 23 chunks from genomic-data-sharing.md
INFO - Total nodes created: 1,247
INFO - Ingestion pipeline completed successfully
INFO - === Indexing Statistics ===
INFO - Total nodes: 1247
INFO - Nodes by category:
INFO -   policy: 342
INFO -   process: 298
INFO -   faq: 187
INFO -   technical: 156
INFO -   resources: 124
INFO -   scope: 89
INFO -   privacy_security: 51
```

## Next Steps

After ingestion, the vector database is ready for:

1. **Query Routing**: Create specialized retrievers per category
2. **ChatEngine Integration**: Build conversational RAG
3. **Metadata Filtering**: Apply context-aware retrieval
4. **API Development**: Expose via FastAPI backend

See the system design document for the complete RAG architecture.

Hello world.
