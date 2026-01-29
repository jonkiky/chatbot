# Cancer Data Sharing Chatbot Setup Guide

## Overview
This chatbot uses LlamaIndex, Qdrant (vector database), and Streamlit to provide an interactive AI assistant for cancer data sharing policies and guidelines.

## Prerequisites

### 1. Install Ollama (for LLM)
```bash
# macOS
brew install ollama

# Start Ollama service
ollama serve

# Pull Llama 3.2 model (in a new terminal)
ollama pull llama3.2
```

### 2. Python Environment
You should have Python 3.9+ installed.

## Installation

### Step 1: Install Dependencies
```bash
# Activate your virtual environment
source myenv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Verify Qdrant Data
Make sure your `qdrant_data` directory exists and contains the `cancer_data_sharing` collection:
```bash
ls -la qdrant_data/
# Should show: meta.json and collection/ directory
```

If you haven't ingested your data yet, run:
```bash
python ingest_pipeline.py --data-dir ./data --vector-store-path ./qdrant_data
```

## Running the Chatbot

### Start the Streamlit App
```bash
streamlit run chatbot_app.py
```

The chatbot will open in your browser at `http://localhost:8501`

## Features

### 1. **Interactive Chat Interface**
- Ask questions in natural language
- Get responses based on NCI data sharing documentation
- View conversation history

### 2. **Source Citations**
- See which documents were used to answer your question
- View metadata (category, document type, policies, repositories)
- Access original text snippets

### 3. **Configurable Settings**
- Adjust response creativity (temperature)
- Change number of sources retrieved
- Clear chat history

### 4. **Example Questions**
The sidebar provides quick-start questions like:
- "What is the NIH Data Management and Sharing Policy?"
- "How do I submit genomic data to dbGaP?"
- "What are the requirements for a Data Management and Sharing Plan?"

## Architecture

```
User Question
    ↓
Streamlit UI
    ↓
LlamaIndex Query Engine
    ↓
Embedding (E5-Large-V2) → Qdrant Vector Search
    ↓
Retrieved Context + Question → Llama 3.2 LLM
    ↓
Response with Sources
    ↓
Display in UI
```

## Configuration

Edit `ChatbotConfig` class in `chatbot_app.py` to customize:

```python
class ChatbotConfig:
    VECTOR_STORE_PATH = "./qdrant_data"
    COLLECTION_NAME = "cancer_data_sharing"
    EMBEDDING_MODEL = "intfloat/e5-large-v2"
    LLM_MODEL = "llama3.2"
    TOP_K_RESULTS = 5
    TEMPERATURE = 0.7
    CONTEXT_WINDOW = 8192
```

## Troubleshooting

### Issue: "Failed to initialize chatbot"
**Solution:** Ensure Qdrant data exists and Ollama is running
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Verify Qdrant data
ls -la qdrant_data/collection/cancer_data_sharing/
```

### Issue: "Connection error to Ollama"
**Solution:** Start Ollama service
```bash
ollama serve
```

### Issue: Slow first response
**Solution:** First query loads the embedding model into memory. Subsequent queries will be faster.

### Issue: Out of memory
**Solution:** Reduce `TOP_K_RESULTS` in settings or use a smaller LLM model

## Advanced Usage

### Custom System Prompt
Modify the `system_prompt` in the `create_chat_engine()` function to customize the chatbot's behavior.

### Add Filters
You can filter by metadata when retrieving documents:
```python
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="category", value="policy"),
        MetadataFilter(key="document_type", value="Guidance")
    ]
)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
    filters=filters
)
```

### Export Chat History
Add this to the sidebar:
```python
if st.button("Export Chat"):
    import json
    chat_export = json.dumps(st.session_state.messages, indent=2)
    st.download_button(
        "Download Chat History",
        chat_export,
        "chat_history.json",
        "application/json"
    )
```

## Performance Tips

1. **Cache the index**: The chatbot uses `@st.cache_resource` to cache the vector index
2. **Adjust TOP_K**: Lower values (3-5) are faster but may miss relevant info
3. **Temperature**: Lower values (0.3-0.5) for more focused, faster responses
4. **GPU**: If available, torch will automatically use GPU for embeddings

## Next Steps

- **Deploy to Cloud**: Use Streamlit Cloud, AWS, or Azure
- **Add Authentication**: Integrate user authentication if needed
- **Analytics**: Track common questions and usage patterns
- **Feedback Loop**: Add rating buttons to collect user feedback
- **Multi-turn Refinement**: Already supported via chat memory

## Support

For issues or questions:
1. Check logs in terminal where Streamlit is running
2. Verify Ollama is running: `ollama list`
3. Check Qdrant data integrity
4. Review LlamaIndex documentation: https://docs.llamaindex.ai/
