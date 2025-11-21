# Quick Start Guide - Cancer Data Sharing Chatbot

## 🚀 Quick Launch (3 Steps)

### 1. Start Ollama (in one terminal)
```bash
ollama serve
```

### 2. Test Setup
```bash
python test_chatbot_setup.py
```

### 3. Launch Chatbot
```bash
./launch_chatbot.sh
# OR
streamlit run chatbot_app.py
```

The app will open at: **http://localhost:8501**

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `chatbot_app.py` | Main Streamlit chatbot application |
| `ingest_pipeline.py` | Data ingestion pipeline (already run) |
| `test_chatbot_setup.py` | Verify all components work |
| `launch_chatbot.sh` | One-click launcher script |
| `requirements.txt` | Python dependencies |
| `qdrant_data/` | Vector database (already populated) |

---

## ⚙️ Prerequisites Check

✅ **Python 3.9+** installed  
✅ **Virtual environment** activated (`source myenv/bin/activate`)  
✅ **Ollama** installed and running  
✅ **llama3.2** model downloaded (`ollama pull llama3.2`)  
✅ **Dependencies** installed (`pip install -r requirements.txt`)  
✅ **Qdrant data** exists (already in `qdrant_data/`)

---

## 💬 Using the Chatbot

### Example Questions
- "What is the NIH Data Management and Sharing Policy?"
- "How do I submit genomic data to dbGaP?"
- "What privacy protections are required for human data?"
- "What repositories are available for cancer data?"
- "How do I write a Data Management and Sharing Plan?"

### Features
- **Interactive chat** with conversation memory
- **Source citations** with metadata
- **Adjustable settings** (creativity, number of sources)
- **Example questions** in sidebar
- **Clear history** button

---

## 🔧 Troubleshooting

### Error: "Failed to initialize chatbot"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Verify Qdrant data
ls -la qdrant_data/collection/cancer_data_sharing/
```

### Error: Connection to Ollama failed
```bash
# Start Ollama in another terminal
ollama serve

# Verify model is downloaded
ollama list
```

### Slow first response
Normal! The first query loads the embedding model into memory (~1-2GB). Subsequent queries are much faster.

---

## 🎯 Architecture Overview

```
┌──────────────────┐
│  Streamlit UI    │  ← User Interface
└────────┬─────────┘
         │
┌────────▼─────────┐
│  LlamaIndex      │  ← Orchestration
│  Chat Engine     │
└────────┬─────────┘
         │
    ┌────▼────┐
    │ Query   │
    └────┬────┘
         │
    ┌────▼──────────────────────┐
    │                            │
┌───▼────────┐        ┌──────▼──────┐
│ E5-Large-V2│        │   Qdrant    │
│ Embeddings │ ←────→ │   Vector    │
└────────────┘        │     DB      │
                      └──────┬──────┘
                             │
                    ┌────────▼────────┐
                    │ Retrieved Docs  │
                    │   + Context     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Llama 3.2 LLM  │
                    │    (Ollama)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Response +    │
                    │    Sources      │
                    └─────────────────┘
```

---

## 📊 System Requirements

- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 5GB free space (for models and data)
- **CPU**: Multi-core recommended
- **GPU**: Optional (speeds up embeddings)

---

## 🔐 Data & Privacy

- All data stays **local** on your machine
- No external API calls (except for model downloads)
- Ollama runs locally (no data sent to cloud)
- Qdrant database is local file-based

---

## 📚 Additional Resources

- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [Ollama Documentation](https://ollama.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)

---

## 🆘 Need Help?

1. Run the test script: `python test_chatbot_setup.py`
2. Check terminal logs where Streamlit is running
3. Verify Ollama logs: `ollama serve` terminal
4. See detailed setup guide: `CHATBOT_SETUP.md`

---

## 🎨 Customization

### Change LLM Model
Edit `chatbot_app.py`:
```python
LLM_MODEL = "llama3.2"  # Change to: llama3.1, mistral, etc.
```

### Adjust Response Style
In Streamlit UI sidebar:
- **Response Creativity**: 0.0 (precise) to 1.0 (creative)
- **Number of Sources**: 1-10 documents

### Modify System Prompt
Edit the `system_prompt` in `create_chat_engine()` function in `chatbot_app.py`

---

**Ready to chat? Run `./launch_chatbot.sh` and start asking questions! 🚀**
