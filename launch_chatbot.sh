#!/bin/bash

# Launch script for Cancer Data Sharing Chatbot

echo "🧬 Cancer Data Sharing Chatbot Launcher"
echo "======================================"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "Activating myenv..."
    source myenv/bin/activate
fi

# Check if Ollama is running
echo "Checking Ollama service..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running"
    echo "Please start Ollama in another terminal:"
    echo "  ollama serve"
    echo ""
    exit 1
else
    echo "✅ Ollama is running"
fi

# Check if llama3.2 model exists
echo "Checking for llama3.2 model..."
if ! ollama list | grep -q "llama3.2"; then
    echo "❌ llama3.2 model not found"
    echo "Pulling llama3.2 model (this may take a while)..."
    ollama pull llama3.2
fi

# Check if Qdrant data exists
if [ ! -d "qdrant_data/collection/cancer_data_sharing" ]; then
    echo "⚠️  Qdrant data not found"
    echo "Run the ingestion pipeline first:"
    echo "  python ingest_pipeline.py"
    echo ""
    exit 1
else
    echo "✅ Qdrant data found"
fi

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Streamlit not installed"
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

echo ""
echo "🚀 Launching chatbot..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the chatbot"
echo ""

# Launch Streamlit
streamlit run chatbot_app.py
