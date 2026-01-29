#!/bin/bash

# Qdrant Cloud Upload Workflow
# This script automates the process of building local index and uploading to cloud

set -e  # Exit on error

echo "=================================================="
echo "  Qdrant Cloud Upload Workflow"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please create a .env file with QDRANT_HOST and QDRANT_API_KEY"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated${NC}"
    echo "Activating virtual environment..."
    
    if [ -f "myenv/bin/activate" ]; then
        source myenv/bin/activate
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo -e "${RED}Error: No virtual environment found${NC}"
        echo "Please create a virtual environment or activate it manually"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Step 1: Check if Ollama is running
echo "Step 1: Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${RED}Error: Ollama is not running${NC}"
    echo "Please start Ollama before running this script"
    echo "Run: ollama serve"
    exit 1
fi
echo -e "${GREEN}✓ Ollama is running${NC}"
echo ""

# Step 2: Check if local data needs to be built
echo "Step 2: Checking local Qdrant data..."
if [ -d "./qdrant_data/collection/cancer_data_sharing" ]; then
    echo -e "${YELLOW}Local Qdrant data already exists${NC}"
    read -p "Do you want to rebuild it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Rebuilding local index..."
        python ingest_pipeline.py --data-dir ./data --vector-store-path ./qdrant_data
    else
        echo "Using existing local data"
    fi
else
    echo "Building local index (this may take a few minutes)..."
    python ingest_pipeline.py --data-dir ./data --vector-store-path ./qdrant_data
fi
echo -e "${GREEN}✓ Local index ready${NC}"
echo ""

# Step 3: Upload to cloud
echo "Step 3: Uploading to Qdrant Cloud..."
echo -e "${YELLOW}This will upload all vectors to your Qdrant Cloud instance${NC}"
read -p "Continue? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    python upload_to_qdrant_cloud.py
    echo -e "${GREEN}✓ Upload completed${NC}"
else
    echo "Upload cancelled"
    exit 0
fi
echo ""

# Step 4: Verify
echo "=================================================="
echo "  Upload Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Verify the upload in Qdrant Cloud dashboard"
echo "2. Run the chatbot:"
echo "   streamlit run chatbot_app.py"
echo "   or"
echo "   streamlit run chatbot_app_openAI.py"
echo ""
echo "The chatbot will automatically use Qdrant Cloud storage."
echo ""
