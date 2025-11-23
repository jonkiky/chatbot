# Qdrant Cloud Setup Guide

This guide explains how to set up your chatbot to use Qdrant Cloud for vector storage, which is recommended for production deployments.

## Overview

The new workflow supports two modes:

1. **Local Mode**: Store vectors in local `./qdrant_data` directory (good for development)
2. **Cloud Mode**: Store vectors in Qdrant Cloud (recommended for production)

## Prerequisites

- Python 3.11+ environment set up
- Ollama running locally (for data ingestion classification)
- Qdrant Cloud account and credentials

## Step 1: Configure Environment Variables

Your `.env` file should already contain the Qdrant Cloud credentials:

```env
# Qdrant Cloud Configuration
QDRANT_API_KEY=e
QDRANT_HOST=
```

The application will automatically detect these credentials and use cloud mode.

## Step 2: Build Local Index First (Recommended)

It's recommended to build your vector index locally first, then upload to cloud:

```bash
# Build local index
python ingest_pipeline.py --data-dir ./data --vector-store-path ./qdrant_data
```

This will:
- Process all markdown files in `./data`
- Classify chunks using local Ollama LLM
- Create embeddings using HuggingFace E5-Large-V2
- Store vectors in `./qdrant_data` directory

## Step 3: Upload to Qdrant Cloud

Once you have the local index built, upload it to cloud:

```bash
# Upload to Qdrant Cloud
python upload_to_qdrant_cloud.py
```

This will:
- Read all vectors from `./qdrant_data`
- Create collection in Qdrant Cloud
- Upload all vectors and metadata
- Verify the upload was successful

You can also specify custom parameters:

```bash
python upload_to_qdrant_cloud.py \
  --local-path ./qdrant_data \
  --collection-name cancer_data_sharing \
  --batch-size 100
```

## Step 4: Run Chatbot with Cloud Storage

Once uploaded, your chatbot applications will automatically use cloud storage:

```bash
# With Ollama (local LLM)
streamlit run chatbot_app.py

# With OpenAI (cloud LLM)
streamlit run chatbot_app_openAI.py
```

The applications automatically detect the cloud credentials and connect to Qdrant Cloud.

## Alternative: Direct Cloud Ingestion

You can also ingest directly to cloud (skipping local storage):

```bash
python ingest_pipeline.py --data-dir ./data --use-cloud
```

⚠️ **Note**: This requires Ollama to be running locally for chunk classification, even though vectors are stored in cloud.

## Verification

To verify your cloud setup:

1. **Check upload logs**: Look for "Upload completed successfully! ✓" message
2. **Check point counts**: Local and cloud counts should match
3. **Test chatbot**: Run the chatbot and verify it can retrieve relevant documents

## Switching Between Local and Cloud

The application automatically switches based on your `.env` configuration:

- **Use Cloud**: Set `QDRANT_HOST` to your cloud URL and provide `QDRANT_API_KEY`
- **Use Local**: Set `QDRANT_HOST=localhost` or remove cloud credentials

## Benefits of Cloud Storage

1. **Scalability**: No local disk space constraints
2. **Performance**: Optimized cloud infrastructure
3. **Reliability**: Managed backups and high availability
4. **Deployment**: Easy to deploy chatbot without bundling large vector data

## Troubleshooting

### "Cloud credentials not found"
- Check that `QDRANT_HOST` and `QDRANT_API_KEY` are set in `.env`
- Verify the `.env` file is in the project root directory

### "Collection not found"
- Make sure you've run the upload script successfully
- Check cloud dashboard to verify collection exists

### "Connection failed"
- Verify your cloud URL is correct
- Check that your API key is valid
- Ensure you have internet connectivity

### Point count mismatch
- Re-run the upload script
- Check for errors during upload process
- Verify local data integrity

## Production Deployment Checklist

- [ ] Local index built successfully
- [ ] Uploaded to Qdrant Cloud
- [ ] Verified point counts match
- [ ] Tested chatbot with cloud storage
- [ ] Updated deployment configs to use cloud credentials
- [ ] Removed local `qdrant_data/` from deployment package (optional)

## Cost Considerations

Qdrant Cloud pricing is based on:
- Storage size (vectors + metadata)
- Query volume
- Cluster size/performance tier

For this chatbot with ~1000-2000 document chunks:
- Storage: ~100-200 MB
- Estimated cost: Check Qdrant Cloud pricing for current rates

## Security Best Practices

1. **Never commit credentials**: Keep `.env` file in `.gitignore`
2. **Use environment variables**: In production, use secure secret management
3. **Rotate API keys**: Periodically regenerate API keys
4. **Monitor access**: Review Qdrant Cloud access logs

## Additional Resources

- [Qdrant Cloud Documentation](https://qdrant.tech/documentation/cloud/)
- [LlamaIndex Qdrant Integration](https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/)
- [Chatbot Setup Guide](CHATBOT_SETUP.md)
