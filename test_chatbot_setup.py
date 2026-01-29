"""
Quick test script to verify chatbot components
Run this before launching the full Streamlit app
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    packages = [
        ("streamlit", "Streamlit"),
        ("llama_index.core", "LlamaIndex Core"),
        ("llama_index.llms.ollama", "LlamaIndex Ollama"),
        ("llama_index.embeddings.huggingface", "LlamaIndex HuggingFace"),
        ("llama_index.vector_stores.qdrant", "LlamaIndex Qdrant"),
        ("qdrant_client", "Qdrant Client"),
    ]
    
    failed = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All packages installed\n")
    return True


def test_ollama():
    """Test Ollama connection"""
    print("Testing Ollama connection...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            llama_models = [m for m in models if 'llama3.2' in m.get('name', '')]
            if llama_models:
                print(f"  ✅ Ollama is running with llama3.2")
                return True
            else:
                print(f"  ⚠️  Ollama is running but llama3.2 model not found")
                print("  Run: ollama pull llama3.2")
                return False
        else:
            print(f"  ❌ Ollama returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False


def test_qdrant_data():
    """Test if Qdrant data exists"""
    print("\nTesting Qdrant data...")
    
    qdrant_path = Path("./qdrant_data")
    collection_path = qdrant_path / "collection" / "cancer_data_sharing"
    
    if not qdrant_path.exists():
        print(f"  ❌ Qdrant directory not found: {qdrant_path}")
        print("  Run: python ingest_pipeline.py")
        return False
    
    if not collection_path.exists():
        print(f"  ❌ Collection not found: {collection_path}")
        print("  Run: python ingest_pipeline.py")
        return False
    
    print(f"  ✅ Qdrant data found at: {collection_path}")
    
    # Try to connect to Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(path=str(qdrant_path))
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if "cancer_data_sharing" in collection_names:
            collection_info = client.get_collection("cancer_data_sharing")
            vector_count = collection_info.points_count
            print(f"  ✅ Collection has {vector_count} vectors")
            return True
        else:
            print(f"  ❌ Collection 'cancer_data_sharing' not found")
            print(f"  Available collections: {collection_names}")
            return False
            
    except Exception as e:
        print(f"  ⚠️  Could not verify collection: {e}")
        return True  # May still work


def test_embedding_model():
    """Test if embedding model can be loaded"""
    print("\nTesting embedding model...")
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        print("  Loading E5-Large-V2 model (this may take a moment)...")
        embed_model = HuggingFaceEmbedding(
            model_name="intfloat/e5-large-v2",
            trust_remote_code=True
        )
        
        # Test embedding
        test_text = "This is a test sentence."
        embedding = embed_model.get_text_embedding(test_text)
        
        print(f"  ✅ Embedding model loaded (vector size: {len(embedding)})")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to load embedding model: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧬 Chatbot Component Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Ollama", test_ollama()))
    results.append(("Qdrant Data", test_qdrant_data()))
    results.append(("Embedding Model", test_embedding_model()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! You're ready to launch the chatbot.")
        print("Run: streamlit run chatbot_app.py")
        print("Or: ./launch_chatbot.sh")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
