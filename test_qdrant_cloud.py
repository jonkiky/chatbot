#!/usr/bin/env python3
"""
Test Qdrant Cloud Connection

This script verifies that your Qdrant Cloud credentials are working correctly.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

def test_cloud_connection():
    """Test connection to Qdrant Cloud"""
    
    # Load environment variables
    load_dotenv()
    
    cloud_url = os.getenv("QDRANT_HOST")
    api_key = os.getenv("QDRANT_API_KEY")
    
    print("=" * 60)
    print("  Qdrant Cloud Connection Test")
    print("=" * 60)
    print()
    
    # Check credentials
    if not cloud_url:
        print("❌ QDRANT_HOST not found in .env file")
        return False
    
    if not api_key:
        print("❌ QDRANT_API_KEY not found in .env file")
        return False
    
    print(f"✓ Cloud URL: {cloud_url}")
    print(f"✓ API Key: {api_key[:20]}..." if len(api_key) > 20 else f"✓ API Key: ***")
    print()
    
    # Test connection
    print("Testing connection...")
    try:
        client = QdrantClient(
            url=cloud_url,
            api_key=api_key,
            https=True,
            timeout=10
        )
        
        # Get collections
        collections = client.get_collections()
        print(f"✓ Connection successful!")
        print()
        
        print(f"Found {len(collections.collections)} collection(s):")
        for collection in collections.collections:
            print(f"  - {collection.name}")
            
            # Get collection info
            try:
                info = client.get_collection(collection.name)
                print(f"    Points: {info.points_count}")
                print(f"    Vector size: {info.config.params.vectors.size}")
            except Exception as e:
                print(f"    (Could not get details: {e})")
        
        print()
        
        # Check for cancer_data_sharing collection
        collection_names = [c.name for c in collections.collections]
        if "cancer_data_sharing" in collection_names:
            print("✓ Target collection 'cancer_data_sharing' found!")
            info = client.get_collection("cancer_data_sharing")
            print(f"  Points: {info.points_count}")
        else:
            print("⚠ Target collection 'cancer_data_sharing' not found")
            print("  You may need to run the upload script first:")
            print("  python upload_to_qdrant_cloud.py")
        
        print()
        print("=" * 60)
        print("  Connection test completed successfully! ✓")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check that QDRANT_HOST is correct in .env")
        print("2. Verify QDRANT_API_KEY is valid")
        print("3. Ensure you have internet connectivity")
        print("4. Check Qdrant Cloud dashboard for service status")
        return False


if __name__ == "__main__":
    success = test_cloud_connection()
    exit(0 if success else 1)
