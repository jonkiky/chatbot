"""
Upload Local Qdrant Data to Qdrant Cloud

This script:
1. Reads from local Qdrant storage (./qdrant_data)
2. Uploads all vectors and metadata to Qdrant Cloud
3. Uses credentials from .env file
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QdrantCloudUploader:
    """Upload local Qdrant data to Qdrant Cloud"""
    
    def __init__(
        self,
        local_path: str = "./qdrant_data",
        collection_name: str = "cancer_data_sharing",
        cloud_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize uploader
        
        Args:
            local_path: Path to local Qdrant storage
            collection_name: Name of the collection
            cloud_url: Qdrant Cloud URL (from env if not provided)
            api_key: Qdrant Cloud API key (from env if not provided)
        """
        # Load environment variables
        load_dotenv()
        
        self.local_path = local_path
        self.collection_name = collection_name
        
        # Get cloud credentials
        self.cloud_url = cloud_url or os.getenv("QDRANT_HOST")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        if not self.cloud_url or not self.api_key:
            raise ValueError(
                "Cloud credentials not found. "
                "Please set QDRANT_HOST and QDRANT_API_KEY in .env file"
            )
        
        # Clean up URL if needed (remove https:// prefix if present for proper connection)
        if self.cloud_url.startswith("https://"):
            # Extract the host part
            self.cloud_host = self.cloud_url.replace("https://", "").split(":")[0]
            self.cloud_port = 6333
            self.use_https = True
        else:
            self.cloud_host = self.cloud_url
            self.cloud_port = 6333
            self.use_https = True
        
        logger.info(f"Local path: {self.local_path}")
        logger.info(f"Cloud host: {self.cloud_host}")
        logger.info(f"Collection: {self.collection_name}")
        
        # Initialize clients
        self.local_client = None
        self.cloud_client = None
        
    def connect_clients(self):
        """Connect to both local and cloud Qdrant instances"""
        try:
            # Connect to local Qdrant
            logger.info("Connecting to local Qdrant...")
            self.local_client = QdrantClient(path=self.local_path)
            
            # Connect to Qdrant Cloud
            logger.info("Connecting to Qdrant Cloud...")
            self.cloud_client = QdrantClient(
                url=self.cloud_url,
                api_key=self.api_key,
                https=self.use_https
            )
            
            logger.info("Successfully connected to both Qdrant instances")
            
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            raise
    
    def get_collection_info(self):
        """Get information about the local collection"""
        try:
            collection_info = self.local_client.get_collection(self.collection_name)
            logger.info(f"Local collection info: {collection_info}")
            return collection_info
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise
    
    def create_cloud_collection(self, vector_size: int = 1024):
        """
        Create collection in Qdrant Cloud
        
        Args:
            vector_size: Size of vectors (default 1024 for e5-large-v2)
        """
        try:
            # Check if collection already exists
            collections = self.cloud_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                logger.warning(
                    f"Collection '{self.collection_name}' already exists in cloud. "
                    "It will be recreated."
                )
                self.cloud_client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection '{self.collection_name}'")
            
            # Create new collection
            self.cloud_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection '{self.collection_name}' in cloud")
            
            # Create payload indexes for filterable fields
            logger.info("Creating payload indexes...")
            self.cloud_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="category",
                field_schema=PayloadSchemaType.KEYWORD
            )
            self.cloud_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_type",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info("Payload indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating cloud collection: {e}")
            raise
    
    def upload_vectors(self, batch_size: int = 100):
        """
        Upload all vectors from local to cloud
        
        Args:
            batch_size: Number of vectors to upload per batch
        """
        try:
            logger.info("Starting vector upload...")
            
            # Get all points from local collection
            # First, get the total count
            collection_info = self.local_client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            logger.info(f"Total points to upload: {total_points}")
            
            if total_points == 0:
                logger.warning("No points found in local collection")
                return
            
            # Scroll through all points
            offset = None
            uploaded_count = 0
            
            while True:
                # Fetch batch of points
                records, offset = self.local_client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not records:
                    break
                
                # Upload batch to cloud
                self.cloud_client.upsert(
                    collection_name=self.collection_name,
                    points=records
                )
                
                uploaded_count += len(records)
                logger.info(f"Uploaded {uploaded_count}/{total_points} points")
                
                # If offset is None, we've reached the end
                if offset is None:
                    break
            
            logger.info(f"Successfully uploaded {uploaded_count} vectors to cloud")
            
        except Exception as e:
            logger.error(f"Error uploading vectors: {e}")
            raise
    
    def verify_upload(self):
        """Verify that upload was successful"""
        try:
            # Get collection info from both
            local_info = self.local_client.get_collection(self.collection_name)
            cloud_info = self.cloud_client.get_collection(self.collection_name)
            
            local_count = local_info.points_count
            cloud_count = cloud_info.points_count
            
            logger.info(f"Local collection: {local_count} points")
            logger.info(f"Cloud collection: {cloud_count} points")
            
            if local_count == cloud_count:
                logger.info("✓ Upload verification successful! Point counts match.")
                return True
            else:
                logger.warning(
                    f"⚠ Point count mismatch! Local: {local_count}, Cloud: {cloud_count}"
                )
                return False
                
        except Exception as e:
            logger.error(f"Error verifying upload: {e}")
            return False
    
    def run(self):
        """Run the complete upload process"""
        try:
            logger.info("=" * 60)
            logger.info("Starting Qdrant Cloud Upload Process")
            logger.info("=" * 60)
            
            # Step 1: Connect to both instances
            self.connect_clients()
            
            # Step 2: Get local collection info
            collection_info = self.get_collection_info()
            vector_size = collection_info.config.params.vectors.size
            
            # Step 3: Create cloud collection
            self.create_cloud_collection(vector_size=vector_size)
            
            # Step 4: Upload vectors
            self.upload_vectors()
            
            # Step 5: Verify upload
            success = self.verify_upload()
            
            if success:
                logger.info("=" * 60)
                logger.info("Upload completed successfully! ✓")
                logger.info("=" * 60)
                logger.info("\nNext steps:")
                logger.info("1. Update your .env file to use cloud configuration:")
                logger.info(f"   QDRANT_HOST={self.cloud_url}")
                logger.info(f"   QDRANT_API_KEY={self.api_key[:20]}...")
                logger.info("2. Update your application to use QdrantClient with cloud connection")
            else:
                logger.warning("Upload completed with warnings. Please verify manually.")
                
        except Exception as e:
            logger.error(f"Upload process failed: {e}")
            raise


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Upload local Qdrant data to Qdrant Cloud"
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default="./qdrant_data",
        help="Path to local Qdrant storage"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="cancer_data_sharing",
        help="Name of the collection"
    )
    parser.add_argument(
        "--cloud-url",
        type=str,
        help="Qdrant Cloud URL (optional, reads from .env if not provided)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Qdrant Cloud API key (optional, reads from .env if not provided)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of vectors to upload per batch"
    )
    
    args = parser.parse_args()
    
    # Create uploader
    uploader = QdrantCloudUploader(
        local_path=args.local_path,
        collection_name=args.collection_name,
        cloud_url=args.cloud_url,
        api_key=args.api_key
    )
    
    # Run upload process
    uploader.run()


if __name__ == "__main__":
    main()
