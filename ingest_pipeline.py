"""
Data Ingestion Pipeline for AI Chatbot Application
Implements High-Level Data & Indexing Design from System Architecture

This module handles:
- Markdown corpus parsing and chunking
- LLM-assisted chunk classification
- Metadata enrichment
- LlamaIndex node creation
- Vector store indexing
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Chunk Categories as defined in the system design
CHUNK_CATEGORIES = [
    "policy",           # Rules and mandates
    "scope",           # Who/what the policy covers
    "process",         # How to submit, share, or access data
    "technical",       # Formats, metadata, standards
    "privacy_security", # Human data protection, access control
    "costs_funding",   # Budgeting and fees
    "data_reuse",      # How to use/cite shared data
    "compliance",      # Oversight, enforcement, modifications
    "resources",       # Training, guides, templates
    "dataset_access",  # Finding and using datasets/repositories
    "glossary",        # Definitions and terminology
    "faq"             # Frequently asked questions
]


@dataclass
class ChunkMetadata:
    """Metadata structure for document chunks"""
    category: str
    source_file: str
    section_title: str
    funding_filters: List[str]
    subject_filters: List[str]
    repository_tags: List[str]
    policy_requirements: List[str]
    keywords: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MarkdownCorpusParser:
    """Parses unified Markdown corpus and extracts structure"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def find_markdown_files(self) -> List[Path]:
        """Find all Markdown files in the corpus"""
        md_files = []
        for pattern in ['*.md', '**/*.md']:
            md_files.extend(self.data_dir.glob(pattern))
        logger.info(f"Found {len(md_files)} Markdown files")
        return md_files
    
    def parse_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter if present"""
        frontmatter = {}
        body = content
        
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    # Simple key-value parsing (not full YAML)
                    fm_text = parts[1].strip()
                    for line in fm_text.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            frontmatter[key.strip()] = value.strip()
                except Exception as e:
                    logger.warning(f"Failed to parse frontmatter: {e}")
                body = parts[2]
        
        return frontmatter, body
    
    def extract_sections(self, content: str) -> List[Dict[str, str]]:
        """Extract sections based on Markdown headings"""
        sections = []
        current_section = {"title": "Introduction", "content": ""}
        
        lines = content.split('\n')
        for line in lines:
            # Match headings (## or ### preferred for sections)
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                # Save current section
                if current_section["content"].strip():
                    sections.append(current_section)
                # Start new section
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                current_section = {"title": title, "level": level, "content": ""}
            else:
                current_section["content"] += line + "\n"
        
        # Add final section
        if current_section["content"].strip():
            sections.append(current_section)
        
        return sections


class ChunkClassifier:
    """LLM-based classifier for document chunks"""
    
    def __init__(self, llm: Optional[Ollama] = None, use_portkey: bool = False):
        """
        Initialize classifier with LLM
        
        Args:
            llm: Ollama LLM instance (Llama 3.2)
            use_portkey: Whether to route through Portkey.ai (if configured)
        """
        if llm is None:
            # Default Llama 3.2 via Ollama
            self.llm = Ollama(
                model="llama3.2",
                temperature=0.1,
                request_timeout=120.0
            )
        else:
            self.llm = llm
        
        self.classification_prompt_template = """You are a document classifier for cancer research data-sharing policies and guidelines.

Classify the following text chunk into ONE of these categories:
{categories}

Category definitions:
- policy: Rules, mandates, requirements that must be followed
- scope: Who or what a policy covers (types of data, researchers, institutions)
- process: Step-by-step procedures for submitting, sharing, or accessing data
- technical: Data formats, metadata standards, technical specifications
- privacy_security: Human subject protection, consent, access controls, security measures
- costs_funding: Budget requirements, fees, funding sources
- data_reuse: How to properly use, cite, or acknowledge shared data
- compliance: Oversight, enforcement, policy modifications, violations
- resources: Training materials, guides, templates, tools
- dataset_access: How to find, request, or use specific datasets and repositories
- glossary: Definitions, terminology, acronyms
- faq: Common questions and answers

Text chunk:
\"\"\"
{chunk_text}
\"\"\"

Section context: {section_title}
File source: {source_file}

Respond with ONLY the category name (one word), nothing else."""
    
    def classify_chunk(
        self,
        chunk_text: str,
        section_title: str = "",
        source_file: str = ""
    ) -> str:
        """Classify a single chunk using LLM"""
        try:
            prompt = self.classification_prompt_template.format(
                categories=", ".join(CHUNK_CATEGORIES),
                chunk_text=chunk_text[:2000],  # Limit length
                section_title=section_title,
                source_file=source_file
            )
            
            response = self.llm.complete(prompt)
            category = response.text.strip().lower()
            
            # Validate category
            if category not in CHUNK_CATEGORIES:
                logger.warning(f"Invalid category '{category}', defaulting to 'policy'")
                category = "policy"
            
            return category
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "policy"  # Default fallback
    
    def batch_classify_chunks(
        self,
        chunks: List[Dict[str, str]],
        batch_size: int = 10
    ) -> List[str]:
        """Classify multiple chunks with batching for efficiency"""
        categories = []
        for chunk_info in chunks:
            category = self.classify_chunk(
                chunk_text=chunk_info["text"],
                section_title=chunk_info.get("section_title", ""),
                source_file=chunk_info.get("source_file", "")
            )
            categories.append(category)
            logger.debug(f"Classified chunk as: {category}")
        
        return categories


class MetadataEnricher:
    """Enriches chunks with inferred metadata"""
    
    # Keyword-based inference rules
    FUNDING_KEYWORDS = {
        "NIH": ["nih", "national institutes of health", "nih-funded"],
        "NCI": ["nci", "national cancer institute", "nci-funded"],
        "DOD": ["dod", "department of defense", "dod-funded"],
        "NSF": ["nsf", "national science foundation"],
    }
    
    SUBJECT_KEYWORDS = {
        "human": ["human", "patient", "clinical", "participant", "subject"],
        "animal": ["animal", "mouse", "mice", "rat", "model organism"],
        "cell_line": ["cell line", "in vitro", "cultured cells"],
    }
    
    REPOSITORY_KEYWORDS = {
        "dbGaP": ["dbgap", "database of genotypes and phenotypes"],
        "SRA": ["sra", "sequence read archive"],
        "GEO": ["geo", "gene expression omnibus"],
        "PDC": ["pdc", "proteomic data commons"],
        "GDC": ["gdc", "genomic data commons"],
        "CDS": ["cds", "cancer data service"],
    }
    
    POLICY_REQUIREMENT_KEYWORDS = {
        "DMS Plan": ["dms plan", "data management", "sharing plan"],
        "Consent": ["consent", "informed consent", "irb"],
        "Access Control": ["access control", "controlled access", "dbgap"],
        "De-identification": ["de-identif", "anonymiz", "phi"],
        "Genomic Data": ["genomic", "sequencing", "genotype"],
    }
    
    def __init__(self):
        pass
    
    def infer_metadata(self, chunk_text: str, source_file: str = "") -> Dict[str, List[str]]:
        """Infer metadata attributes from chunk text"""
        text_lower = chunk_text.lower()
        
        metadata = {
            "funding_filters": [],
            "subject_filters": [],
            "repository_tags": [],
            "policy_requirements": [],
            "keywords": []
        }
        
        # Check funding sources
        for funder, keywords in self.FUNDING_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                metadata["funding_filters"].append(funder)
        
        # Check subject types
        for subject, keywords in self.SUBJECT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                metadata["subject_filters"].append(subject)
        
        # Check repositories
        for repo, keywords in self.REPOSITORY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                metadata["repository_tags"].append(repo)
        
        # Check policy requirements
        for req, keywords in self.POLICY_REQUIREMENT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                metadata["policy_requirements"].append(req)
        
        # Extract additional keywords (simple approach)
        metadata["keywords"] = self._extract_keywords(chunk_text)
        
        return metadata
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Simple keyword extraction based on frequency"""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'can', 'could', 'may', 'might', 'must', 'shall', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'them', 'their'
        }
        
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]


class IngestionPipeline:
    """Main ingestion pipeline orchestrator"""
    
    def __init__(
        self,
        data_dir: str,
        vector_store_path: str = "./qdrant_data",
        collection_name: str = "cancer_data_sharing",
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        use_portkey: bool = False,
        portkey_api_key: Optional[str] = None
    ):
        """
        Initialize ingestion pipeline
        
        Args:
            data_dir: Path to Markdown corpus
            vector_store_path: Path for Qdrant storage
            collection_name: Name of vector collection
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            use_portkey: Whether to route LLM calls through Portkey
            portkey_api_key: Portkey API key if using gateway
        """
        self.data_dir = data_dir
        self.collection_name = collection_name
        
        # Initialize components
        self.parser = MarkdownCorpusParser(data_dir)
        
        # LLM setup (Llama 3.2 via Ollama)
        if use_portkey and portkey_api_key:
            # Configure Ollama client to use Portkey gateway
            llm = Ollama(
                model="llama3.2",
                base_url="https://api.portkey.ai/v1",
                additional_kwargs={"api_key": portkey_api_key},
                temperature=0.1,
                request_timeout=120.0
            )
        else:
            llm = None  # Use default Ollama
        
        self.classifier = ChunkClassifier(llm=llm, use_portkey=use_portkey)
        self.enricher = MetadataEnricher()
        
        # Text splitter
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Vector store setup
        self.client = QdrantClient(path=vector_store_path)
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name
        )
        
        # Embedding model (Llama 3.2 embeddings via Ollama)
        self.embed_model = OllamaEmbedding(
            model_name="llama3.2",
            base_url="http://localhost:11434"
        )
        
        logger.info("Ingestion pipeline initialized")
    
    def process_document(self, file_path: Path) -> List[TextNode]:
        """Process a single Markdown document into nodes"""
        logger.info(f"Processing: {file_path}")
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse frontmatter and body
        frontmatter, body = self.parser.parse_frontmatter(content)
        
        # Extract sections
        sections = self.parser.extract_sections(body)
        
        # Create chunks with section context
        chunks_with_context = []
        for section in sections:
            section_chunks = self.splitter.split_text(section["content"])
            for chunk in section_chunks:
                chunks_with_context.append({
                    "text": chunk,
                    "section_title": section["title"],
                    "source_file": str(file_path.relative_to(self.data_dir))
                })
        
        logger.info(f"Created {len(chunks_with_context)} chunks from {file_path.name}")
        
        # Classify chunks
        categories = self.classifier.batch_classify_chunks(chunks_with_context)
        
        # Create nodes with metadata
        nodes = []
        for chunk_info, category in zip(chunks_with_context, categories):
            # Enrich metadata
            inferred_metadata = self.enricher.infer_metadata(
                chunk_info["text"],
                chunk_info["source_file"]
            )
            
            # Create full metadata
            metadata = ChunkMetadata(
                category=category,
                source_file=chunk_info["source_file"],
                section_title=chunk_info["section_title"],
                funding_filters=inferred_metadata["funding_filters"],
                subject_filters=inferred_metadata["subject_filters"],
                repository_tags=inferred_metadata["repository_tags"],
                policy_requirements=inferred_metadata["policy_requirements"],
                keywords=inferred_metadata["keywords"]
            )
            
            # Create TextNode
            node = TextNode(
                text=chunk_info["text"],
                metadata=metadata.to_dict()
            )
            nodes.append(node)
        
        return nodes
    
    def run_pipeline(self) -> VectorStoreIndex:
        """Run the complete ingestion pipeline"""
        logger.info("Starting ingestion pipeline")
        
        # Find all Markdown files
        md_files = self.parser.find_markdown_files()
        
        # Process all documents
        all_nodes = []
        for file_path in md_files:
            try:
                nodes = self.process_document(file_path)
                all_nodes.extend(nodes)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Total nodes created: {len(all_nodes)}")
        
        # Create vector index
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        index = VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True
        )
        
        logger.info("Ingestion pipeline completed successfully")
        
        # Log statistics
        self._log_statistics(all_nodes)
        
        return index
    
    def _log_statistics(self, nodes: List[TextNode]):
        """Log statistics about indexed content"""
        category_counts = {}
        for node in nodes:
            category = node.metadata.get("category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
        
        logger.info("=== Indexing Statistics ===")
        logger.info(f"Total nodes: {len(nodes)}")
        logger.info("Nodes by category:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {category}: {count}")


def main():
    """Main entry point for ingestion pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest Markdown corpus into vector database"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to Markdown corpus directory"
    )
    parser.add_argument(
        "--vector-store-path",
        type=str,
        default="./qdrant_data",
        help="Path for Qdrant vector store"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="cancer_data_sharing",
        help="Name of vector collection"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Target chunk size in tokens"
    )
    parser.add_argument(
        "--use-portkey",
        action="store_true",
        help="Route LLM calls through Portkey.ai"
    )
    parser.add_argument(
        "--portkey-api-key",
        type=str,
        help="Portkey API key (if using gateway)"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = IngestionPipeline(
        data_dir=args.data_dir,
        vector_store_path=args.vector_store_path,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        use_portkey=args.use_portkey,
        portkey_api_key=args.portkey_api_key
    )
    
    # Run pipeline
    index = pipeline.run_pipeline()
    
    logger.info("Pipeline execution completed")


if __name__ == "__main__":
    main()
