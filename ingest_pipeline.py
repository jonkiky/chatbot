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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
    """Enhanced metadata structure for document chunks based on corpus analysis"""
    
    # Core identifiers
    chunk_id: str
    source_file: str
    section_title: str
    document_type: str  # About, Data, Guidance, Process, News
    
    # Content classification
    category: str  # Primary category (policy, scope, process, etc.)
    subcategory: Optional[str] = None
    
    # Contextual attributes
    agencies: List[str] = None
    funding_sources: List[str] = None
    repositories: List[str] = None
    data_types: List[str] = None
    subject_types: List[str] = None
    
    # Policy context
    policy_references: List[str] = None
    requirements: List[str] = None
    compliance_level: Optional[str] = None
    
    # Process metadata
    process_stage: Optional[str] = None
    audience: List[str] = None
    
    # Semantic enrichment
    keywords: List[str] = None
    related_topics: List[str] = None
    
    # Quality metrics
    confidence_score: float = 0.0
    
    def __post_init__(self):
        """Initialize None fields as empty lists"""
        for field_name in ['agencies', 'funding_sources', 'repositories', 'data_types', 
                           'subject_types', 'policy_references', 'requirements', 
                           'audience', 'keywords', 'related_topics']:
            if getattr(self, field_name) is None:
                setattr(self, field_name, [])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values and empty lists"""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None and v != [] and v != 0.0}


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
    
    def __init__(self, llm: Optional[Ollama] = None):
        """
        Initialize classifier with LLM
        
        Args:
            llm: Ollama LLM instance (Llama 3.2)
        """
        if llm is None:
            # Default Llama 3.2 via Ollama
            self.llm = Ollama(
                model="llama3.2",
                temperature=0.1,
                request_timeout=120.0,
                context_window=8192
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
    """Enhanced metadata enricher based on corpus analysis"""
    
    # Agency patterns
    AGENCY_PATTERNS = {
        "NIH": ["nih", "national institutes of health"],
        "NCI": ["nci", "national cancer institute"],
        "FDA": ["fda", "food and drug administration"],
        "CDC": ["cdc", "centers for disease control"],
        "NSF": ["nsf", "national science foundation"],
        "DOD": ["dod", "department of defense"],
        "OSTP": ["ostp", "office of science and technology policy"],
        "OMB": ["omb", "office of management and budget"],
    }
    
    # Repository patterns
    REPOSITORY_PATTERNS = {
        "dbGaP": ["dbgap", "database of genotypes and phenotypes"],
        "GEO": ["geo", "gene expression omnibus"],
        "SRA": ["sra", "sequence read archive"],
        "GDC": ["gdc", "genomic data commons"],
        "PDC": ["pdc", "proteomic data commons"],
        "CDS": ["cds", "cancer data service"],
        "IDC": ["idc", "imaging data commons"],
    }
    
    # Data type patterns
    DATA_TYPE_PATTERNS = {
        "genomic": ["genomic", "genome", "dna", "sequencing", "whole genome", "exome"],
        "proteomic": ["proteomic", "protein", "proteome"],
        "transcriptomic": ["transcriptomic", "rna", "rna-seq", "transcriptome"],
        "imaging": ["imaging", "radiology", "mri", "ct scan", "pathology image"],
        "clinical": ["clinical", "patient data", "medical record", "ehr"],
        "metabolomic": ["metabolomic", "metabolite", "metabolome"],
        "single_cell": ["single-cell", "single cell", "sc-rna", "scrna"],
        "spatial": ["spatial", "spatial transcriptomic", "spatial genomic"],
    }
    
    # Subject type patterns
    SUBJECT_TYPE_PATTERNS = {
        "human": ["human", "patient", "participant", "clinical trial", "subject"],
        "animal": ["animal", "mouse", "mice", "rat", "model organism", "preclinical"],
        "cell_line": ["cell line", "in vitro", "cultured cell", "cell culture"],
        "tissue": ["tissue", "tissue sample", "biopsy", "specimen"],
    }
    
    # Policy reference patterns
    POLICY_PATTERNS = {
        "NIH_DMS_Policy": ["nih data management", "nih dms policy", "nih policy for data management"],
        "GDSP": ["gdsp", "genomic data sharing policy", "nih genomic data sharing"],
        "GDS_Policy": ["gds policy", "genomic data sharing"],
        "DMSP_Requirement": ["dms plan", "dmsp", "data management and sharing plan"],
    }
    
    # Requirement patterns
    REQUIREMENT_PATTERNS = {
        "DMS_Plan": ["dms plan", "data management", "sharing plan", "dmsp"],
        "Consent": ["consent", "informed consent", "patient consent"],
        "IRB_Approval": ["irb", "institutional review board", "ethics approval"],
        "De_identification": ["de-identif", "anonymiz", "phi", "protected health information"],
        "Access_Control": ["access control", "controlled access", "access restriction"],
        "Metadata_Standards": ["metadata", "metadata standard", "data dictionary"],
        "Data_Format": ["data format", "file format", "format requirement"],
    }
    
    # Process stage patterns
    PROCESS_STAGE_PATTERNS = {
        "submission": ["submit", "submission", "upload", "deposit"],
        "access": ["access", "download", "retrieve", "request data"],
        "review": ["review", "approval", "evaluation"],
        "registration": ["register", "registration", "account creation"],
    }
    
    # Audience patterns
    AUDIENCE_PATTERNS = {
        "investigator": ["investigator", "researcher", "pi", "principal investigator"],
        "data_manager": ["data manager", "data steward", "data coordinator"],
        "institutional_official": ["institutional official", "signing official", "so"],
        "irb": ["irb", "institutional review board", "ethics committee"],
        "data_user": ["data user", "data requester", "secondary researcher"],
    }
    
    # Compliance level patterns
    COMPLIANCE_PATTERNS = {
        "mandatory": ["must", "required", "shall", "mandatory", "obligation"],
        "recommended": ["should", "recommended", "encouraged", "best practice"],
        "optional": ["may", "optional", "can", "at discretion"],
    }
    
    def __init__(self):
        pass
    
    def infer_metadata(
        self,
        chunk_text: str,
        source_file: str = "",
        section_title: str = ""
    ) -> Dict[str, Any]:
        """Infer comprehensive metadata from chunk text"""
        text_lower = chunk_text.lower()
        
        metadata = {
            "agencies": self._extract_patterns(text_lower, self.AGENCY_PATTERNS),
            "repositories": self._extract_patterns(text_lower, self.REPOSITORY_PATTERNS),
            "data_types": self._extract_patterns(text_lower, self.DATA_TYPE_PATTERNS),
            "subject_types": self._extract_patterns(text_lower, self.SUBJECT_TYPE_PATTERNS),
            "policy_references": self._extract_patterns(text_lower, self.POLICY_PATTERNS),
            "requirements": self._extract_patterns(text_lower, self.REQUIREMENT_PATTERNS),
            "process_stage": self._extract_single_pattern(text_lower, self.PROCESS_STAGE_PATTERNS),
            "audience": self._extract_patterns(text_lower, self.AUDIENCE_PATTERNS),
            "compliance_level": self._extract_single_pattern(text_lower, self.COMPLIANCE_PATTERNS),
            "keywords": self._extract_keywords(chunk_text),
            "document_type": self._infer_document_type(source_file),
        }
        
        return metadata
    
    def _extract_patterns(
        self,
        text: str,
        pattern_dict: Dict[str, List[str]]
    ) -> List[str]:
        """Extract all matching patterns"""
        matches = []
        for key, patterns in pattern_dict.items():
            if any(pattern in text for pattern in patterns):
                matches.append(key)
        return matches
    
    def _extract_single_pattern(
        self,
        text: str,
        pattern_dict: Dict[str, List[str]]
    ) -> Optional[str]:
        """Extract single best matching pattern"""
        for key, patterns in pattern_dict.items():
            if any(pattern in text for pattern in patterns):
                return key
        return None
    
    def _infer_document_type(self, source_file: str) -> str:
        """Infer document type from file path"""
        parts = Path(source_file).parts
        if len(parts) > 0:
            first_dir = parts[0]
            if first_dir in ["About", "Data", "Guidance", "Process", "News", "documents"]:
                return first_dir
        return "unknown"
    
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
        chunk_overlap: int = 200
    ):
        """
        Initialize ingestion pipeline
        
        Args:
            data_dir: Path to Markdown corpus
            vector_store_path: Path for Qdrant storage
            collection_name: Name of vector collection
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
        """
        self.data_dir = data_dir
        self.collection_name = collection_name
        
        # Initialize components
        self.parser = MarkdownCorpusParser(data_dir)
        
        # LLM setup (Llama 3.2 via Ollama)
        self.classifier = ChunkClassifier()
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
        
        # Embedding model (E5-Large-V2 from HuggingFace)
        self.embed_model = HuggingFaceEmbedding(
            model_name="intfloat/e5-large-v2",
            trust_remote_code=True
        )
        
        logger.info("Ingestion pipeline initialized")
    
   # Update process_document method (lines 420-480)

    def process_document(self, file_path: Path) -> List[TextNode]:
        """Process a single Markdown document into nodes with enhanced metadata"""
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
        
        # Create nodes with enhanced metadata
        nodes = []
        for idx, (chunk_info, category) in enumerate(zip(chunks_with_context, categories)):
            # Enrich metadata
            inferred_metadata = self.enricher.infer_metadata(
                chunk_info["text"],
                chunk_info["source_file"],
                chunk_info["section_title"]
            )
            
            # Generate chunk ID
            chunk_id = f"{file_path.stem}_{idx:04d}"
            
            # Create full metadata with enhanced structure
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                source_file=chunk_info["source_file"],
                section_title=chunk_info["section_title"],
                document_type=inferred_metadata["document_type"],
                category=category,
                agencies=inferred_metadata["agencies"],
                repositories=inferred_metadata["repositories"],
                data_types=inferred_metadata["data_types"],
                subject_types=inferred_metadata["subject_types"],
                policy_references=inferred_metadata["policy_references"],
                requirements=inferred_metadata["requirements"],
                compliance_level=inferred_metadata["compliance_level"],
                process_stage=inferred_metadata["process_stage"],
                audience=inferred_metadata["audience"],
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
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = IngestionPipeline(
        data_dir=args.data_dir,
        vector_store_path=args.vector_store_path,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size
    )
    
    # Run pipeline
    index = pipeline.run_pipeline()
    
    logger.info("Pipeline execution completed")


if __name__ == "__main__":
    main()
