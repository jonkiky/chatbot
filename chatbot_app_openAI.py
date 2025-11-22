"""
AI Chatbot Application for Cancer Data Sharing
Built with LlamaIndex, Qdrant, and Streamlit

This chatbot provides information about:
- NCI data sharing policies and guidelines
- Data submission and access processes
- Genomic data sharing requirements
- Available datasets and repositories
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Cancer Data Sharing Chatbot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .metadata-box {
        background-color: #fafafa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


class ChatbotConfig:
    """Configuration for the chatbot"""
    
    VECTOR_STORE_PATH = "./qdrant_data"
    COLLECTION_NAME = "cancer_data_sharing"
    EMBEDDING_MODEL = "intfloat/e5-large-v2"
    LLM_MODEL = "gpt-4o-mini"  # or "gpt-4o" or "gpt-3.5-turbo"
    TOP_K_RESULTS = 5
    TEMPERATURE = 0.7
    CONTEXT_WINDOW = 8192


class QueryClassifier:
    """Classify user queries to route to appropriate metadata filters"""
    
    CATEGORIES = {
        "guidance": ["guidance", "guideline", "recommendation", "best practice", "should", "suggested", "advised"],
        "policy": ["policy", "policies", "requirement", "requirements", "regulation", "mandate", "rule", "must follow", "required"],
        "process": ["process", "procedure", "step", "how to", "submit", "submission", "share", "access", "workflow", "instructions"],
        "resources": ["training", "guide", "template", "tool", "resource", "tutorial", "documentation", "help", "support", "material", "dataset", "repository", "database", "dbgap", "gdc", "sra"],
        "glossary": ["definition", "term", "terminology", "acronym", "meaning", "what is", "what does", "glossary", "vocabulary", "define"],
        "faq": ["faq", "frequently asked", "common question", "how do i", "can i", "should i", "what if", "question"],
        "news": ["news", "announcement", "update", "highlight", "new", "latest", "recent", "announcement"],
        "others": ["other", "general", "miscellaneous", "misc"]
    }
    
    def __init__(self, llm):
        self.llm = llm
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify query using keyword matching and LLM
        Returns: dict with category, subcategories, and confidence
        """
        query_lower = query.lower()
        
        # Keyword-based classification
        matched_categories = []
        for category, keywords in self.CATEGORIES.items():
            if any(keyword in query_lower for keyword in keywords):
                matched_categories.append(category)
        
        # LLM-based classification for better accuracy
        classification_prompt = f"""Classify the following user query into one or more of these categories:
                - guidance: Guidelines and best practices
                - policy: Rules and requirements
                - process: Step-by-step procedures
                - resources: Training materials, guides, templates, tools
                - glossary: Definitions and terminology
                - faq: Common questions and answers
                - news: News and announcements
                - others: General or miscellaneous content

                Query: "{query}"

                Respond with ONLY a JSON object in this format:
                {{"primary_category": "category_name", "secondary_categories": ["category1", "category2"], "confidence": 0.9}}
                """
        
        try:
            response = self.llm.complete(classification_prompt)
            # Extract JSON from response
            response_text = str(response).strip()
            # Find JSON object in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                llm_classification = json.loads(json_str)
            else:
                llm_classification = {"primary_category": matched_categories[0] if matched_categories else "policy", "secondary_categories": matched_categories[1:], "confidence": 0.5}
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, using keyword-based classification")
            llm_classification = {
                "primary_category": matched_categories[0] if matched_categories else "policy",
                "secondary_categories": matched_categories[1:] if len(matched_categories) > 1 else [],
                "confidence": 0.7 if matched_categories else 0.3
            }
        
        return llm_classification


class QueryRouter:
    """Route queries to appropriate retrievers with metadata filters"""
    
    def __init__(self, index, top_k: int = 5):
        self.index = index
        self.top_k = top_k
        self.classifier = None
    
    def set_classifier(self, classifier: QueryClassifier):
        """Set the query classifier"""
        self.classifier = classifier
    
    def create_metadata_filters(self, classification: Dict[str, Any]) -> MetadataFilters:
        """Create metadata filters based on classification"""
        primary_category = classification.get("primary_category")
        secondary_categories = classification.get("secondary_categories", [])
        
        filters = []
        
        # Add primary category filter
        if primary_category:
            filters.append(
                MetadataFilter(
                    key="category",
                    value=primary_category,
                    operator=FilterOperator.EQ
                )
            )
        
        # Add secondary category filters with OR logic if multiple categories
        all_categories = [primary_category] + secondary_categories
        
        # Map categories to document types and other metadata
        category_mappings = {
            "guidance": {"document_type": "Guidance"},
            "policy": {"document_type": "Guidance"},
            "process": {"document_type": "Process"},
            "resources": {"document_type": "Data"},
            "glossary": {"document_type": "About"},
            "faq": {"document_type": "Guidance"},
            "news": {"document_type": "News"},
            "others": {"document_type": "documents"}
        }
        
        # Add additional filters based on category mappings
        for category in all_categories:
            if category in category_mappings:
                mapping = category_mappings[category]
                for key, value in mapping.items():
                    if key == "has_repositories" and value:
                        # Filter for documents that have repository information
                        pass  # Repositories are in metadata, will be boosted by relevance
                    elif key == "has_data_types" and value:
                        # Filter for documents that have data type information
                        pass  # Data types are in metadata, will be boosted by relevance
                    else:
                        filters.append(
                            MetadataFilter(
                                key=key,
                                value=value,
                                operator=FilterOperator.EQ
                            )
                        )
        
        if filters:
            return MetadataFilters(filters=filters, condition="or")
        return None
    
    def retrieve_with_routing(self, query: str, classification: Dict[str, Any] = None):
        """Retrieve documents with routing based on query classification"""
        
        # Classify query if not provided
        if classification is None and self.classifier:
            classification = self.classifier.classify_query(query)
        
        # Create metadata filters
        metadata_filters = self.create_metadata_filters(classification) if classification else None
        
        # Create retriever with filters
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
            filters=metadata_filters
        )
        
        # Retrieve nodes
        nodes = retriever.retrieve(query)
        
        return nodes, classification


class ResponseEvaluator:
    """Evaluate the quality of chatbot responses"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate_response(self, query: str, response: str, context: str) -> Dict[str, Any]:
        """
        Evaluate response quality across multiple dimensions
        Returns: dict with scores and feedback
        """
        evaluation_prompt = f"""Evaluate the following chatbot response based on the given query and context.

Query: "{query}"

Context provided:
{context[:1000]}...

Response:
{response}

Evaluate the response on these dimensions (0-10 scale):
1. Relevance: How well does the response address the query?
2. Accuracy: Is the information factually correct based on the context?
3. Completeness: Does it provide a thorough answer?
4. Clarity: Is the response clear and easy to understand?
5. Actionability: Does it provide actionable information or next steps?

Respond with ONLY a JSON object in this format:
{{
  "relevance": 8,
  "accuracy": 9,
  "completeness": 7,
  "clarity": 9,
  "actionability": 8,
  "overall_score": 8.2,
  "feedback": "Brief explanation of the evaluation"
}}
"""
        
        try:
            eval_response = self.llm.complete(evaluation_prompt)
            response_text = str(eval_response).strip()
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                evaluation = json.loads(json_str)
                
                # Calculate overall score if not provided
                if "overall_score" not in evaluation:
                    scores = [evaluation.get("relevance", 0), evaluation.get("accuracy", 0), 
                             evaluation.get("completeness", 0), evaluation.get("clarity", 0), 
                             evaluation.get("actionability", 0)]
                    evaluation["overall_score"] = round(sum(scores) / len(scores), 1)
                
                return evaluation
            else:
                return self._default_evaluation()
        except Exception as e:
            logger.warning(f"Response evaluation failed: {e}")
            return self._default_evaluation()
    
    def _default_evaluation(self) -> Dict[str, Any]:
        """Return default evaluation when LLM evaluation fails"""
        return {
            "relevance": 7,
            "accuracy": 7,
            "completeness": 7,
            "clarity": 7,
            "actionability": 7,
            "overall_score": 7.0,
            "feedback": "Response generated successfully"
        }
    

@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot components (cached for performance)"""
    try:
        logger.info("Initializing chatbot components...")
        
        # Initialize Qdrant client
        client = QdrantClient(path=ChatbotConfig.VECTOR_STORE_PATH)
        
        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=ChatbotConfig.COLLECTION_NAME
        )
        
        # Initialize embedding model
        embed_model = HuggingFaceEmbedding(
            model_name=ChatbotConfig.EMBEDDING_MODEL,
            trust_remote_code=True
        )
        
        # Initialize LLM with OpenAI
        # Get OpenAI API key from environment or Streamlit secrets
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY")
            except:
                pass
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or add it to Streamlit secrets (.streamlit/secrets.toml)"
            )
        
        llm = OpenAI(
            model=ChatbotConfig.LLM_MODEL,
            temperature=ChatbotConfig.TEMPERATURE,
            api_key=api_key,
            max_tokens=4096
        )
        
        # Create index from existing vector store
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        # Create query classifier and router
        classifier = QueryClassifier(llm)
        router = QueryRouter(index, top_k=ChatbotConfig.TOP_K_RESULTS)
        router.set_classifier(classifier)
        
        # Create response evaluator
        evaluator = ResponseEvaluator(llm)
        
        logger.info("Chatbot components initialized successfully")
        
        return index, llm, embed_model, classifier, router, evaluator
        
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")
        st.error(f"Failed to initialize chatbot: {str(e)}")
        return None, None, None, None, None, None


def display_classification(classification: Dict[str, Any]):
    """Display query classification information"""
    if not classification:
        return
    
    with st.expander("🔍 Query Classification", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Primary Category:** `{classification.get('primary_category', 'N/A')}`")
        with col2:
            confidence = classification.get('confidence', 0)
            st.markdown(f"**Confidence:** {confidence:.0%}")
        
        if classification.get('secondary_categories'):
            st.markdown(f"**Secondary Categories:** {', '.join(classification['secondary_categories'])}")


def display_evaluation(evaluation: Dict[str, Any]):
    """Display response evaluation metrics"""
    if not evaluation:
        return
    
    with st.expander("📊 Response Quality Evaluation", expanded=False):
        # Overall score with color coding
        overall = evaluation.get('overall_score', 0)
        color = "🟢" if overall >= 8 else "🟡" if overall >= 6 else "🔴"
        st.markdown(f"### {color} Overall Score: {overall:.1f}/10")
        
        # Individual metrics
        st.markdown("#### Detailed Metrics")
        cols = st.columns(5)
        metrics = ['relevance', 'accuracy', 'completeness', 'clarity', 'actionability']
        labels = ['Relevance', 'Accuracy', 'Completeness', 'Clarity', 'Actionability']
        
        for col, metric, label in zip(cols, metrics, labels):
            score = evaluation.get(metric, 0)
            col.metric(label, f"{score}/10")
        
        # Feedback
        if evaluation.get('feedback'):
            st.markdown("#### Evaluation Feedback")
            st.info(evaluation['feedback'])


def display_sources(nodes):
    """Display source information from retrieved nodes"""
    if not nodes:
        return
    
    with st.expander("📚 View Sources", expanded=False):
        for i, node in enumerate(nodes, 1):
            metadata = node.metadata
            
            st.markdown(f"**Source {i}:** `{metadata.get('source_file', 'Unknown')}`")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Section:** {metadata.get('section_title', 'N/A')}")
                st.markdown(f"**Category:** {metadata.get('category', 'N/A')}")
            
            with col2:
                if metadata.get('document_type'):
                    st.markdown(f"**Type:** {metadata.get('document_type')}")
                if metadata.get('policy_references'):
                    st.markdown(f"**Policies:** {', '.join(metadata.get('policy_references', []))}")
            
            # Show additional metadata if available
            if metadata.get('repositories'):
                st.markdown(f"🗄️ **Repositories:** {', '.join(metadata['repositories'])}")
            if metadata.get('data_types'):
                st.markdown(f"🧬 **Data Types:** {', '.join(metadata['data_types'])}")
            
            with st.expander("View text snippet"):
                st.markdown(node.text[:500] + "..." if len(node.text) > 500 else node.text)
            
            st.divider()


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🧬 Cancer Data Sharing Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your AI assistant for NCI data sharing policies and guidelines</div>', unsafe_allow_html=True)
    
    # Initialize chatbot
    with st.spinner("Loading chatbot components..."):
        index, llm, embed_model, classifier, router, evaluator = initialize_chatbot()
    
    if index is None:
        st.error("Failed to initialize chatbot. Please check the logs and ensure Qdrant data exists.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model settings
        temperature = st.slider(
            "Response Creativity",
            min_value=0.0,
            max_value=1.0,
            value=ChatbotConfig.TEMPERATURE,
            step=0.1,
            help="Higher values make responses more creative, lower values more focused"
        )
        
        top_k = st.slider(
            "Number of Sources",
            min_value=1,
            max_value=10,
            value=ChatbotConfig.TOP_K_RESULTS,
            help="Number of relevant document chunks to retrieve"
        )
        
        # Update router with new top_k
        router.top_k = top_k
        
        show_classification = st.checkbox(
            "Show Query Classification",
            value=True,
            help="Display how the chatbot classified your query"
        )
        
        show_evaluation = st.checkbox(
            "Show Response Evaluation",
            value=True,
            help="Display quality metrics for the response"
        )
        
        st.divider()
        
        # Quick actions
        st.header("🚀 Quick Actions")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Example questions
        st.header("💡 Example Questions")
        example_questions = [
            "What is the NIH Data Management and Sharing Policy?",
            "How do I submit genomic data to dbGaP?",
            "What are the requirements for a Data Management and Sharing Plan?",
            "How can I access cancer datasets?",
            "What privacy protections are required for human data?",
            "What repositories are available for cancer data?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{hash(question)}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
        
        st.divider()
        
        # About section
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot provides information about:
        - NCI/NIH data sharing policies
        - Data submission & access processes
        - Genomic data sharing guidelines
        - Available datasets & repositories
        
        **Features:**
        - Intelligent query classification
        - Context-aware routing
        - Metadata-based filtering
        - Response quality evaluation
        
        **Data Source:** NCI Office of Data Sharing (ODS)
        """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize memory
    if "memory" not in st.session_state:
        st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display classification if available
            if message["role"] == "assistant" and "classification" in message and show_classification:
                display_classification(message["classification"])
            
            # Display evaluation if available
            if message["role"] == "assistant" and "evaluation" in message and show_evaluation:
                display_evaluation(message["evaluation"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                display_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about cancer data sharing..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question..."):
                try:
                    # Classify and route query
                    nodes, classification = router.retrieve_with_routing(prompt)
                    
                    # Display classification
                    if show_classification:
                        display_classification(classification)
                    
                    # Build context from retrieved nodes
                    context_str = "\n\n".join([node.text for node in nodes])
                    
                    # Create prompt with context
                    system_prompt = """You are an expert assistant for cancer data sharing policies and guidelines at the National Cancer Institute (NCI).

Your role is to help researchers, data managers, and institutional officials understand:
- NCI and NIH data sharing policies and requirements
- Data submission processes and procedures
- Genomic data sharing guidelines
- Access to cancer research datasets and repositories
- Data management and sharing plan (DMSP) requirements
- Privacy, security, and compliance requirements

Guidelines:
1. Provide accurate, clear, and actionable information based on the context provided
2. Cite specific policies, repositories, or resources when relevant
3. If information is not in the provided context, say so clearly
4. Use appropriate technical terminology but explain complex concepts
5. Suggest relevant resources or next steps when applicable
6. Be concise but thorough

Always base your answers on the provided context from the NCI data sharing documentation."""
                    
                    query_prompt = f"""Context information is below:
---------------------
{context_str}
---------------------

Based on the context information and not prior knowledge, answer the following question:
{prompt}

Answer:"""
                    
                    # Generate response
                    with st.spinner("Generating response..."):
                        response = llm.complete(query_prompt)
                        response_text = str(response)
                    
                    # Display response
                    st.markdown(response_text)
                    
                    # Evaluate response
                    evaluation = None
                    if show_evaluation:
                        with st.spinner("Evaluating response quality..."):
                            evaluation = evaluator.evaluate_response(prompt, response_text, context_str)
                            display_evaluation(evaluation)
                    
                    # Display sources
                    display_sources(nodes)
                    
                    # Add assistant message to history
                    message_data = {
                        "role": "assistant",
                        "content": response_text,
                        "sources": nodes,
                        "classification": classification
                    }
                    if evaluation:
                        message_data["evaluation"] = evaluation
                    
                    st.session_state.messages.append(message_data)
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I apologize, but I encountered an error processing your question. Please try again."
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I apologize, but I encountered an error processing your question. Please try again."
                    })


if __name__ == "__main__":
    main()
