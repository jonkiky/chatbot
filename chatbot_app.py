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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

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
    LLM_MODEL = "llama3.2"
    TOP_K_RESULTS = 5
    TEMPERATURE = 0.7
    CONTEXT_WINDOW = 8192
    

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
        
        # Initialize LLM
        llm = Ollama(
            model=ChatbotConfig.LLM_MODEL,
            temperature=ChatbotConfig.TEMPERATURE,
            request_timeout=120.0,
            context_window=ChatbotConfig.CONTEXT_WINDOW
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
        
        # Create retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=ChatbotConfig.TOP_K_RESULTS,
        )
        
        logger.info("Chatbot components initialized successfully")
        
        return index, retriever, llm, embed_model
        
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")
        st.error(f"Failed to initialize chatbot: {str(e)}")
        return None, None, None, None


def create_chat_engine(index, llm):
    """Create a chat engine with memory"""
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    
    chat_engine = index.as_chat_engine(
        llm=llm,
        chat_mode="context",
        memory=memory,
        system_prompt="""You are an expert assistant for cancer data sharing policies and guidelines at the National Cancer Institute (NCI).

Your role is to help researchers, data managers, and institutional officials understand:
- NCI and NIH data sharing policies and requirements
- Data submission processes and procedures
- Genomic data sharing guidelines
- Access to cancer research datasets and repositories
- Data management and sharing plan (DMSP) requirements
- Privacy, security, and compliance requirements

Guidelines:
1. Provide accurate, clear, and actionable information
2. Cite specific policies, repositories, or resources when relevant
3. If information is not in your knowledge base, say so clearly
4. Use appropriate technical terminology but explain complex concepts
5. Suggest relevant resources or next steps when applicable
6. Be concise but thorough

Always base your answers on the provided context from the NCI data sharing documentation.""",
        verbose=True
    )
    
    return chat_engine


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
        index, retriever, llm, embed_model = initialize_chatbot()
    
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
        
        **Data Source:** NCI Office of Data Sharing (ODS)
        """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize chat engine in session state
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = create_chat_engine(index, llm)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
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
            with st.spinner("Thinking..."):
                try:
                    # Get response from chat engine
                    response = st.session_state.chat_engine.chat(prompt)
                    
                    # Display response
                    st.markdown(response.response)
                    
                    # Get and display sources
                    source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
                    display_sources(source_nodes)
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.response,
                        "sources": source_nodes
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I apologize, but I encountered an error processing your question. Please try again."
                    })


if __name__ == "__main__":
    main()
