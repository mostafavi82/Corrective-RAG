"""Streamlit app for Corrective RAG with PDF chat."""

import streamlit as st
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from src.vector_store import VectorStoreManager
from src.evaluator import RelevanceEvaluator
from src.corrective_rag import CorrectiveRAG


# Page config
st.set_page_config(
    page_title="Corrective RAG Chat",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .metadata-box {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_pdf(pdf_file):
    """Load PDF file."""
    # Save uploaded file temporarily
    temp_path = f"temp_{pdf_file.name}"
    with open(temp_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader(temp_path)
    pages = loader.load()
    texts = [page.page_content for page in pages]

    # Clean up temp file
    os.remove(temp_path)

    return texts, len(pages)


def initialize_rag():
    """Initialize RAG system."""
    if 'rag_initialized' not in st.session_state:
        with st.spinner('Initializing Corrective RAG system...'):
            vector_store = VectorStoreManager(
                collection_name="streamlit_chat",
                embedding_model="text-embedding-3-small",
                use_avalai=True,
                chunk_size=800,
                chunk_overlap=100
            )

            evaluator = RelevanceEvaluator(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                threshold_correct=0.5,
                threshold_incorrect=0.3
            )

            rag = CorrectiveRAG(
                vector_store=vector_store,
                evaluator=evaluator,
                llm_model="gpt-4o-mini",
                temperature=0.7
            )

            st.session_state.vector_store = vector_store
            st.session_state.rag = rag
            st.session_state.rag_initialized = True


def main():
    """Main Streamlit app."""

    # Header
    st.markdown('<div class="main-header">📚 Corrective RAG Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Chat with your PDF using intelligent retrieval</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # PDF Upload
        st.subheader("1. Upload PDF")
        pdf_file = st.file_uploader("Choose a PDF file", type=['pdf'])

        if pdf_file:
            if 'current_pdf' not in st.session_state or st.session_state.current_pdf != pdf_file.name:
                with st.spinner('Loading PDF...'):
                    texts, num_pages = load_pdf(pdf_file)
                    st.session_state.pdf_texts = texts
                    st.session_state.pdf_pages = num_pages
                    st.session_state.current_pdf = pdf_file.name
                    st.session_state.pdf_loaded = True

                    # Load into vector store
                    initialize_rag()
                    with st.spinner('Indexing document...'):
                        st.session_state.vector_store.load_documents(texts)
                        st.session_state.documents_loaded = True

                st.success(f"✅ Loaded {num_pages} pages")
            else:
                st.info(f"📄 {pdf_file.name} ({st.session_state.pdf_pages} pages)")

        # Advanced settings
        st.subheader("2. Advanced Settings")

        show_metadata = st.checkbox("Show metadata", value=True)
        st.session_state.show_metadata = show_metadata

        k_docs = st.slider("Documents to retrieve", 2, 10, 4)
        st.session_state.k_docs = k_docs

        # Info
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses **Corrective RAG** which:
        - 🟢 **Correct**: Uses filtered documents
        - 🔴 **Incorrect**: Searches the web
        - 🟡 **Ambiguous**: Hybrid approach
        """)

        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Main chat area
    if 'documents_loaded' in st.session_state and st.session_state.documents_loaded:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message["role"] == "assistant" and "metadata" in message:
                    if st.session_state.get('show_metadata', True):
                        metadata = message["metadata"]

                        # Path indicator
                        path = metadata['path_type'].upper()
                        if path == "CORRECT":
                            path_emoji = "🟢"
                            path_color = "#4caf50"
                        elif path == "INCORRECT":
                            path_emoji = "🔴"
                            path_color = "#f44336"
                        else:
                            path_emoji = "🟡"
                            path_color = "#ff9800"

                        st.markdown(f"""
                        <div class="metadata-box">
                            <strong>{path_emoji} Path:</strong> {path}<br>
                            <strong>📊 Relevance:</strong> {metadata['avg_relevance_score']:.3f}<br>
                            <strong>📁 Source:</strong> {metadata['knowledge_source']}<br>
                            <strong>📄 Docs:</strong> {metadata['num_documents']}
                        </div>
                        """, unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Ask a question about the PDF..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.rag.query(
                        prompt,
                        k=st.session_state.get('k_docs', 4),
                        return_metadata=True
                    )

                    st.markdown(result['answer'])

                    if st.session_state.get('show_metadata', True):
                        # Path indicator
                        path = result['path_type'].upper()
                        if path == "CORRECT":
                            path_emoji = "🟢"
                        elif path == "INCORRECT":
                            path_emoji = "🔴"
                        else:
                            path_emoji = "🟡"

                        st.markdown(f"""
                        <div class="metadata-box">
                            <strong>{path_emoji} Path:</strong> {path}<br>
                            <strong>📊 Relevance:</strong> {result['avg_relevance_score']:.3f}<br>
                            <strong>📁 Source:</strong> {result['knowledge_source']}<br>
                            <strong>📄 Docs:</strong> {result['num_documents']}
                        </div>
                        """, unsafe_allow_html=True)

            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": result['answer'],
                "metadata": {
                    "path_type": result['path_type'],
                    "avg_relevance_score": result['avg_relevance_score'],
                    "knowledge_source": result['knowledge_source'],
                    "num_documents": result['num_documents']
                }
            })

    else:
        # Welcome message
        st.info("👈 Please upload a PDF file from the sidebar to start chatting!")

        # Example queries
        st.markdown("### Example Questions")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **About the document:**
            - What is this document about?
            - Summarize the main points
            - What are the key findings?
            """)

        with col2:
            st.markdown("""
            **Specific queries:**
            - What methodology was used?
            - What are the results?
            - What are the conclusions?
            """)


if __name__ == "__main__":
    main()
