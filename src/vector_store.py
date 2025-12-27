"""Vector store management for document retrieval."""

import os
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .avalai_embeddings import AvalAIEmbeddings
from .metis_embeddings import MetisEmbeddings

# Default persist directory
DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")


class VectorStoreManager:
    """Manages document storage and retrieval using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "corrective_rag",
        embedding_model: str = "text-embedding-3-small",
        embedding_provider: str = "metis",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        persist_directory: str = None
    ):
        """
        Initialize vector store manager.

        Args:
            collection_name: Name of the collection in ChromaDB
            embedding_model: Embedding model name
            embedding_provider: "metis", "avalai", or "huggingface"
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            persist_directory: Directory to persist ChromaDB (default: ./chroma_db)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or DEFAULT_PERSIST_DIR

        # Create persist directory if not exists
        os.makedirs(self.persist_directory, exist_ok=True)

        # Choose embedding provider
        if embedding_provider == "metis":
            self.embeddings = MetisEmbeddings(model=embedding_model)
            print(f"[OK] Using Metis AI embeddings: {embedding_model}")
        elif embedding_provider == "avalai":
            self.embeddings = AvalAIEmbeddings(model=embedding_model)
            print(f"[OK] Using Aval AI embeddings: {embedding_model}")
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            print(f"[OK] Using HuggingFace embeddings: {embedding_model}")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vectorstore: Optional[Chroma] = None

        # Try to load existing collection
        self._load_existing()

    def _load_existing(self) -> bool:
        """Try to load existing collection from disk."""
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            # Check if collection has documents
            count = self.vectorstore._collection.count()
            if count > 0:
                print(f"[OK] Loaded existing collection '{self.collection_name}' with {count} chunks from disk")
                return True
            else:
                self.vectorstore = None
                return False
        except Exception:
            self.vectorstore = None
            return False

    def load_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None, clear_existing: bool = True) -> None:
        """
        Load and index documents into vector store.

        Args:
            texts: List of text documents
            metadatas: Optional metadata for each document
            clear_existing: Clear existing collection before loading (default: True)
        """
        # Clear existing collection if requested
        if clear_existing and self.vectorstore:
            try:
                self.vectorstore.delete_collection()
                print(f"[OK] Cleared existing collection '{self.collection_name}'")
            except Exception:
                pass
            self.vectorstore = None

        # Create Document objects
        documents = [
            Document(page_content=text, metadata=metadatas[i] if metadatas else {})
            for i, text in enumerate(texts)
        ]

        # Split documents into chunks
        splits = self.text_splitter.split_documents(documents)

        # Create vector store with persistence
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )

        print(f"[OK] Loaded {len(texts)} documents, created {len(splits)} chunks")
        print(f"[OK] Persisted to: {self.persist_directory}")

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        if self.vectorstore:
            self.vectorstore.delete_collection()
            self.vectorstore = None
            print(f"[OK] Cleared collection '{self.collection_name}'")

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call load_documents first.")

        return self.vectorstore.similarity_search(query, k=k)

    def retrieve_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """
        Retrieve relevant documents with similarity scores.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call load_documents first.")

        return self.vectorstore.similarity_search_with_score(query, k=k)
