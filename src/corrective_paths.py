"""Corrective RAG paths: Correct, Incorrect, and Ambiguous."""

from typing import List, Tuple
from langchain_core.documents import Document
from duckduckgo_search import DDGS


class CorrectivePaths:
    """Implements the three corrective paths for RAG."""

    def __init__(self):
        """Initialize corrective paths handler."""
        pass

    def correct_path(
        self,
        query: str,
        scored_docs: List[Tuple[Document, float]],
        min_score: float = 0.3
    ) -> Tuple[List[Document], str]:
        """
        Correct path: Documents are relevant, filter and use them.

        Process:
        1. Decompose: Split scored documents
        2. Filter: Keep only highly relevant documents
        3. Recompose: Combine filtered documents

        Args:
            query: Original query
            scored_docs: List of (document, score) tuples
            min_score: Minimum score to keep a document

        Returns:
            Tuple of (filtered documents, knowledge source)
        """
        # DECOMPOSE: Already have individual documents with scores

        # FILTER: Keep documents above threshold
        filtered_docs = [
            doc for doc, score in scored_docs
            if score >= min_score
        ]

        # RECOMPOSE: Documents are ready to use
        knowledge_source = f"vector_db (filtered {len(filtered_docs)}/{len(scored_docs)} documents)"

        print(f"[OK] Correct path: Filtered {len(filtered_docs)}/{len(scored_docs)} relevant documents")

        return filtered_docs, knowledge_source

    def incorrect_path(
        self,
        query: str,
        max_results: int = 3
    ) -> Tuple[List[Document], str]:
        """
        Incorrect path: Documents are not relevant, perform web search.

        Args:
            query: Original query
            max_results: Maximum number of web results

        Returns:
            Tuple of (web search documents, knowledge source)
        """
        print(f"[OK] Incorrect path: Performing web search for: '{query}'")

        try:
            # Perform web search using DuckDuckGo
            ddgs = DDGS()
            results = ddgs.text(query, max_results=max_results)

            # Convert results to documents
            web_docs = []
            for result in results:
                doc = Document(
                    page_content=result.get('body', ''),
                    metadata={
                        'title': result.get('title', ''),
                        'url': result.get('href', ''),
                        'source': 'web_search'
                    }
                )
                web_docs.append(doc)

            knowledge_source = f"web_search ({len(web_docs)} results)"
            print(f"[OK] Found {len(web_docs)} web results")

            return web_docs, knowledge_source

        except Exception as e:
            print(f"[FAIL] Web search failed: {e}")
            return [], "web_search (failed)"

    def ambiguous_path(
        self,
        query: str,
        scored_docs: List[Tuple[Document, float]],
        min_score: float = 0.3,
        max_web_results: int = 2
    ) -> Tuple[List[Document], str]:
        """
        Ambiguous path: Mix of relevant and irrelevant documents.
        Use hybrid approach: filter relevant docs + web search.

        Args:
            query: Original query
            scored_docs: List of (document, score) tuples
            min_score: Minimum score to keep a document
            max_web_results: Maximum number of web results

        Returns:
            Tuple of (hybrid documents, knowledge source)
        """
        print(f"[OK] Ambiguous path: Using hybrid approach")

        # Filter relevant documents from vector DB
        filtered_docs, _ = self.correct_path(query, scored_docs, min_score)

        # Perform web search for additional information
        web_docs, _ = self.incorrect_path(query, max_web_results)

        # Combine both sources
        hybrid_docs = filtered_docs + web_docs

        knowledge_source = (
            f"hybrid (vector_db: {len(filtered_docs)}, "
            f"web_search: {len(web_docs)})"
        )

        print(f"[OK] Hybrid: {len(filtered_docs)} filtered docs + {len(web_docs)} web results")

        return hybrid_docs, knowledge_source

    def execute_path(
        self,
        path_type: str,
        query: str,
        scored_docs: List[Tuple[Document, float]] = None
    ) -> Tuple[List[Document], str]:
        """
        Execute the appropriate corrective path.

        Args:
            path_type: "correct", "incorrect", or "ambiguous"
            query: Original query
            scored_docs: List of (document, score) tuples (required for correct/ambiguous)

        Returns:
            Tuple of (documents, knowledge source)
        """
        if path_type == "correct":
            return self.correct_path(query, scored_docs)
        elif path_type == "incorrect":
            return self.incorrect_path(query)
        elif path_type == "ambiguous":
            return self.ambiguous_path(query, scored_docs)
        else:
            raise ValueError(f"Unknown path type: {path_type}")
