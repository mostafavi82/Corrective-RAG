"""Corrective RAG paths: Correct, Incorrect, and Ambiguous."""

import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()


def safe_print(text):
    """Print text safely, handling encoding errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


class CorrectivePaths:
    """Implements the three corrective paths for RAG."""

    def __init__(self):
        """Initialize corrective paths handler."""
        # Initialize Tavily search
        self.web_search_tool = TavilySearchResults(k=3)

        # Initialize question rewriter
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("METIS_API_KEY"),
            openai_api_base=os.getenv("METIS_BASE_URL")
        )

        system = """You are a question re-writer that converts an input question to a better version that is optimized for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""

        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ])

        self.question_rewriter = self.rewrite_prompt | self.llm | StrOutputParser()

        safe_print("[OK] Initialized corrective paths with Tavily search")

    def correct_path(
        self,
        query: str,
        scored_docs: List[Tuple[Document, float]],
        min_score: float = 0.3
    ) -> Tuple[List[Document], str]:
        """
        Correct path: Documents are relevant, filter and use them.

        Args:
            query: Original query
            scored_docs: List of (document, score) tuples
            min_score: Minimum score to keep a document

        Returns:
            Tuple of (filtered documents, knowledge source)
        """
        # Filter documents with score >= min_score
        filtered_docs = [
            doc for doc, score in scored_docs
            if score >= min_score
        ]

        knowledge_source = f"vector_db (filtered {len(filtered_docs)}/{len(scored_docs)} documents)"

        safe_print(f"[OK] Correct path: Filtered {len(filtered_docs)}/{len(scored_docs)} relevant documents")

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
        safe_print(f"[OK] Incorrect path: Performing web search")

        try:
            # Rewrite question for better web search
            safe_print(f"[OK] Rewriting question for web search...")
            rewritten_query = self.question_rewriter.invoke({"question": query})
            safe_print(f"[OK] Rewritten query: {rewritten_query}")

            # Perform web search using Tavily
            results = self.web_search_tool.invoke({"query": rewritten_query})

            safe_print(f"[DEBUG] Tavily results type: {type(results)}")
            safe_print(f"[DEBUG] Tavily results: {results[:500] if isinstance(results, str) else results}")

            # Convert results to documents
            web_docs = []
            for result in results[:max_results]:
                # Handle both dict and string results
                if isinstance(result, dict):
                    content = result.get('content', '')
                    title = result.get('title', '')
                    url = result.get('url', '')
                elif isinstance(result, str):
                    content = result
                    title = ''
                    url = ''
                else:
                    continue

                doc = Document(
                    page_content=content,
                    metadata={
                        'title': title,
                        'url': url,
                        'source': 'web_search'
                    }
                )
                web_docs.append(doc)

            knowledge_source = f"web_search ({len(web_docs)} results)"
            safe_print(f"[OK] Found {len(web_docs)} web results")

            return web_docs, knowledge_source

        except Exception as e:
            safe_print(f"[FAIL] Web search failed: {e}")
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
        safe_print(f"[OK] Ambiguous path: Using hybrid approach")

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

        safe_print(f"[OK] Hybrid: {len(filtered_docs)} filtered docs + {len(web_docs)} web results")

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
