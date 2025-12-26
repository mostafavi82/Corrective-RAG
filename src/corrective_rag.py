"""Main Corrective RAG orchestrator."""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
import sys
from dotenv import load_dotenv


def safe_print(text):
    """Print text safely, handling encoding errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))

from .vector_store import VectorStoreManager
from .evaluator import RelevanceEvaluator
from .corrective_paths import CorrectivePaths


class CorrectiveRAG:
    """
    Corrective RAG system with evaluation and corrective paths.

    Workflow:
    1. Retrieve documents from vector store
    2. Evaluate relevance of documents
    3. Take corrective action based on evaluation:
       - Correct: Filter and use relevant documents
       - Incorrect: Perform web search
       - Ambiguous: Hybrid approach (filter + web search)
    4. Generate answer using selected documents
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        evaluator: RelevanceEvaluator,
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        """
        Initialize Corrective RAG system.

        Args:
            vector_store: Vector store manager
            evaluator: Relevance evaluator
            llm_model: OpenAI model name
            temperature: LLM temperature
        """
        load_dotenv()

        self.vector_store = vector_store
        self.evaluator = evaluator
        self.corrective_paths = CorrectivePaths()

        # Initialize LLM with Aval AI
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=os.getenv("AVALAI_API_KEY"),
            openai_api_base=os.getenv("AVALAI_BASE_URL")
        )

        # Create prompt template
        self.prompt = PromptTemplate(
            input_variables=["context", "question", "knowledge_source"],
            template="""You are a helpful assistant. Answer the question based on the provided context.

Context (from {knowledge_source}):
{context}

Question: {question}

Answer: Provide a detailed and accurate answer based on the context. If the context doesn't contain enough information, say so."""
        )

        safe_print("[OK] Corrective RAG system initialized")

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve documents from vector store.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of retrieved documents
        """
        return self.vector_store.retrieve(query, k=k)

    def evaluate(
        self,
        query: str,
        documents: List[Document]
    ) -> tuple:
        """
        Evaluate document relevance and determine path.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            Tuple of (scored_docs, path_type)
        """
        return self.evaluator.evaluate_documents(query, documents)

    def apply_corrective_action(
        self,
        path_type: str,
        query: str,
        scored_docs: List[tuple]
    ) -> tuple:
        """
        Apply corrective action based on path type.

        Args:
            path_type: "correct", "incorrect", or "ambiguous"
            query: User query
            scored_docs: List of (document, score) tuples

        Returns:
            Tuple of (final_documents, knowledge_source)
        """
        return self.corrective_paths.execute_path(path_type, query, scored_docs)

    def generate_answer(
        self,
        query: str,
        documents: List[Document],
        knowledge_source: str
    ) -> str:
        """
        Generate answer using LLM and selected documents.

        Args:
            query: User query
            documents: Selected documents
            knowledge_source: Source of knowledge (for prompt)

        Returns:
            Generated answer
        """
        if not documents:
            return "I couldn't find relevant information to answer your question."

        # Combine document contents
        context = "\n\n".join([
            f"[Document {i+1}]:\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])

        # Format prompt
        formatted_prompt = self.prompt.format(
            context=context,
            question=query,
            knowledge_source=knowledge_source
        )

        # Generate answer
        response = self.llm.invoke(formatted_prompt)
        return response.content

    def query(
        self,
        question: str,
        k: int = 4,
        return_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Complete Corrective RAG pipeline.

        Args:
            question: User question
            k: Number of documents to retrieve
            return_metadata: Whether to return detailed metadata

        Returns:
            Dictionary with answer and optional metadata
        """
        safe_print("\n" + "="*60)
        safe_print(f"Query: {question}")
        safe_print("="*60)

        # Step 1: Retrieve
        safe_print("\n[1/4] Retrieving documents...")
        documents = self.retrieve(question, k=k)
        safe_print(f"[OK] Retrieved {len(documents)} documents")

        # Step 2: Evaluate
        safe_print("\n[2/4] Evaluating relevance...")
        scored_docs, path_type = self.evaluate(question, documents)
        avg_score = sum(score for _, score in scored_docs) / len(scored_docs) if scored_docs else 0
        safe_print(f"[OK] Path determined: {path_type.upper()} (avg score: {avg_score:.3f})")

        # Step 3: Corrective Action
        safe_print(f"\n[3/4] Applying corrective action ({path_type})...")
        final_docs, knowledge_source = self.apply_corrective_action(
            path_type,
            question,
            scored_docs
        )

        # Step 4: Generate Answer
        safe_print("\n[4/4] Generating answer...")
        answer = self.generate_answer(question, final_docs, knowledge_source)
        safe_print("[OK] Answer generated")

        # Prepare response
        response = {"answer": answer}

        if return_metadata:
            response.update({
                "path_type": path_type,
                "knowledge_source": knowledge_source,
                "num_documents": len(final_docs),
                "avg_relevance_score": avg_score,
                "retrieved_docs": len(documents),
                "documents": final_docs
            })

        return response

    def batch_query(
        self,
        questions: List[str],
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.

        Args:
            questions: List of questions
            k: Number of documents to retrieve per query

        Returns:
            List of response dictionaries
        """
        results = []
        for question in questions:
            result = self.query(question, k=k, return_metadata=True)
            results.append(result)

        return results
