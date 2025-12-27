"""LLM-based relevance evaluator using batch evaluation."""

import os
import re
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


def safe_print(text):
    """Print text safely, handling encoding errors."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


class LLMRelevanceEvaluator:
    """Evaluates document relevance using LLM with batch evaluation."""

    def __init__(
        self,
        model_name: str = "cf.gemma-2b-it-lora",
        threshold_correct: float = 0.5,
        threshold_incorrect: float = 0.3
    ):
        """
        Initialize LLM-based relevance evaluator.

        Args:
            model_name: OpenAI model name
            threshold_correct: Threshold above which documents are considered correct
            threshold_incorrect: Threshold below which documents are considered incorrect
        """
        load_dotenv()

        self.threshold_correct = threshold_correct
        self.threshold_incorrect = threshold_incorrect

        # Use Aval AI for evaluation
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=os.getenv("AVALAI_API_KEY"),
            openai_api_base=os.getenv("AVALAI_BASE_URL")
        )

        # Create batch evaluation prompt
        self.batch_prompt = ChatPromptTemplate.from_messages([
            ("human", """You are a document retrieval evaluator. Check which documents are relevant to the question.

Question: {question}

Documents:
{documents}

For each document, answer 'yes' if relevant or 'no' if not relevant.
Format your answer as a comma-separated list in order. Example: yes, no, yes, no"""),
        ])

        # Create the chain with string output parser
        self.batch_grader = self.batch_prompt | self.llm | StrOutputParser()

        safe_print(f"[OK] Loaded LLM evaluator: {model_name}")

    def _parse_batch_response(self, response: str, num_docs: int) -> List[float]:
        """Parse batch LLM response to extract yes/no answers."""
        response_lower = response.lower().strip()

        # Try to extract yes/no patterns
        # Match patterns like "yes", "no", "1", "0"
        pattern = r'\b(yes|no)\b'
        matches = re.findall(pattern, response_lower)

        scores = []
        for i, match in enumerate(matches[:num_docs]):
            if match == 'yes':
                scores.append(1.0)
            else:
                scores.append(0.0)

        # If we didn't get enough matches, fill with 0.5 (ambiguous)
        while len(scores) < num_docs:
            scores.append(0.5)

        return scores

    def evaluate_documents(
        self,
        query: str,
        documents: List[Document]
    ) -> Tuple[List[Tuple[Document, float]], str]:
        """
        Evaluate all documents in a single LLM call.

        Args:
            query: User query
            documents: List of retrieved documents

        Returns:
            Tuple of (list of (document, score) tuples, path_type)
            path_type: "correct", "incorrect", or "ambiguous"
        """
        if not documents:
            return [], "incorrect"

        try:
            # Format all documents for batch evaluation
            docs_text = ""
            for i, doc in enumerate(documents):
                # Truncate each document to avoid token limits
                content = doc.page_content[:500]
                docs_text += f"\n[Document {i+1}]:\n{content}\n"

            # Single LLM call for all documents
            response = self.batch_grader.invoke({
                "question": query,
                "documents": docs_text
            })

            safe_print(f"[DEBUG] Batch LLM response: {response}")

            # Parse the response
            scores = self._parse_batch_response(response, len(documents))

            # Create scored docs list
            scored_docs = list(zip(documents, scores))
            relevant_count = sum(1 for s in scores if s >= 0.5)

        except Exception as e:
            safe_print(f"[WARN] Batch LLM scoring failed: {e}")
            # Default all to ambiguous
            scored_docs = [(doc, 0.5) for doc in documents]
            relevant_count = len(documents)  # Treat as ambiguous

        # Calculate relevance ratio
        relevance_ratio = relevant_count / len(documents)

        # Determine path based on how many documents are relevant
        if relevance_ratio >= self.threshold_correct:
            path_type = "correct"
        elif relevance_ratio <= self.threshold_incorrect:
            path_type = "incorrect"
        else:
            path_type = "ambiguous"

        safe_print(f"[DEBUG] Relevant: {relevant_count}/{len(documents)} = {relevance_ratio:.2f}")

        return scored_docs, path_type

    def filter_relevant_documents(
        self,
        scored_docs: List[Tuple[Document, float]],
        threshold: float = None
    ) -> List[Document]:
        """
        Filter documents based on relevance score.

        Args:
            scored_docs: List of (document, score) tuples
            threshold: Minimum score to keep (defaults to 0.5)

        Returns:
            List of relevant documents
        """
        if threshold is None:
            threshold = 0.5

        filtered = [doc for doc, score in scored_docs if score >= threshold]
        return filtered
