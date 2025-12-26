"""Relevance evaluator using fine-tuned transformer model."""

from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document


class RelevanceEvaluator:
    """Evaluates document relevance using a fine-tuned model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        threshold_correct: float = 0.5,
        threshold_incorrect: float = 0.3
    ):
        """
        Initialize relevance evaluator.

        Args:
            model_name: HuggingFace model name (cross-encoder for relevance)
            threshold_correct: Threshold above which documents are considered correct
            threshold_incorrect: Threshold below which documents are considered incorrect
                               Between these thresholds is considered ambiguous
        """
        self.model_name = model_name
        self.threshold_correct = threshold_correct
        self.threshold_incorrect = threshold_incorrect

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        print(f"[OK] Loaded evaluator model: {model_name}")

    def score_relevance(self, query: str, document: str) -> float:
        """
        Score relevance between query and document.

        Args:
            query: User query
            document: Document text

        Returns:
            Relevance score between 0 and 1
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            query,
            document,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            score = torch.sigmoid(logits).item()

        return score

    def evaluate_documents(
        self,
        query: str,
        documents: List[Document]
    ) -> Tuple[List[Tuple[Document, float]], str]:
        """
        Evaluate all documents and determine the path to take.

        Args:
            query: User query
            documents: List of retrieved documents

        Returns:
            Tuple of (list of (document, score) tuples, path_type)
            path_type: "correct", "incorrect", or "ambiguous"
        """
        if not documents:
            return [], "incorrect"

        # Score all documents
        scored_docs = []
        for doc in documents:
            score = self.score_relevance(query, doc.page_content)
            scored_docs.append((doc, score))

        # Calculate average score
        avg_score = sum(score for _, score in scored_docs) / len(scored_docs)

        # Determine path based on average score
        if avg_score >= self.threshold_correct:
            path_type = "correct"
        elif avg_score <= self.threshold_incorrect:
            path_type = "incorrect"
        else:
            path_type = "ambiguous"

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
            threshold: Minimum score to keep (defaults to threshold_incorrect)

        Returns:
            List of relevant documents
        """
        if threshold is None:
            threshold = self.threshold_incorrect

        filtered = [doc for doc, score in scored_docs if score >= threshold]
        return filtered
