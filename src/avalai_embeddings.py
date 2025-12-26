"""Aval AI Embeddings integration."""

from typing import List
from langchain.embeddings.base import Embeddings
import os
import requests
from dotenv import load_dotenv


class AvalAIEmbeddings(Embeddings):
    """Aval AI embeddings using OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = None,
        base_url: str = None
    ):
        """
        Initialize Aval AI embeddings.

        Args:
            model: Embedding model name
            api_key: Aval AI API key (or from env)
            base_url: Aval AI base URL (or from env)
        """
        load_dotenv()

        self.model = model
        self.api_key = api_key or os.getenv("AVALAI_API_KEY")
        self.base_url = base_url or os.getenv("AVALAI_BASE_URL")

        if not self.api_key:
            raise ValueError("AVALAI_API_KEY not found in environment variables")
        if not self.base_url:
            raise ValueError("AVALAI_BASE_URL not found in environment variables")

        # Remove trailing slash from base_url
        self.base_url = self.base_url.rstrip('/')
        self.embeddings_url = f"{self.base_url}/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": texts,
            "model": self.model
        }

        try:
            response = requests.post(
                self.embeddings_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings

        except Exception as e:
            print(f"Error calling Aval AI embeddings API: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embed_documents([text])[0]
