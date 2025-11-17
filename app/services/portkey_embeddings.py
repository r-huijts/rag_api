"""
Portkey Embeddings implementation for LangChain.

This module provides a LangChain-compatible embeddings class that uses
Portkey's API endpoint for generating embeddings via HTTP requests.
"""

from typing import List, Optional
import requests
import logging
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class PortkeyEmbeddings(Embeddings):
    """LangChain-compatible embeddings class for Portkey API."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.portkey.ai/v1",
        chunk_size: int = 200,
        virtual_key: Optional[str] = None,
        config: Optional[str] = None,
    ):
        """
        Initialize Portkey embeddings.

        Args:
            model: The model identifier to use for embeddings
            api_key: Portkey API key
            base_url: Base URL for Portkey API (default: https://api.portkey.ai/v1)
            chunk_size: Chunk size for batch requests (default: 200)
            virtual_key: Optional Portkey virtual key
            config: Optional Portkey config ID
        """
        self.model = model
        self.api_key = api_key
        self.virtual_key = virtual_key
        self.base_url = base_url.rstrip("/")
        self.chunk_size = chunk_size
        self.config = config
        self.embeddings_url = f"{self.base_url}/embeddings"

    def _get_headers(self) -> dict:
        """Get headers for Portkey API requests."""
        headers = {
            "Content-Type": "application/json",
            "x-portkey-api-key": self.api_key,
        }
        if self.virtual_key:
            headers["x-portkey-virtual-key"] = self.virtual_key
        if self.config:
            headers["x-portkey-config"] = self.config
        return headers

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """
        Internal method to embed a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        if not texts:
            return []

        # Prepare request payload - input must be a list
        payload = {
            "input": texts,
            "model": self.model,
            "encoding_format": "float",
        }

        try:
            response = requests.post(
                self.embeddings_url,
                headers=self._get_headers(),
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            response_data = response.json()

            # Parse response structure: {"data": [{"embedding": [...]}, ...], ...}
            if "data" not in response_data:
                raise ValueError("Invalid response format: missing 'data' field")

            embeddings = []
            for item in response_data["data"]:
                if "embedding" not in item:
                    raise ValueError("Invalid response format: missing 'embedding' field")
                embeddings.append(item["embedding"])

            # Ensure we have the same number of embeddings as inputs
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Response count mismatch: expected {len(texts)} embeddings, got {len(embeddings)}"
                )

            return embeddings

        except requests.exceptions.RequestException as e:
            logger.error(f"Portkey API request failed: {str(e)}")
            if hasattr(e.response, "text"):
                logger.error(f"Response body: {e.response.text}")
            raise ValueError(f"Failed to get embeddings from Portkey API: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        if not texts:
            return []

        # Handle chunking for large batches
        all_embeddings = []
        for i in range(0, len(texts), self.chunk_size):
            chunk = texts[i : i + self.chunk_size]
            chunk_embeddings = self._embed(chunk)
            all_embeddings.extend(chunk_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as a list of floats
        """
        # For single queries, we still send as a list to match API format
        embeddings = self._embed([text])
        return embeddings[0] if embeddings else []

