"""Embedding providers for OpenMemory."""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts. Default: call embed() per text."""
        return [self.embed(text) for text in texts]


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using character n-gram hashing.

    Generates a fixed-size vector by hashing character 3-grams and 4-grams
    of the input text. No external dependencies or API keys required.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _tokenize(self, text: str) -> list[str]:
        """Normalize text and extract character n-grams."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        ngrams: list[str] = []
        # Character 3-grams
        for i in range(len(text) - 2):
            ngrams.append(text[i : i + 3])
        # Character 4-grams
        for i in range(len(text) - 3):
            ngrams.append(text[i : i + 4])
        # Also add whole words for better semantic signal
        words = text.split()
        ngrams.extend(words)
        return ngrams

    def embed(self, text: str) -> list[float]:
        """Generate an embedding using character n-gram hashing."""
        vector = [0.0] * self.dimensions
        ngrams = self._tokenize(text)

        if not ngrams:
            return vector

        for ngram in ngrams:
            h = hashlib.md5(ngram.encode("utf-8")).hexdigest()
            # Use the hash to determine bucket and sign
            bucket = int(h[:8], 16) % self.dimensions
            sign = 1.0 if int(h[8:16], 16) % 2 == 0 else -1.0
            vector[bucket] += sign

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenAI's API.

    Requires the `openai` package and an API key.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAIEmbeddingProvider. "
                "Install it with: pip install openai"
            )

        import os

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.model = model
        self._client = openai.OpenAI(api_key=self.api_key)

    def embed(self, text: str) -> list[float]:
        """Generate an embedding using OpenAI's API."""
        response = self._client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch using OpenAI's API."""
        response = self._client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Uses numpy if available, otherwise falls back to pure Python.
    """
    if len(a) != len(b) or len(a) == 0:
        return 0.0

    try:
        import numpy as np

        va = np.array(a)
        vb = np.array(b)
        dot = np.dot(va, vb)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
    except ImportError:
        # Pure Python fallback
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
