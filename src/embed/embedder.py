"""
Pluggable embedding backend.

All backends return L2-normalised float vectors of length settings.embed_dim.
Switch backends by setting SEC_EMBEDDER in .env — no code changes needed.

Backends
--------
hashing           — sklearn HashingVectorizer; works fully offline, no download.
                    Vectors are unit-normalised so cosine ≈ dot-product.
sentence_transformers — BAAI/bge-small-en-v1.5 (or any SBERT model); first run
                    downloads ~130 MB to the HuggingFace cache.
openai            — text-embedding-3-small via the OpenAI API; requires OPENAI_API_KEY.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from src.config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)


class EmbedderProtocol(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dim(self) -> int: ...


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return L2-normalised float vectors, one per input text."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @staticmethod
    def _l2_normalise(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms


class HashingEmbedder(BaseEmbedder):
    """
    Fully offline, zero-download embedder using TF-IDF-style hashing.

    Not semantically meaningful but useful for CI and development without
    needing model downloads.  Produces reproducible vectors for the same text.
    """

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self._dim = settings.embed_dim
        self._vec = HashingVectorizer(
            n_features=self._dim,
            norm="l2",
            alternate_sign=False,
            ngram_range=(1, 2),
        )
        logger.info("HashingEmbedder initialised", extra={"dim": self._dim})

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self._vec.transform(texts).toarray().astype(np.float32)
        # HashingVectorizer with norm="l2" already normalises each row
        return matrix.tolist()


class SentenceTransformerEmbedder(BaseEmbedder):
    """Semantic embeddings via sentence-transformers (downloads on first use)."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        model_name = settings.embed_model
        logger.info("Loading SentenceTransformer model", extra={"model": model_name})
        self._model = SentenceTransformer(model_name)
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self._dim = get_dim()
        logger.info("SentenceTransformer ready", extra={"dim": self._dim})

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI text-embedding-3-small (or any openai embedding model)."""

    def __init__(self) -> None:
        import openai  # type: ignore[import]

        self._client = openai.OpenAI()
        self._model = settings.llm_model or "text-embedding-3-small"
        self._dim = settings.embed_dim
        logger.info("OpenAIEmbedder initialised", extra={"model": self._model})

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # OpenAI API accepts up to 2048 inputs per call
        batch_size = 512
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(input=batch, model=self._model)
            vecs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            matrix = np.array(vecs, dtype=np.float32)
            matrix = self._l2_normalise(matrix)
            all_vecs.extend(matrix.tolist())
        return all_vecs


def get_embedder() -> BaseEmbedder:
    """Factory — returns the configured embedder singleton."""
    backend = settings.embedder
    if backend == "hashing":
        return HashingEmbedder()
    elif backend == "sentence_transformers":
        return SentenceTransformerEmbedder()
    elif backend == "openai":
        return OpenAIEmbedder()
    else:
        raise ValueError(f"Unknown embedder backend: {backend!r}")
