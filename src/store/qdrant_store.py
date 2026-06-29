"""
Qdrant vector store wrapper.

The connection is driven by SEC_QDRANT_LOCATION:
  ":memory:"         — in-process ephemeral store (tests, CI)
  "./qdrant_data"    — embedded persistent store (local dev, no server needed)
  "http://host:port" — remote Qdrant server (Docker / Qdrant Cloud)

All three paths use identical application code — only the client constructor
changes, which is handled in _get_client() below.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from qdrant_client.models import ScoredPoint

from src.config import settings
from src.logging_utils import get_logger
from src.models import Chunk, RetrievedChunk

logger = get_logger(__name__)


def _get_client() -> QdrantClient:
    loc = settings.qdrant_location
    if loc == ":memory:":
        return QdrantClient(":memory:")
    elif loc.startswith("http://") or loc.startswith("https://"):
        return QdrantClient(url=loc)
    else:
        # Embedded persistent mode (no server process required)
        return QdrantClient(path=loc)


class QdrantStore:
    """
    Thin wrapper around QdrantClient for chunk upsert and similarity search.

    Collection is created on first use with cosine distance (equivalent to
    dot-product on L2-normalised vectors, which all embedders guarantee).
    """

    def __init__(self, client: QdrantClient | None = None) -> None:
        self._client = client or _get_client()
        self._collection = settings.qdrant_collection
        self._dim = settings.embed_dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection",
                        extra={"collection": self._collection, "dim": self._dim})

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """
        Upsert chunks that have embeddings.  Returns the number upserted.

        Chunks without an embedding vector are skipped with a warning.
        """
        points: list[PointStruct] = []
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning("Chunk missing embedding; skipping",
                               extra={"chunk_id": chunk.chunk_id})
                continue
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "filing_id": chunk.filing_id,
                "cik": chunk.cik,
                "company_name": chunk.company_name,
                "form_type": chunk.form_type,
                "filed_date": chunk.filed_date.isoformat(),
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "char_count": chunk.char_count,
            }
            points.append(PointStruct(
                id=_uuid_to_uint64(chunk.chunk_id),
                vector=chunk.embedding,
                payload=payload,
            ))

        if not points:
            return 0

        self._client.upsert(collection_name=self._collection, points=points)
        logger.info("Upserted chunks to Qdrant",
                    extra={"count": len(points), "collection": self._collection})
        return len(points)

    def search(self, query_vector: list[float], top_k: int | None = None) -> list[RetrievedChunk]:
        """Return top-k chunks most similar to query_vector."""
        k = top_k or settings.top_k
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=k,
            with_payload=True,
        )
        results: list[ScoredPoint] = response.points
        retrieved: list[RetrievedChunk] = []
        for r in results:
            p = r.payload or {}
            chunk = Chunk(
                chunk_id=p.get("chunk_id", ""),
                filing_id=p.get("filing_id", ""),
                cik=p.get("cik", ""),
                company_name=p.get("company_name", ""),
                form_type=p.get("form_type", ""),
                filed_date=p.get("filed_date", "1970-01-01"),
                chunk_index=p.get("chunk_index", 0),
                text=p.get("text", ""),
                char_count=p.get("char_count", 0),
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=r.score))
        return retrieved

    def collection_count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count or 0

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection)


def _uuid_to_uint64(uuid_str: str) -> int:
    """
    Qdrant point IDs must be unsigned 64-bit integers or UUIDs.

    We use UUID strings directly — qdrant-client accepts them as-is.
    This function is kept as documentation of the decision.
    """
    return uuid_str  # type: ignore[return-value]  # qdrant-client accepts uuid strings
