"""
RAG query pipeline — retrieval + optional generation.

The pipeline is intentionally thin: it delegates to the embedder for query
encoding, the vector store for retrieval, and an optional LLM for generation.
Every query is logged to the warehouse for analytics and evaluation.

LLM providers
-------------
none       — retrieval-only; works with no API key (default)
openai     — ChatCompletion with cited sources
anthropic  — Messages API with cited sources
groq       — OpenAI-compatible endpoint
"""

from __future__ import annotations

import time
from typing import Callable

from src.config import settings
from src.embed.embedder import BaseEmbedder
from src.logging_utils import get_logger
from src.models import QueryResult, RetrievedChunk
from src.store.qdrant_store import QdrantStore
from src.store.warehouse import Warehouse

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant.  Answer the user's question using ONLY "
    "the provided SEC filing excerpts.  For each claim, cite the filing (company name, "
    "form type, and date) in parentheses.  If the answer cannot be found in the "
    "provided excerpts, respond with: 'The answer is not found in the provided filings.'"
)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, rc in enumerate(chunks, 1):
        c = rc.chunk
        parts.append(
            f"[{i}] {c.company_name} ({c.form_type}, {c.filed_date}):\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


def _generate_openai(question: str, context: str) -> str:
    import openai  # type: ignore[import]

    model = settings.llm_model or "gpt-4o-mini"
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def _generate_anthropic(question: str, context: str) -> str:
    import anthropic  # type: ignore[import]

    model = settings.llm_model or "claude-haiku-4-5-20251001"
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.content[0].text if response.content else ""


def _generate_groq(question: str, context: str) -> str:
    from groq import Groq  # type: ignore[import]

    model = settings.llm_model or "llama3-8b-8192"
    client = Groq()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


_GENERATORS: dict[str, Callable[[str, str], str]] = {
    "openai": _generate_openai,
    "anthropic": _generate_anthropic,
    "groq": _generate_groq,
}


class RAGPipeline:
    """
    End-to-end query handler.

    Instantiate once and call .query() repeatedly.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: QdrantStore,
        warehouse: Warehouse,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._warehouse = warehouse
        self._generator = _GENERATORS.get(settings.llm_provider)

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        t0 = time.perf_counter()

        # 1. Embed the query
        query_vec = self._embedder.embed([question])[0]

        # 2. Retrieve
        retrieved = self._store.search(query_vec, top_k=top_k or settings.top_k)

        # 3. Generate (optional)
        answer: str | None = None
        if self._generator and retrieved:
            context = _build_context(retrieved)
            answer = self._generator(question, context)
        elif not retrieved:
            answer = "No relevant filings found in the vector store."

        latency_ms = (time.perf_counter() - t0) * 1000

        result = QueryResult(
            question=question,
            retrieved_chunks=retrieved,
            answer=answer,
            latency_ms=latency_ms,
        )

        # 4. Log to warehouse
        self._warehouse.log_query(result)

        logger.info(
            "Query processed",
            extra={
                "query_id": result.query_id,
                "question": question[:80],
                "chunks_retrieved": len(retrieved),
                "latency_ms": round(latency_ms, 1),
                "llm": settings.llm_provider,
            },
        )
        return result
