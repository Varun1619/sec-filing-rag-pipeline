"""Tests for the embedding backends."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.embed.embedder import HashingEmbedder, get_embedder
from src.config import settings


def test_hashing_embedder_shape():
    emb = HashingEmbedder()
    texts = ["Apple revenue", "SEC filing 10-K", "quarterly earnings"]
    vecs = emb.embed(texts)
    assert len(vecs) == len(texts)
    for v in vecs:
        assert len(v) == emb.dim, f"Expected dim {emb.dim}, got {len(v)}"


def test_hashing_embedder_normalised():
    emb = HashingEmbedder()
    vecs = emb.embed(["This is a test sentence for normalisation."])
    v = np.array(vecs[0])
    norm = np.linalg.norm(v)
    assert abs(norm - 1.0) < 1e-5 or norm == pytest.approx(1.0, abs=1e-4), \
        f"Vector should be L2-normalised; got norm={norm}"


def test_hashing_embedder_empty():
    emb = HashingEmbedder()
    result = emb.embed([])
    assert result == []


def test_hashing_embedder_dim_matches_config():
    emb = HashingEmbedder()
    assert emb.dim == settings.embed_dim


def test_get_embedder_default_is_hashing(monkeypatch):
    monkeypatch.setattr(settings, "embedder", "hashing")
    emb = get_embedder()
    assert isinstance(emb, HashingEmbedder)


def test_hashing_deterministic():
    """Same text should always produce the same vector."""
    emb = HashingEmbedder()
    v1 = emb.embed(["reproducible text"])
    v2 = emb.embed(["reproducible text"])
    assert v1 == v2


def test_different_texts_differ():
    emb = HashingEmbedder()
    v1 = np.array(emb.embed(["apple pie"])[0])
    v2 = np.array(emb.embed(["nuclear physics"])[0])
    # Cosine similarity should not be 1.0
    sim = float(np.dot(v1, v2))
    assert sim < 0.99, "Completely different texts should not be identical vectors"
