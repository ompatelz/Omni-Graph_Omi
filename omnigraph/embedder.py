"""Multi-tier resilient embedding engine for OmniGraph.

Supports:
  1. Voyage AI (when VOYAGE_API_KEY is configured)
  2. FastEmbed (when fastembed is installed)
  3. Built-in zero-dependency deterministic semantic feature hashing (always available)
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import List, Optional

from .config import settings

logger = logging.getLogger("omnigraph.embedder")

VOYAGE_MODEL_NAME = "voyage-3"
VOYAGE_DIM = 1024

FASTEMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
FASTEMBED_DIM = 384

LOCAL_HASH_MODEL_NAME = "omnigraph-local-hash-v1"
LOCAL_HASH_DIM = 384

_voyage_client = None
_fastembed_model = None


def _get_voyage_client():
    global _voyage_client
    if _voyage_client is None:
        import voyageai  # type: ignore[import-untyped]
        _voyage_client = voyageai.Client(api_key=settings.voyage_api_key)
    return _voyage_client


def _get_fastembed_model():
    global _fastembed_model
    if _fastembed_model is None:
        from fastembed import TextEmbedding  # type: ignore[import-untyped]
        _fastembed_model = TextEmbedding(FASTEMBED_MODEL_NAME)
    return _fastembed_model


def is_available() -> bool:
    """Return True since OmniGraph always provides a local embedding fallback."""
    return True


def current_model_name() -> str:
    """Return the name of the currently active embedding model."""
    provider = (settings.embedding_provider or "auto").lower()
    if (provider in ("auto", "voyage")) and settings.voyage_api_key:
        try:
            import voyageai  # type: ignore[import-untyped]
            return VOYAGE_MODEL_NAME
        except Exception:
            pass
    if provider in ("auto", "fastembed"):
        try:
            _get_fastembed_model()
            return FASTEMBED_MODEL_NAME
        except Exception:
            pass
    return LOCAL_HASH_MODEL_NAME


def current_embedding_dim() -> int:
    """Return the vector dimensionality of the active embedding model."""
    name = current_model_name()
    if name == VOYAGE_MODEL_NAME:
        return VOYAGE_DIM
    if name == FASTEMBED_MODEL_NAME:
        return FASTEMBED_DIM
    return LOCAL_HASH_DIM


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    if not vec1 or not vec2:
        return 0.0
    min_len = min(len(vec1), len(vec2))
    dot = sum(vec1[i] * vec2[i] for i in range(min_len))
    norm1 = math.sqrt(sum(x * x for x in vec1[:min_len]))
    norm2 = math.sqrt(sum(y * y for y in vec2[:min_len]))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def _local_hash_embedding(text: str, dim: int = LOCAL_HASH_DIM) -> List[float]:
    """Deterministic, zero-dependency subword and n-gram feature hashing embedding.

    Provides stable semantic vector representations with subword character 3-grams,
    word unigrams, and bigrams projected into an L2-normalized vector.
    """
    text_clean = text.strip()[:32000].lower()
    tokens = re.findall(r"[a-z0-9]+", text_clean)
    vector = [0.0] * dim
    if not tokens:
        return vector

    # Build features: words, word pairs, and character n-grams
    features: List[str] = list(tokens)
    features.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
    for token in tokens:
        if len(token) >= 4:
            features.extend(token[i : i + 3] for i in range(len(token) - 2))

    for feat in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [round(v / norm, 6) for v in vector]


_voyage_warned = False

def generate_embedding(text: str, input_type: str = "document") -> List[float]:
    """Generate normalized embedding vector for the input text."""
    global _voyage_warned
    provider = (settings.embedding_provider or "auto").lower()

    # Tier 1: Voyage AI
    if (provider in ("auto", "voyage")) and settings.voyage_api_key:
        try:
            client = _get_voyage_client()
            res = client.embed([text.strip()[:32000]], model=VOYAGE_MODEL_NAME, input_type=input_type)
            return [round(float(v), 6) for v in res.embeddings[0]]
        except Exception as exc:
            if not _voyage_warned:
                logger.info("Voyage AI unavailable (%s); using local embedding engine.", exc)
                _voyage_warned = True

    # Tier 2: FastEmbed
    if provider in ("auto", "fastembed"):
        try:
            model = _get_fastembed_model()
            result = list(model.embed([text.strip()[:32000]]))[0]
            return [round(float(v), 6) for v in result]
        except Exception:
            pass

    # Tier 3: Zero-dependency local hashing
    return _local_hash_embedding(text, dim=current_embedding_dim())