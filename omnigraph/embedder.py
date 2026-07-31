from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import List, Optional

from .config import settings

logger = logging.getLogger("omnigraph.embedder")

VOYAGE_MODEL_NAME = "voyage-3"
LOCAL_MODEL_NAME = "local-hash-embedding-v1"
EMBEDDING_DIM = 1024

_client = None
_unavailable: Optional[Exception] = None


def _get_client():
    global _client, _unavailable
    if _unavailable is not None:
        raise _unavailable
    if _client is None:
        try:
            import voyageai
        except ImportError as exc:
            _unavailable = ImportError(
                "voyageai is required for semantic search. "
                "Install it with: pip install voyageai"
            )
            raise _unavailable from exc
        api_key = settings.voyage_api_key
        if not api_key:
            _unavailable = EnvironmentError(
                "VOYAGE_API_KEY environment variable is not set. "
                "Get your key at https://www.voyageai.com/"
            )
            raise _unavailable
        _client = voyageai.Client(api_key=api_key)
    return _client


def is_available() -> bool:
    return True


def generate_embedding(text: str, input_type: str = "document") -> List[float]:
    if settings.voyage_api_key:
        try:
            client = _get_client()
            result = client.embed(
                [text.strip()[:32000]],
                model=VOYAGE_MODEL_NAME,
                input_type=input_type,
            )
            return [round(float(v), 6) for v in result.embeddings[0]]
        except Exception as exc:
            logger.warning("Voyage embedding failed; using local fallback: %s", exc)
    return _local_hash_embedding(text)


def current_model_name() -> str:
    return VOYAGE_MODEL_NAME if settings.voyage_api_key else LOCAL_MODEL_NAME


def _local_hash_embedding(text: str) -> List[float]:
    """Deterministic bag-of-words hashing embedding for no-cost local search."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    vector = [0.0] * EMBEDDING_DIM
    if not tokens:
        return vector

    features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [round(v / norm, 6) for v in vector]
