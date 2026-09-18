"""Tests for multi-tier resilient embedding engine."""
import math
import sys
from unittest.mock import MagicMock, patch

import pytest
from omnigraph import embedder
from omnigraph.embedder import (
    FASTEMBED_DIM,
    FASTEMBED_MODEL_NAME,
    LOCAL_HASH_DIM,
    LOCAL_HASH_MODEL_NAME,
    VOYAGE_DIM,
    VOYAGE_MODEL_NAME,
    _local_hash_embedding,
    compute_cosine_similarity,
    current_embedding_dim,
    current_model_name,
    generate_embedding,
    is_available,
)


def test_is_available():
    """Embedding engine must always report available due to local hash fallback."""
    assert is_available() is True


def test_cosine_similarity_perfect_match():
    """Identical vectors must have cosine similarity of 1.0."""
    vec1 = [1.0, 0.0, 2.0, -1.0]
    vec2 = [1.0, 0.0, 2.0, -1.0]
    sim = compute_cosine_similarity(vec1, vec2)
    assert pytest.approx(sim, abs=1e-5) == 1.0


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors must have cosine similarity of 0.0."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    sim = compute_cosine_similarity(vec1, vec2)
    assert pytest.approx(sim, abs=1e-5) == 0.0


def test_cosine_similarity_opposite():
    """Opposite vectors must have cosine similarity of -1.0."""
    vec1 = [1.0, 2.0, 3.0]
    vec2 = [-1.0, -2.0, -3.0]
    sim = compute_cosine_similarity(vec1, vec2)
    assert pytest.approx(sim, abs=1e-5) == -1.0


def test_cosine_similarity_zero_norm():
    """Vectors with all zeros must return 0.0 without division by zero."""
    zero_vec = [0.0, 0.0, 0.0]
    nonzero_vec = [1.0, 2.0, 3.0]
    assert compute_cosine_similarity(zero_vec, nonzero_vec) == 0.0
    assert compute_cosine_similarity(nonzero_vec, zero_vec) == 0.0
    assert compute_cosine_similarity(zero_vec, zero_vec) == 0.0


def test_cosine_similarity_empty_or_mismatched():
    """Empty vectors or differing dimensions should be handled safely."""
    assert compute_cosine_similarity([], [1.0, 2.0]) == 0.0
    assert compute_cosine_similarity([1.0, 2.0], []) == 0.0
    # Mismatched length uses min_len
    vec1 = [1.0, 0.0]
    vec2 = [1.0, 0.0, 50.0]
    assert pytest.approx(compute_cosine_similarity(vec1, vec2), abs=1e-5) == 1.0


def test_deterministic_hash_embedding():
    """Hash embedding must be deterministic, normalized, and case-insensitive."""
    text1 = "PostgreSQL Knowledge Graph with Vector Search"
    text2 = "postgresql knowledge graph with vector search"
    
    vec1 = _local_hash_embedding(text1, dim=384)
    vec2 = _local_hash_embedding(text2, dim=384)

    assert len(vec1) == 384
    assert len(vec2) == 384
    # Determinism and case-insensitivity
    assert vec1 == vec2

    # L2 normalized
    norm = math.sqrt(sum(v * v for v in vec1))
    assert pytest.approx(norm, abs=1e-3) == 1.0

    # Empty or whitespace input
    empty_vec = _local_hash_embedding("   ", dim=384)
    assert len(empty_vec) == 384
    assert all(v == 0.0 for v in empty_vec)

    # Custom dimension support
    custom_vec = _local_hash_embedding("Machine Learning", dim=128)
    assert len(custom_vec) == 128


def test_hash_embedding_semantic_property():
    """Texts sharing n-grams and tokens should have higher similarity than disjoint texts."""
    v_ai1 = _local_hash_embedding("Kubernetes container orchestration on cloud platforms")
    v_ai2 = _local_hash_embedding("Kubernetes deployment and container clusters")
    v_baking = _local_hash_embedding("Sourdough bread flour water yeast fermentation recipe")

    sim_related = compute_cosine_similarity(v_ai1, v_ai2)
    sim_unrelated = compute_cosine_similarity(v_ai1, v_baking)

    assert sim_related > sim_unrelated
    assert sim_related > 0.1


def test_dimension_queries_for_providers():
    """Verify dimension and model name reporting under different provider configurations."""
    # Local hash (default without keys or when explicitly set)
    with patch.object(embedder.settings, "embedding_provider", "auto"), \
         patch.object(embedder.settings, "voyage_api_key", ""):
        assert current_model_name() in (LOCAL_HASH_MODEL_NAME, FASTEMBED_MODEL_NAME)
        assert current_embedding_dim() in (LOCAL_HASH_DIM, FASTEMBED_DIM)

    # FastEmbed provider configured
    with patch.object(embedder.settings, "embedding_provider", "fastembed"), \
         patch("omnigraph.embedder._get_fastembed_model", return_value=MagicMock()):
        assert current_model_name() == FASTEMBED_MODEL_NAME
        assert current_embedding_dim() == FASTEMBED_DIM

    # Voyage AI provider configured
    with patch.object(embedder.settings, "embedding_provider", "voyage"), \
         patch.object(embedder.settings, "voyage_api_key", "mock-key"), \
         patch.dict(sys.modules, {"voyageai": MagicMock()}):
        assert current_model_name() == VOYAGE_MODEL_NAME
        assert current_embedding_dim() == VOYAGE_DIM


def test_generate_embedding_tier1_voyage():
    """Verify Tier 1 Voyage AI invocation when configured."""
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.embeddings = [[0.123456, 0.654321] + [0.0] * (VOYAGE_DIM - 2)]
    mock_client.embed.return_value = mock_res

    with patch.object(embedder.settings, "embedding_provider", "voyage"), \
         patch.object(embedder.settings, "voyage_api_key", "test-voyage-key"), \
         patch.dict(sys.modules, {"voyageai": MagicMock()}), \
         patch("omnigraph.embedder._get_voyage_client", return_value=mock_client):
        vec = generate_embedding("Test Voyage document", input_type="document")
        assert len(vec) == VOYAGE_DIM
        assert vec[0] == 0.123456
        assert vec[1] == 0.654321
        mock_client.embed.assert_called_once()


def test_generate_embedding_tier2_fastembed():
    """Verify Tier 2 FastEmbed invocation when configured."""
    mock_model = MagicMock()
    mock_model.embed.return_value = [[0.2] * FASTEMBED_DIM]

    with patch.object(embedder.settings, "embedding_provider", "fastembed"), \
         patch("omnigraph.embedder._get_fastembed_model", return_value=mock_model):
        vec = generate_embedding("Test FastEmbed document")
        assert len(vec) == FASTEMBED_DIM
        assert vec[0] == 0.2


def test_generate_embedding_tier3_local_hash_fallback():
    """When Voyage AI raises an exception, generate_embedding must fall back to local hash."""
    mock_client = MagicMock()
    mock_client.embed.side_effect = RuntimeError("Voyage service unavailable")

    with patch.object(embedder.settings, "embedding_provider", "voyage"), \
         patch.object(embedder.settings, "voyage_api_key", "test-key"), \
         patch.dict(sys.modules, {"voyageai": MagicMock()}), \
         patch("omnigraph.embedder._get_voyage_client", return_value=mock_client):
        # Should NOT raise, must return local hash embedding
        vec = generate_embedding("Document with fallback", input_type="document")
        assert len(vec) == current_embedding_dim()
        assert any(v != 0.0 for v in vec)
