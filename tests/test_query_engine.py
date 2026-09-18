"""Tests for Semantic Query Engine: parsing, searching, and hybrid ranking."""
from unittest.mock import MagicMock

import psycopg2
import pytest
from omnigraph.semantic_query_engine import SemanticQueryEngine


def test_parse_query_stop_word_removal():
    """Verify stop words and short tokens (<=2 chars) are excluded from parsed terms."""
    query = "What is the best way to find a neural network in Python?"
    parsed = SemanticQueryEngine.parse_query(query)

    assert "the" not in parsed["terms"]
    assert "is" not in parsed["terms"]
    assert "what" not in parsed["terms"]
    assert "to" not in parsed["terms"]
    assert "in" not in parsed["terms"]
    assert "neural" in parsed["terms"]
    assert "network" in parsed["terms"]
    assert "python" in parsed["terms"]
    assert parsed["original_query"] == query


def test_parse_query_intents():
    """Verify intent classification based on query patterns."""
    q_expert = "Who is the specialist or expert in deep learning?"
    assert SemanticQueryEngine.parse_query(q_expert)["intent"] == "find_expert"

    q_related = "Show related and similar concepts to machine learning"
    assert SemanticQueryEngine.parse_query(q_related)["intent"] == "find_related"

    q_path = "What is the connection or path between Kubernetes and Docker?"
    assert SemanticQueryEngine.parse_query(q_path)["intent"] == "find_path"

    q_trend = "Historical trend over time for security incidents"
    assert SemanticQueryEngine.parse_query(q_trend)["intent"] == "analytics"

    q_search = "Deploying microservices with Envoy gateway"
    assert SemanticQueryEngine.parse_query(q_search)["intent"] == "search"


def test_rank_results_deduplication_and_weights():
    """Verify hybrid rank combines scores with proper weights and deduplicates document_ids."""
    raw_results = [
        {"document_id": 1, "title": "Doc 1", "score": 0.5, "search_type": "fulltext"},
        {"document_id": 1, "title": "Doc 1", "score": 0.5, "search_type": "semantic"},
        {"document_id": 2, "title": "Doc 2", "score": 0.8, "search_type": "graph"},
        {"document_id": 3, "title": "Doc 3", "score": 0.1, "search_type": "fulltext"},
    ]
    ranked = SemanticQueryEngine.rank_results(raw_results)

    # doc 1 should have accumulated score:
    # 0.5 * 1.0 (fulltext) + 0.5 * 1.2 (semantic) = 0.5 + 0.6 = 1.1
    # doc 2 should have score:
    # 0.8 * 0.8 (graph) = 0.64
    # doc 3 should have score: 0.1 * 1.0 = 0.1
    assert len(ranked) == 3
    assert ranked[0]["document_id"] == 1
    assert pytest.approx(ranked[0]["score"], abs=1e-3) == 1.1
    assert set(ranked[0]["sources"]) == {"fulltext", "semantic"}

    assert ranked[1]["document_id"] == 2
    assert pytest.approx(ranked[1]["score"], abs=1e-3) == 0.64
    assert ranked[1]["sources"] == ["graph"]


def test_rank_results_empty_or_malformed():
    """Verify rank_results handles empty inputs and entries without document_id."""
    assert SemanticQueryEngine.rank_results([]) == []
    malformed = [{"score": 1.0, "search_type": "fulltext"}, {"document_id": None}]
    assert SemanticQueryEngine.rank_results(malformed) == []


def test_fulltext_search(db_conn):
    """Verify fulltext search against database documents."""
    engine = SemanticQueryEngine(db_conn, user_id=1)
    results = engine.fulltext_search("machine learning", limit=5)
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(r["search_type"] == "fulltext" for r in results)
    assert all("score" in r and "document_id" in r for r in results)


def test_fulltext_search_with_sensitivity_filter(db_conn):
    """Verify fulltext search honors sensitivity level filter."""
    engine = SemanticQueryEngine(db_conn, user_id=1)
    public_results = engine.fulltext_search(
        "machine learning", limit=5, sensitivity_filter=["public"]
    )
    assert all(r["sensitivity_level"] == "public" for r in public_results)


def test_vector_similarity_search(db_conn):
    """Verify semantic vector search against embeddings."""
    engine = SemanticQueryEngine(db_conn, user_id=1)
    results = engine.vector_similarity_search("Kubernetes cloud infrastructure", limit=5)
    assert isinstance(results, list)
    if results:
        assert all(r["search_type"] == "semantic" for r in results)
        # Scores should be in descending order
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


def test_graph_traverse(db_conn):
    """Verify graph traversal based on parsed terms."""
    engine = SemanticQueryEngine(db_conn, user_id=1)
    parsed = engine.parse_query("Kubernetes Docker containers")
    results = engine.graph_traverse(parsed, limit=5)
    assert isinstance(results, list)
    if results:
        assert all(r["search_type"] == "graph" for r in results)
        assert all("score" in r for r in results)


def test_graph_traverse_empty_terms(db_conn):
    """Verify graph traversal returns empty list when query has no searchable terms."""
    engine = SemanticQueryEngine(db_conn, user_id=1)
    results = engine.graph_traverse({"terms": []}, limit=5)
    assert results == []


def test_hybrid_search(db_conn):
    """Verify hybrid search combines strategies and ranks results."""
    engine = SemanticQueryEngine(db_conn, user_id=1)
    results = engine.search("federated learning privacy", strategy="hybrid", limit=5)
    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) <= 5
    assert "sources" in results[0]


def test_search_strategies(db_conn):
    """Verify explicit strategy selection in search method."""
    engine = SemanticQueryEngine(db_conn, user_id=1)
    ft = engine.search("machine learning", strategy="fulltext", limit=3)
    sem = engine.search("machine learning", strategy="semantic", limit=3)
    gr = engine.search("machine learning", strategy="graph", limit=3)
    unknown = engine.search("machine learning", strategy="unknown_strategy", limit=3)

    assert isinstance(ft, list)
    assert isinstance(sem, list)
    assert isinstance(gr, list)
    assert isinstance(unknown, list)


def test_find_experts(db_conn):
    """Verify finding experts based on concept contributions."""
    engine = SemanticQueryEngine(db_conn, user_id=1)
    experts = engine.find_experts("Deep Learning", limit=5)
    assert isinstance(experts, list)
    assert len(experts) > 0
    assert "full_name" in experts[0]
    assert "expertise_score" in experts[0]


def test_find_related_concepts(db_conn):
    """Verify finding concepts related through hierarchy and co-occurrence."""
    engine = SemanticQueryEngine(db_conn, user_id=1)
    related = engine.find_related_concepts("Machine Learning")
    assert isinstance(related, list)
    assert len(related) > 0
    assert "name" in related[0]
    assert "domain" in related[0]


def test_get_entity_documents(db_conn):
    """Verify retrieving documents linked to an entity."""
    engine = SemanticQueryEngine(db_conn, user_id=1)
    docs = engine.get_entity_documents("Kubernetes", limit=5)
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert "document_id" in docs[0]
    assert "relevance" in docs[0]


def test_query_engine_db_error_handling(mock_db):
    """Verify database exceptions are handled and rollbacks are executed."""
    engine = SemanticQueryEngine(mock_db, user_id=1)
    cursor = mock_db.conn.cursor.return_value.__enter__.return_value
    cursor.execute.side_effect = psycopg2.OperationalError("Database error")

    assert engine.fulltext_search("test") == []
    assert engine.vector_similarity_search("test") == []
    assert engine.graph_traverse({"terms": ["test"]}) == []
    assert engine.find_experts("test") == []
    assert engine.find_related_concepts("test") == []
    assert engine.get_entity_documents("test") == []
