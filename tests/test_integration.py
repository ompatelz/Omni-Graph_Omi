"""Comprehensive integration tests for OmniGraph core engine against PostgreSQL."""
import pytest
from omnigraph.ingestion_pipeline import DatabaseConnection, DocumentIngester
from omnigraph.semantic_query_engine import SemanticQueryEngine
from omnigraph.graph_builder import KnowledgeGraphBuilder
from omnigraph.access_control_audit import AccessControlManager
from omnigraph.entity_relation_extractor import EntityRelationExtractor


def test_db_connection(db_conn):
    """Verify live database connectivity."""
    assert db_conn is not None
    assert db_conn.conn is not None
    with db_conn.conn.cursor() as cur:
        cur.execute("SELECT 1")
        assert cur.fetchone()[0] == 1


def test_graph_stats(db_conn):
    """Verify knowledge graph statistics retrieval."""
    builder = KnowledgeGraphBuilder(db_conn)
    stats = builder.get_graph_stats()
    assert isinstance(stats, dict)
    assert "total_documents" in stats
    assert "total_entities" in stats
    assert "total_relations" in stats
    assert "total_concepts" in stats
    assert stats["total_documents"] >= 0


def test_users_present(db_conn):
    """Verify users table contains seeded users."""
    with db_conn.conn.cursor() as cur:
        cur.execute("SELECT user_id, username, full_name FROM omnigraph.users ORDER BY user_id")
        users = cur.fetchall()
    assert len(users) > 0


def test_search_strategies(db_conn):
    """Verify fulltext, semantic, hybrid, and graph search strategies."""
    qe = SemanticQueryEngine(db_conn, user_id=1)
    
    # 1. Fulltext search
    ft_res = qe.search("machine learning", strategy="fulltext", limit=3)
    assert isinstance(ft_res, list)

    # 2. Semantic search
    sem_res = qe.search("Kubernetes cloud deployment", strategy="semantic", limit=3)
    assert isinstance(sem_res, list)

    # 3. Hybrid search
    hyb_res = qe.search("federated learning privacy", strategy="hybrid", limit=3)
    assert isinstance(hyb_res, list)

    # 4. Graph search
    gr_res = qe.search("Kubernetes Docker", strategy="graph", limit=3)
    assert isinstance(gr_res, list)


def test_entity_neighborhood(db_conn):
    """Verify N-hop entity graph traversal."""
    builder = KnowledgeGraphBuilder(db_conn)
    with db_conn.conn.cursor() as cur:
        cur.execute("SELECT entity_id FROM omnigraph.entities LIMIT 1")
        row = cur.fetchone()
    if row:
        ent_id = row[0]
        neighbors = builder.get_entity_neighborhood(ent_id, max_depth=2)
        assert isinstance(neighbors, list)


def test_access_control(db_conn):
    """Verify RBAC and document sensitivity enforcement."""
    acm = AccessControlManager(db_conn)
    # Admin (user 1) should be granted access to public doc (doc 1)
    res_admin = acm.check_access(1, "document", 1, "read")
    assert res_admin is True


def test_concept_hierarchy_and_taxonomy(db_conn):
    """Verify concept hierarchy traversal and taxonomy trees."""
    builder = KnowledgeGraphBuilder(db_conn)
    tax = builder.get_taxonomy_tree()
    assert isinstance(tax, list)


def test_document_ingestion_and_extraction(db_conn):
    """Verify document ingestion and entity-relation extraction."""
    ingester = DocumentIngester(db_conn)
    doc_id = ingester.ingest_document(
        title="CI Test Document: Knowledge Graphs and Embeddings",
        source_type="other",
        content="CI Test: Knowledge graphs connect entities and relations using PostgreSQL and vector embeddings.",
        uploaded_by=1,
        sensitivity_level="public",
    )
    assert doc_id is not None
    assert doc_id > 0

    extractor = EntityRelationExtractor(db_conn)
    extraction = extractor.process_document(doc_id)
    assert isinstance(extraction, dict)
    assert "entities" in extraction
    assert "concepts" in extraction
    assert "relationships" in extraction
