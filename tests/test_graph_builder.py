"""Tests for Knowledge Graph Builder: neighborhood, shortest path, and stats."""
from unittest.mock import MagicMock

import psycopg2
import pytest
from omnigraph.graph_builder import KnowledgeGraphBuilder


def test_graph_statistics(db_conn):
    """Verify graph statistics return aggregate counts and breakdown dictionaries."""
    builder = KnowledgeGraphBuilder(db_conn)
    stats = builder.get_graph_stats()

    assert "total_entities" in stats
    assert "total_relations" in stats
    assert "total_concepts" in stats
    assert "total_documents" in stats
    assert "total_taxonomy_nodes" in stats
    assert "entities_by_type" in stats
    assert "relations_by_type" in stats

    assert stats["total_entities"] > 0
    assert stats["total_relations"] > 0
    assert isinstance(stats["entities_by_type"], dict)
    assert isinstance(stats["relations_by_type"], dict)


def test_entity_neighborhood_depth1(db_conn):
    """Verify depth-1 neighborhood retrieval with directional relation indicators."""
    builder = KnowledgeGraphBuilder(db_conn)
    # Kubernetes is entity_id=4
    neighbors = builder.get_entity_neighborhood(4, max_depth=1)
    assert len(neighbors) > 0
    assert all(n["depth"] == 1 for n in neighbors)

    names = {n["name"] for n in neighbors}
    assert "Docker" in names or "Google" in names or "Istio" in names

    # Verify relation types include attributes
    assert all("relation_type" in n and "strength" in n for n in neighbors)


def test_entity_neighborhood_depth2(db_conn):
    """Verify depth-2 neighborhood retrieves multi-hop connected entities."""
    builder = KnowledgeGraphBuilder(db_conn)
    neighbors_d1 = builder.get_entity_neighborhood(4, max_depth=1)
    neighbors_d2 = builder.get_entity_neighborhood(4, max_depth=2)

    assert len(neighbors_d2) >= len(neighbors_d1)
    depths = {n["depth"] for n in neighbors_d2}
    assert 1 in depths
    assert 2 in depths


def test_entity_neighborhood_nonexistent(db_conn):
    """Neighborhood queries for unknown entity IDs return empty list."""
    builder = KnowledgeGraphBuilder(db_conn)
    assert builder.get_entity_neighborhood(999999, max_depth=2) == []


def test_shortest_path_traversal_bidirectional(db_conn):
    """Verify bidirectional shortest path traversal using omnigraph.sp_shortest_path."""
    with db_conn.conn.cursor() as cur:
        # Forward: Kubernetes (4) -> Transformer (21)
        cur.execute("SELECT path_length, path_entities, path_relations FROM omnigraph.sp_shortest_path(4, 21, 6)")
        forward_paths = cur.fetchall()
        assert len(forward_paths) > 0

        first_fwd = forward_paths[0]
        length_fwd, entities_fwd, relations_fwd = first_fwd
        assert length_fwd > 0
        assert entities_fwd[0] == "Kubernetes"
        assert entities_fwd[-1] == "Transformer"

        # Reverse: Transformer (21) -> Kubernetes (4)
        cur.execute("SELECT path_length, path_entities, path_relations FROM omnigraph.sp_shortest_path(21, 4, 6)")
        reverse_paths = cur.fetchall()
        assert len(reverse_paths) > 0

        first_rev = reverse_paths[0]
        length_rev, entities_rev, relations_rev = first_rev
        assert length_rev > 0
        assert entities_rev[0] == "Transformer"
        assert entities_rev[-1] == "Kubernetes"


def test_concept_hierarchy(db_conn):
    """Verify recursive concept hierarchy retrieval."""
    builder = KnowledgeGraphBuilder(db_conn)
    hier = builder.get_concept_hierarchy("Machine Learning")
    assert len(hier) > 0
    assert hier[0]["name"] == "Machine Learning"
    assert hier[0]["depth"] == 0
    # Must have subconcepts
    assert any(h["depth"] == 1 for h in hier)


def test_taxonomy_tree(db_conn):
    """Verify recursive taxonomy tree traversal."""
    builder = KnowledgeGraphBuilder(db_conn)
    tree = builder.get_taxonomy_tree()
    assert len(tree) > 0
    assert all("taxonomy_id" in node and "path" in node for node in tree)


def test_duplicate_nodes_detection(db_conn):
    """Verify duplicate node detector executes without errors."""
    builder = KnowledgeGraphBuilder(db_conn)
    dupes = builder.detect_duplicate_nodes()
    assert isinstance(dupes, list)


def test_entity_node_crud_lifecycle(db_conn):
    """Verify creating, updating, and removing an entity node."""
    builder = KnowledgeGraphBuilder(db_conn)
    test_name = "Pytest Temp Graph Node"
    test_type = "technology"


    # 1. Create
    entity_id = builder.add_entity_node(
        name=test_name,
        entity_type=test_type,
        description="Temporary entity node for automated testing",
        confidence=0.95,
    )
    assert entity_id is not None

    # Idempotent re-add returns same entity_id
    re_id = builder.add_entity_node(name=test_name, entity_type=test_type)
    assert re_id == entity_id

    # 2. Update
    updated = builder.update_entity_node(
        entity_id=entity_id,
        description="Updated description",
        confidence=0.99,
    )
    assert updated is True

    # 3. Delete
    deleted = builder.remove_entity_node(entity_id)
    assert deleted is True

    # Deleting again returns False
    assert builder.remove_entity_node(entity_id) is False


def test_graph_builder_db_error_handling(mock_db):
    """Verify error handling on database failures."""
    builder = KnowledgeGraphBuilder(mock_db)
    cursor = mock_db.conn.cursor.return_value.__enter__.return_value
    cursor.execute.side_effect = psycopg2.OperationalError("Graph DB failure")

    assert builder.add_entity_node("Test", "tech") is None
    assert builder.remove_entity_node(1) is False
    assert builder.update_entity_node(1, name="Test") is False
    assert builder.add_relationship(1, 2, "uses") is None
    assert builder.remove_relationship(1) is False
    assert builder.get_taxonomy_tree() == []
    assert builder.get_concept_hierarchy("ML") == []
    assert builder.get_entity_neighborhood(1) == []
    assert builder.detect_duplicate_nodes() == []
    assert builder.get_graph_stats() == {}
