"""Integration and route tests for OmniGraph REST API."""
import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as client:
        yield client


def test_health_endpoint(api_client):
    """Verify /health endpoint returns database connectivity status."""
    res = api_client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "database" in data


def test_explorer_endpoints(api_client):
    """Verify web knowledge graph explorer is served at / and /explorer."""
    res_root = api_client.get("/")
    assert res_root.status_code == 200
    assert "text/html" in res_root.headers.get("content-type", "")
    assert "OmniGraph" in res_root.text

    res_explorer = api_client.get("/explorer")
    assert res_explorer.status_code == 200
    assert "text/html" in res_explorer.headers.get("content-type", "")
    assert "vis-network" in res_explorer.text


def test_graph_data_endpoint(api_client):
    """Verify /api/v1/graph/data returns nodes and edges payload."""
    res = api_client.get("/api/v1/graph/data?limit=50", headers={"X-API-Key": "changeme"})
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert "edges" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)


def test_graph_stats_endpoint(api_client):
    """Verify /api/v1/graph/stats returns aggregate metrics."""
    res = api_client.get("/api/v1/graph/stats", headers={"X-API-Key": "changeme"})
    assert res.status_code == 200
    data = res.json()
    assert "total_documents" in data
    assert "total_entities" in data
