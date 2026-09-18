"""Pytest fixtures and configuration for OmniGraph test suite."""
import os
import sys
from unittest.mock import MagicMock

import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from omnigraph.ingestion_pipeline import DatabaseConnection
from api.main import app
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def db_conn():
    """Provides a live DatabaseConnection to PostgreSQL."""
    db = DatabaseConnection()
    try:
        db.connect()
    except Exception as exc:
        pytest.skip(f"Database unavailable: {exc}")
    yield db
    try:
        db.disconnect()
    except Exception:
        pass


@pytest.fixture
def mock_db():
    """Provides a mock DatabaseConnection for isolated unit tests."""
    mock = MagicMock(spec=DatabaseConnection)
    mock.conn = MagicMock()
    mock_cursor = MagicMock()
    mock.conn.cursor.return_value.__enter__.return_value = mock_cursor
    return mock


@pytest.fixture(scope="module")
def client():
    """Provides a FastAPI TestClient using lifespan context."""
    with TestClient(app) as c:
        yield c
