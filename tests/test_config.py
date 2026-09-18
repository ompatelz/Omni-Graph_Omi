"""Tests for OmniGraph configuration settings loading, defaults, and aliases."""
import os
from unittest.mock import patch

import pytest
from omnigraph.config import Settings, settings


def test_default_settings():
    """Verify default configuration values."""
    s = Settings(_env_file=None)
    assert s.db_host == "localhost"
    assert s.db_port == 5432
    assert s.db_name == "omnigraph"
    assert s.db_user == "postgres"
    assert s.db_password == "postgres"
    assert s.db_pool_min == 2
    assert s.db_pool_max == 10
    assert s.embedding_provider == "auto"
    assert s.llm_provider == "auto"
    assert s.llm_model == ""
    assert s.ollama_base_url == "http://localhost:11434/v1"
    assert s.cors_origins == ["*"]
    assert s.log_level == "INFO"


def test_singleton_settings_instance():
    """Verify that settings singleton is instantiated properly."""
    assert isinstance(settings, Settings)
    assert hasattr(settings, "db_host")
    assert hasattr(settings, "api_key")


def test_environment_variable_aliases():
    """Verify that validation_aliases properly map environment variables."""
    test_env = {
        "OMNIGRAPH_DB_HOST": "db.production.local",
        "OMNIGRAPH_DB_PORT": "5433",
        "OMNIGRAPH_DB_NAME": "omnigraph_prod",
        "OMNIGRAPH_DB_USER": "admin_user",
        "OMNIGRAPH_DB_PASSWORD": "secret_password",
        "OMNIGRAPH_DB_POOL_MIN": "5",
        "OMNIGRAPH_DB_POOL_MAX": "25",
        "ANTHROPIC_API_KEY": "sk-ant-test123",
        "OPENROUTER_API_KEY": "sk-or-test123",
        "OPENAI_API_KEY": "sk-ai-test123",
        "VOYAGE_API_KEY": "vy-test123",
        "OLLAMA_BASE_URL": "http://ollama.local:11434/v1",
        "OMNIGRAPH_EMBEDDING_PROVIDER": "voyage",
        "OMNIGRAPH_LLM_PROVIDER": "openrouter",
        "OMNIGRAPH_LLM_MODEL": "meta-llama/llama-3.3-70b-instruct:free",
        "OMNIGRAPH_API_KEY": "omni-secret-key",
        "OMNIGRAPH_LOG_LEVEL": "warning",
        "OMNIGRAPH_CORS_ORIGINS": "https://example.com, http://localhost:3000",
    }
    with patch.dict(os.environ, test_env, clear=False):
        s = Settings(_env_file=None)
        assert s.db_host == "db.production.local"
        assert s.db_port == 5433
        assert s.db_name == "omnigraph_prod"
        assert s.db_user == "admin_user"
        assert s.db_password == "secret_password"
        assert s.db_pool_min == 5
        assert s.db_pool_max == 25
        assert s.anthropic_api_key == "sk-ant-test123"
        assert s.openrouter_api_key == "sk-or-test123"
        assert s.openai_api_key == "sk-ai-test123"
        assert s.voyage_api_key == "vy-test123"
        assert s.ollama_base_url == "http://ollama.local:11434/v1"
        assert s.embedding_provider == "voyage"
        assert s.llm_provider == "openrouter"
        assert s.llm_model == "meta-llama/llama-3.3-70b-instruct:free"
        assert s.api_key == "omni-secret-key"
        assert s.log_level == "WARNING"
        assert s.cors_origins == ["https://example.com", "http://localhost:3000"]


def test_cors_origins_parsing():
    """Verify parsing comma-separated strings or retaining list of strings."""
    s1 = Settings(cors_origins="http://a.com, http://b.com, http://c.com", _env_file=None)
    assert s1.cors_origins == ["http://a.com", "http://b.com", "http://c.com"]

    s2 = Settings(cors_origins=["http://direct.com"], _env_file=None)
    assert s2.cors_origins == ["http://direct.com"]


def test_log_level_uppercasing():
    """Verify log_level validator always returns uppercase strings."""
    s = Settings(log_level="debug", _env_file=None)
    assert s.log_level == "DEBUG"


def test_extra_fields_ignored():
    """Verify extra fields are ignored without raising validation errors."""
    s = Settings(unrecognized_custom_field="value", _env_file=None)
    assert not hasattr(s, "unrecognized_custom_field")
