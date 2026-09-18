from __future__ import annotations

from typing import Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
        populate_by_name=True,
    )

    # ── Database ──────────────────────────────────────────────────────────
    db_host: str = Field("localhost", validation_alias="OMNIGRAPH_DB_HOST")
    db_port: int = Field(5432, validation_alias="OMNIGRAPH_DB_PORT")
    db_name: str = Field("omnigraph", validation_alias="OMNIGRAPH_DB_NAME")
    db_user: str = Field("postgres", validation_alias="OMNIGRAPH_DB_USER")
    db_password: str = Field("postgres", validation_alias="OMNIGRAPH_DB_PASSWORD")
    db_pool_min: int = Field(2, validation_alias="OMNIGRAPH_DB_POOL_MIN")
    db_pool_max: int = Field(10, validation_alias="OMNIGRAPH_DB_POOL_MAX")

    # ── AI APIs ───────────────────────────────────────────────────────────
    anthropic_api_key: str = Field("", validation_alias="ANTHROPIC_API_KEY")
    openrouter_api_key: str = Field("", validation_alias="OPENROUTER_API_KEY")
    openai_api_key: str = Field("", validation_alias="OPENAI_API_KEY")
    voyage_api_key: str = Field("", validation_alias="VOYAGE_API_KEY")
    ollama_base_url: str = Field("http://localhost:11434/v1", validation_alias="OLLAMA_BASE_URL")
    embedding_provider: str = Field("auto", validation_alias="OMNIGRAPH_EMBEDDING_PROVIDER")
    llm_provider: str = Field("auto", validation_alias="OMNIGRAPH_LLM_PROVIDER")
    llm_model: str = Field("", validation_alias="OMNIGRAPH_LLM_MODEL")

    # ── REST API ──────────────────────────────────────────────────────────
    api_key: str = Field("", validation_alias="OMNIGRAPH_API_KEY")
    cors_origins: Any = Field(["*"], validation_alias="OMNIGRAPH_CORS_ORIGINS")
    log_level: str = Field("INFO", validation_alias="OMNIGRAPH_LOG_LEVEL")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, v: object) -> list[str]:
        if isinstance(v, str):
            v_strip = v.strip()
            if v_strip.startswith("[") and v_strip.endswith("]"):
                import json
                try:
                    loaded = json.loads(v_strip)
                    if isinstance(loaded, list):
                        return [str(o).strip() for o in loaded if str(o).strip()]
                except Exception:
                    pass
            return [o.strip() for o in v.split(",") if o.strip()]
        return v  # type: ignore[return-value]

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, v: object) -> str:
        return str(v).upper()


settings = Settings()
