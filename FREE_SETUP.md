# Free Setup Guide

OmniGraph can run without paid APIs.

## What You Need

- Docker Desktop, for the free local PostgreSQL + pgvector database.
- Python 3.11+, if you want to run the API or console outside Docker.
- No paid database account.
- No paid embedding account.
- No LLM account is required for ingest, search, graph build, REST, MCP, or console commands.

## Optional Free Account

Create a free OpenRouter account only if you want chat answers or LLM-based entity extraction:

1. Go to https://openrouter.ai/
2. Create a free account.
3. Create an API key.
4. Put it in `.env`:

```bash
OPENROUTER_API_KEY=your_key_here
```

The agent is configured to use free OpenRouter models by default. If the key is empty, OmniGraph still works with keyword extraction and local embeddings.

## Fully Free Local Run

```bash
git clone https://github.com/ompatelz/Omni-Graph_Omi.git
cd Omni-Graph_Omi
cp .env.example .env
docker compose up
```

Then open:

- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- PostgreSQL: localhost:5432

Default database values from `.env.example`:

```bash
OMNIGRAPH_DB_HOST=localhost
OMNIGRAPH_DB_PORT=5432
OMNIGRAPH_DB_NAME=omnigraph
OMNIGRAPH_DB_USER=postgres
OMNIGRAPH_DB_PASSWORD=postgres
```

## Free Embeddings

When `VOYAGE_API_KEY` is empty, OmniGraph uses `local-hash-embedding-v1`.

This is a deterministic local embedding:

- It costs nothing.
- It does not call any API.
- It works with pgvector cosine search.
- It is lower quality than hosted embedding models, but good enough for a free demo and interview walkthrough.

To use Voyage later, set:

```bash
VOYAGE_API_KEY=your_key_here
```

## What To Say In An Interview

"The free path uses Dockerized PostgreSQL with pgvector as the database and vector index. For embeddings, the app falls back to a deterministic local hashing embedder, so semantic search works without hosted embedding APIs. LLM features are optional: with no OpenRouter key, ingestion still extracts entities and concepts through regex and keyword dictionaries; with a free OpenRouter key, the RAG agent and LLM extraction use free models."

