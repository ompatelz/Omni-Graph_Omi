# RAG Pipeline — OpenRouter backend
from __future__ import annotations

import json
import logging
import os
import time
import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import openai

from .access_control_audit import AccessControlManager
from .config import settings
from .ingestion_pipeline import DatabaseConnection
from .semantic_query_engine import SemanticQueryEngine

logger = logging.getLogger(__name__)

def _format_docs(docs: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    out: List[str] = []
    total = 0
    for d in docs:
        title = d.get("title", "Untitled")
        summary = (d.get("summary") or "")[:600]
        doc_id = d.get("document_id", "")
        hint = f"  (call get_document_content({doc_id}) for full text)" if doc_id else ""
        line = f"[doc_id={doc_id}] {title}\n  {summary}{hint}"
        if total + len(line) > max_chars:
            break
        out.append(line)
        total += len(line)
    return "\n\n".join(out) if out else "No documents found."


class _OmniTool(NamedTuple):
    schema: Dict[str, Any]
    fn: Callable


def _create_tools(
    query_engine: SemanticQueryEngine,
    access_manager: AccessControlManager,
    user_id: int,
    db: DatabaseConnection,
) -> List[_OmniTool]:

    def hybrid_search(query: str, limit: int = 10) -> str:
        results = query_engine.search(query, strategy="hybrid", limit=limit)
        filtered = [
            r for r in results
            if r.get("document_id") is not None
            and access_manager.check_access(user_id, "document", r["document_id"], "read")
        ]
        return _format_docs(filtered)

    def find_experts(concept: str, limit: int = 5) -> str:
        experts = query_engine.find_experts(concept, limit=limit)
        if not experts:
            return "No experts found for that concept."
        lines = [
            f"- {e['full_name']} ({e.get('department', '')}): {e.get('expertise_score', 0):.1f}"
            for e in experts
        ]
        return "\n".join(lines)

    def get_entity_documents(entity_name: str, limit: int = 10) -> str:
        docs = query_engine.get_entity_documents(entity_name, limit=limit)
        filtered = [
            d for d in docs
            if d.get("document_id") is not None
            and access_manager.check_access(user_id, "document", d["document_id"], "read")
        ]
        return _format_docs(filtered)

    def find_related_concepts(concept: str) -> str:
        related = query_engine.find_related_concepts(concept)
        if not related:
            return "No related concepts found."
        lines = [
            f"- {c['name']} [{c.get('domain', '')}] ({c.get('relationship_types', '')})"
            for c in related[:15]
        ]
        return "\n".join(lines)

    def get_document_content(document_id: int, max_chars: int = 4000) -> str:
        if not access_manager.check_access(user_id, "document", document_id, "read"):
            return "Access denied to this document."
        try:
            with db.conn.cursor() as cur:
                cur.execute(
                    "SELECT title, content FROM omnigraph.documents WHERE document_id = %s",
                    (document_id,),
                )
                row = cur.fetchone()
            if not row:
                return "Document not found."
            title, content = row[0], (row[1] or "")[:max_chars]
            return f"Title: {title}\n\nContent:\n{content}"
        except Exception as e:
            return f"Error fetching document: {e}"

    return [
        _OmniTool(
            schema={
                "type": "function",
                "function": {
                    "name": "hybrid_search",
                    "description": "Search the knowledge graph using full-text, semantic, and graph traversal. Use for finding documents relevant to a topic or question.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "description": "Maximum number of results (default 10)"},
                        },
                        "required": ["query"],
                    }
                }
            },
            fn=hybrid_search,
        ),
        _OmniTool(
            schema={
                "type": "function",
                "function": {
                    "name": "find_experts",
                    "description": "Find users who are domain experts on a concept, ranked by document contributions and relevance.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "concept": {"type": "string", "description": "Concept or topic name"},
                            "limit": {"type": "integer", "description": "Maximum number of experts to return (default 5)"},
                        },
                        "required": ["concept"],
                    }
                }
            },
            fn=find_experts,
        ),
        _OmniTool(
            schema={
                "type": "function",
                "function": {
                    "name": "get_entity_documents",
                    "description": "List documents linked to a specific entity (person, org, technology).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_name": {"type": "string", "description": "Entity name to look up"},
                            "limit": {"type": "integer", "description": "Maximum results (default 10)"},
                        },
                        "required": ["entity_name"],
                    }
                }
            },
            fn=get_entity_documents,
        ),
        _OmniTool(
            schema={
                "type": "function",
                "function": {
                    "name": "find_related_concepts",
                    "description": "Get concepts related to a given concept via hierarchy and co-occurrence in documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "concept": {"type": "string", "description": "Concept name"},
                        },
                        "required": ["concept"],
                    }
                }
            },
            fn=find_related_concepts,
        ),
        _OmniTool(
            schema={
                "type": "function",
                "function": {
                    "name": "get_document_content",
                    "description": "Fetch the full text content of a document by ID. Use after search when you need to read the actual content. Requires read access.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "integer", "description": "Document ID"},
                            "max_chars": {"type": "integer", "description": "Maximum characters to return (default 4000)"},
                        },
                        "required": ["document_id"],
                    }
                }
            },
            fn=get_document_content,
        ),
    ]

VERIFIED_FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-7b-instruct:free",
    "deepseek/deepseek-chat:free",
]

FREE_MODELS = VERIFIED_FREE_MODELS  # backward compatibility alias


class AnthropicOmniGraphAgent:
    _SYSTEM = """\
You are OmniGraph Assistant, an AI that answers questions from an enterprise knowledge graph.

## RAG Workflow â€” follow this order for every factual question:
1. **Search first**: call hybrid_search with the user's topic/question to find candidate documents.
2. **Read before answering**: for each promising result, call get_document_content(doc_id) to fetch the full text. Do not answer from titles or summaries alone.
3. **Cite sources**: every factual claim in your answer must include a [doc_id=X] citation referencing the document you read.
4. **Explore the graph**: use find_related_concepts, get_entity_documents, or find_experts when the user's question involves entities, relationships, or expertise.

## Output format:
- Lead with a direct answer to the question.
- Follow with supporting details and [doc_id=X] citations.
- If no relevant documents were found after searching, say so clearly rather than guessing.
- Keep responses concise unless the user asks for depth.
"""

    def __init__(
        self,
        db: DatabaseConnection,
        user_id: int,
        model: str = "",
    ) -> None:
        self.db = db
        self.user_id = user_id
        self.access_manager = AccessControlManager(db)
        self.query_engine = SemanticQueryEngine(db, user_id=user_id)

        tools = _create_tools(self.query_engine, self.access_manager, user_id, db)
        self._tool_map: Dict[str, Callable] = {t.schema["function"]["name"]: t.fn for t in tools}
        self._openai_tools: List[Dict[str, Any]] = [t.schema for t in tools]

        self.client: Optional[openai.OpenAI] = None
        self._models: List[str] = []
        self._current_model_idx = 0
        self._init_client(model)

    def _init_client(self, model_pref: str = "") -> None:
        provider = (settings.llm_provider or "auto").lower()

        # 1. OpenAI direct
        if (provider in ("auto", "openai")) and settings.openai_api_key:
            try:
                self.client = openai.OpenAI(
                    api_key=settings.openai_api_key,
                    max_retries=1,
                )
                self._models = [settings.llm_model or model_pref or "gpt-4o-mini"]
                logger.info("OmniGraph Agent initialized with OpenAI (%s).", self._models[0])
                return
            except Exception as exc:
                logger.warning("Failed to initialize OpenAI client: %s", exc)

        # 2. OpenRouter
        if (provider in ("auto", "openrouter")) and settings.openrouter_api_key:
            try:
                self.client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=settings.openrouter_api_key,
                    max_retries=0,
                )
                models: List[str] = []
                if settings.llm_model:
                    models.append(settings.llm_model)
                if model_pref and model_pref not in models:
                    models.append(model_pref)
                for m in VERIFIED_FREE_MODELS:
                    if m not in models:
                        models.append(m)
                self._models = models
                logger.info("OmniGraph Agent initialized with OpenRouter (primary: %s).", self._models[0])
                return
            except Exception as exc:
                logger.warning("Failed to initialize OpenRouter client: %s", exc)

        # 3. Ollama local
        if (provider in ("auto", "ollama")) and settings.ollama_base_url:
            try:
                self.client = openai.OpenAI(
                    base_url=settings.ollama_base_url,
                    api_key="ollama",
                    max_retries=1,
                )
                self._models = [settings.llm_model or model_pref or "llama3"]
                logger.info("OmniGraph Agent initialized with Ollama (%s).", self._models[0])
                return
            except Exception as exc:
                logger.warning("Failed to initialize Ollama client: %s", exc)

        logger.info("No remote LLM configured. OmniGraph Agent will use local extractive RAG.")

    def _rotate_model(self, error_msg: str) -> Optional[str]:
        if not self._models or len(self._models) <= 1:
            return None
        old_model = self._models[self._current_model_idx]
        self._current_model_idx = (self._current_model_idx + 1) % len(self._models)
        new_model = self._models[self._current_model_idx]
        logger.warning("Model %s failed (%s). Switched to %s.", old_model, error_msg, new_model)
        return new_model

    def _local_extractive_run(
        self,
        question: str,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_text_chunk: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Perform deterministic knowledge graph retrieval and extract a cited answer without remote LLM."""
        tools_used: List[Dict[str, Any]] = []

        if on_tool_call:
            on_tool_call("hybrid_search", {"query": question, "limit": 5})
        tools_used.append({"name": "hybrid_search", "input": {"query": question, "limit": 5}})

        search_results = self.query_engine.search(question, strategy="hybrid", limit=5)
        readable_docs = [
            r for r in search_results
            if r.get("document_id") is not None
            and self.access_manager.check_access(self.user_id, "document", r["document_id"], "read")
        ]

        fetched_contents: Dict[int, str] = {}
        for d in readable_docs[:3]:
            doc_id = d["document_id"]
            if on_tool_call:
                on_tool_call("get_document_content", {"document_id": doc_id, "max_chars": 2000})
            tools_used.append({"name": "get_document_content", "input": {"document_id": doc_id}})
            content_str = self._tool_map["get_document_content"](doc_id, max_chars=2000)
            fetched_contents[doc_id] = content_str

        expert_mentions: List[str] = []
        if any(w in question.lower() for w in ["expert", "who", "lead", "engineer", "author", "specialist"]):
            keywords = [w for w in re.findall(r"\b\w+\b", question) if len(w) > 3 and w.lower() not in ("about", "what", "which", "where")]
            for kw in keywords[:2]:
                if on_tool_call:
                    on_tool_call("find_experts", {"concept": kw, "limit": 3})
                exp_res = self._tool_map["find_experts"](kw, limit=3)
                if exp_res and "No experts" not in exp_res:
                    expert_mentions.append(f"Experts for **{kw}**:\n{exp_res}")
                    tools_used.append({"name": "find_experts", "input": {"concept": kw}})
                    break

        if not readable_docs:
            answer = (
                f"No accessible documents in OmniGraph matched the query: **{question}**.\n\n"
                "Please verify your query terms or verify account read permissions."
            )
        else:
            sections: List[str] = []
            top = readable_docs[0]
            summary_snippet = (top.get("summary") or "").strip()

            sections.append(f"Based on OmniGraph knowledge base retrieval, here is what was found regarding **{question}**:\n")
            if summary_snippet:
                sections.append(f"> {summary_snippet} [doc_id={top['document_id']}]\n")

            sections.append("### Key Findings & Document Evidence")
            for doc in readable_docs:
                doc_id = doc["document_id"]
                title = doc.get("title", f"Document #{doc_id}")
                stype = doc.get("source_type", "doc")
                content_text = fetched_contents.get(doc_id, "")

                paragraphs = [p.strip() for p in content_text.split("\n\n") if len(p.strip()) > 30 and not p.startswith("Title:")]
                detail = paragraphs[0] if paragraphs else (doc.get("summary") or f"Reference document for {title}")
                sections.append(f"- **{title}** (`{stype}`): {detail} [doc_id={doc_id}]")

            if expert_mentions:
                sections.append("\n### Identified Domain Experts")
                sections.extend(expert_mentions)

            answer = "\n\n".join(sections)

        if on_text_chunk:
            chunk_size = 64
            for i in range(0, len(answer), chunk_size):
                on_text_chunk(answer[i : i + chunk_size])

        citations = self._extract_citations(answer)
        return {
            "answer": answer,
            "citations": citations,
            "tools_used": tools_used,
            "stop_reason": "end_turn",
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

    def run(
        self,
        question: str,
        *,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_text_chunk: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        if not self.client or not self._models:
            return self._local_extractive_run(question, on_tool_call=on_tool_call, on_text_chunk=on_text_chunk)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._SYSTEM},
            {"role": "user", "content": question}
        ]
        tools_used: List[Dict[str, Any]] = []

        max_turns = 6
        model_retries = 0
        max_model_retries = max(len(self._models), 2)

        for _ in range(max_turns):
            model = self._models[self._current_model_idx]
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=self._openai_tools,
                    stream=True,
                )

                full_text = ""
                tool_calls: Dict[int, Dict[str, Any]] = {}
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_text += delta.content
                        if on_text_chunk:
                            on_text_chunk(delta.content)
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index if tc.index is not None else 0
                            if idx not in tool_calls:
                                tool_calls[idx] = {
                                    "id": tc.id or f"call_{idx}",
                                    "function": {"name": tc.function.name or "", "arguments": ""},
                                }
                            if tc.function and tc.function.name:
                                tool_calls[idx]["function"]["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                tool_calls[idx]["function"]["arguments"] += tc.function.arguments

                assistant_msg: Dict[str, Any] = {"role": "assistant"}
                if full_text:
                    assistant_msg["content"] = full_text

                if not tool_calls:
                    messages.append(assistant_msg)
                    break

                tcs_list = [v for _, v in sorted(tool_calls.items())]
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in tcs_list
                ]
                messages.append(assistant_msg)

                for tc in tcs_list:
                    name = tc["function"]["name"]
                    args_str = tc["function"]["arguments"]
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except Exception:
                        args = {}

                    if on_tool_call:
                        on_tool_call(name, args)

                    fn = self._tool_map.get(name)
                    if fn:
                        try:
                            res = fn(**args)
                        except Exception as e:
                            res = f"Tool error: {e}"
                    else:
                        res = f"Unknown tool: {name}"

                    tools_used.append({"name": name, "input": args})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": name,
                        "content": str(res),
                    })

            except (openai.APIStatusError, openai.OpenAIError, Exception) as exc:
                if isinstance(exc, openai.APIStatusError) and exc.status_code in (401, 403):
                    logger.warning("LLM authentication failed (%s); switching to local extractive RAG immediately.", exc)
                    return self._local_extractive_run(question, on_tool_call=on_tool_call, on_text_chunk=on_text_chunk)

                model_retries += 1
                logger.warning("Model invocation failed (%s). Retry %d/%d.", exc, model_retries, max_model_retries)
                next_model = self._rotate_model(str(exc))
                if model_retries >= max_model_retries or not next_model:
                    logger.info("Falling back to deterministic local extractive RAG.")
                    return self._local_extractive_run(question, on_tool_call=on_tool_call, on_text_chunk=on_text_chunk)
                time.sleep(0.5)


        answer = messages[-1].get("content", "") if messages else ""
        if not answer:
            return self._local_extractive_run(question, on_tool_call=on_tool_call, on_text_chunk=on_text_chunk)

        citations = self._extract_citations(answer)
        return {
            "answer": answer,
            "citations": citations,
            "tools_used": tools_used,
            "stop_reason": "end_turn",
            "messages": messages,
        }

    def _extract_citations(self, answer: str) -> List[Dict[str, Any]]:
        ids: List[int] = []
        seen = set()
        for m in re.finditer(r"\[doc_id=(\d+)\]", answer):
            doc_id = int(m.group(1))
            if doc_id not in seen:
                seen.add(doc_id)
                ids.append(doc_id)
        if not ids:
            return []
        try:
            with self.db.conn.cursor() as cur:
                cur.execute(
                    "SELECT document_id, title, source_type FROM omnigraph.documents WHERE document_id = ANY(%s)",
                    (ids,),
                )
                rows = {r[0]: {"document_id": r[0], "title": r[1], "source_type": r[2]} for r in cur.fetchall()}
        except Exception:
            try:
                self.db.conn.rollback()
            except Exception:
                pass
            rows = {}
        return [rows.get(i, {"document_id": i, "title": "(unknown)", "source_type": ""}) for i in ids]


def get_anthropic_agent(
    db: DatabaseConnection,
    user_id: int,
    model: str = "",
) -> AnthropicOmniGraphAgent:
    """Factory creating an OmniGraph agent instance with resilient fallback."""
    return AnthropicOmniGraphAgent(db, user_id, model=model)


__all__ = ["AnthropicOmniGraphAgent", "get_anthropic_agent", "_create_tools", "_format_docs"]

