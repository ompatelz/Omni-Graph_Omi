# RAG Pipeline — OpenRouter backend
from __future__ import annotations

import json
import logging
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

FREE_MODELS = [
    "openrouter/free",
    "google/gemma-4-31b-it:free",
]

class AnthropicOmniGraphAgent:
    _SYSTEM = """\
You are OmniGraph Assistant, an AI that answers questions from an enterprise knowledge graph.

## RAG Workflow — follow this order for every factual question:
1. **Search first**: call hybrid_search with the user's topic/question to find candidate documents.
2. **Read before answering**: for each promising result, call get_document_content(doc_id) to fetch the full text. Do not answer from titles or summaries alone.
3. **Cite sources**: every factual claim in your answer must include a [doc_id=X] citation referencing the document you read.
4. **Explore the graph**: use ind_related_concepts, get_entity_documents, or ind_experts when the user's question involves entities, relationships, or expertise.

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
        
        # We rotate models from FREE_MODELS
        self._current_model_idx = 0
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            max_retries=0
        )
        self.access_manager = AccessControlManager(db)
        self.query_engine = SemanticQueryEngine(db, user_id=user_id)
        tools = _create_tools(self.query_engine, self.access_manager, user_id, db)
        self._tool_map: Dict[str, Callable] = {t.schema["function"]["name"]: t.fn for t in tools}
        self._openai_tools: List[Dict[str, Any]] = [t.schema for t in tools]

    def _rotate_model(self, error_msg: str) -> str:
        old_model = FREE_MODELS[self._current_model_idx]
        self._current_model_idx = (self._current_model_idx + 1) % len(FREE_MODELS)
        new_model = FREE_MODELS[self._current_model_idx]
        logger.warning(f"Model {old_model} failed ({error_msg}). Switched to {new_model}.")
        return new_model

    def run(
        self,
        question: str,
        *,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_text_chunk: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._SYSTEM},
            {"role": "user", "content": question}
        ]
        tools_used: List[Dict[str, Any]] = []

        max_attempts = len(FREE_MODELS) * 3
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            model = FREE_MODELS[self._current_model_idx]
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=self._openai_tools,
                    stream=True,
                )
                
                # Accumulate stream
                full_text = ""
                tool_calls = {}
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_text += delta.content
                        if on_text_chunk:
                            on_text_chunk(delta.content)
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            if tc.index not in tool_calls:
                                tool_calls[tc.index] = {"id": tc.id, "function": {"name": tc.function.name, "arguments": ""}}
                            if tc.function.arguments:
                                tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

                assistant_msg = {"role": "assistant"}
                if full_text:
                    assistant_msg["content"] = full_text
                
                if not tool_calls:
                    messages.append(assistant_msg)
                    break

                # Execute tools
                tcs_list = [v for k, v in sorted(tool_calls.items())]
                assistant_msg["tool_calls"] = [{"id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}} for tc in tcs_list]
                messages.append(assistant_msg)
                
                for tc in tcs_list:
                    name = tc["function"]["name"]
                    args_str = tc["function"]["arguments"]
                    try:
                        args = json.loads(args_str)
                    except:
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
                        "content": str(res)
                    })
            except openai.APIStatusError as e:
                if e.status_code in (404, 429, 502, 503, 529):
                    self._rotate_model(f"HTTP {e.status_code}")
                    time.sleep(0.5)
                else:
                    raise
            except Exception as e:
                self._rotate_model(str(e))
                time.sleep(0.5)
        raise RuntimeError("Agent failed after trying all configured free models.")

        answer = messages[-1].get("content", "")
        citations = self._extract_citations(answer)
        return {
            "answer": answer,
            "citations": citations,
            "tools_used": tools_used,
            "stop_reason": "end_turn",
            "messages": messages,
        }

    def _extract_citations(self, answer: str) -> List[Dict[str, Any]]:
        ids = []
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
                    "SELECT document_id, title, source_type FROM omnigraph.documents "
                    "WHERE document_id = ANY(%s)",
                    (ids,),
                )
                rows = {r[0]: {"document_id": r[0], "title": r[1], "source_type": r[2]}
                        for r in cur.fetchall()}
        except Exception:
            try:
                self.db.conn.rollback()
            except Exception:
                pass
            rows = {}
        return [rows.get(i, {"document_id": i, "title": "(unknown)", "source_type": ""})
                for i in ids]


def get_anthropic_agent(
    db: DatabaseConnection,
    user_id: int,
    model: str = "",
) -> Optional[AnthropicOmniGraphAgent]:
    if not settings.openrouter_api_key:
        return None
    return AnthropicOmniGraphAgent(db, user_id, model=model)


__all__ = ["AnthropicOmniGraphAgent", "get_anthropic_agent", "_create_tools", "_format_docs"]
