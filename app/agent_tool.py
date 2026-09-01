"""MiniAgent-compatible MiniRAG tool adapter.

The adapter deliberately binds ``user_id`` outside of the model-provided
arguments.  A model may choose a knowledge base and formulate a question, but
it must not be able to impersonate a more privileged MiniRAG user.
"""

from __future__ import annotations

from typing import Any, Callable

from app.kb import KnowledgeBase


KNOWLEDGE_BASE_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "knowledge_base_search",
        "description": (
            "Search the selected MiniRAG knowledge base for private documents. "
            "Use this before answering questions that depend on uploaded or "
            "organization-specific material. The result includes an answer and "
            "the evidence snippets that support it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's factual question to search for.",
                },
                "kb_id": {
                    "type": "string",
                    "description": "Knowledge base ID. Omit to use the default knowledge base.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def make_knowledge_base_search(
    resolve_kb: Callable[[str | None], KnowledgeBase], user_id: str
) -> Callable[..., dict[str, Any]]:
    """Create one tool handler bound to the authenticated agent user.

    ``resolve_kb`` is supplied by the host application so this adapter has no
    dependency on FastAPI globals and is easy to embed in a CLI or MiniAgent.
    """
    if not user_id or not user_id.strip():
        raise ValueError("MiniAgent 会话缺少 MiniRAG 用户身份")

    def knowledge_base_search(query: str, kb_id: str = "") -> dict[str, Any]:
        result = resolve_kb(kb_id or None).ask(query, user_id=user_id.strip())
        return {
            "kb_id": kb_id or "main",
            "answer": result["answer"],
            "refusal": result["refusal"],
            "citations": result["citations"],
            "sources": [
                {
                    "citation": evidence["rank"],
                    "document_id": evidence["document_id"],
                    "label": evidence["label"],
                    "text": evidence["text"],
                }
                for evidence in result["evidence"]
            ],
        }

    return knowledge_base_search
