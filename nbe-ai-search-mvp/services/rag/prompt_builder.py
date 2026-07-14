"""Prompt builder for grounded NBE RAG generation."""

from __future__ import annotations

from services.llm_service.llm import RAG_USER_PROMPT, SYSTEM_PROMPT


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_rag_prompt(query: str, context: str, language: str) -> str:
    return RAG_USER_PROMPT.format(context=context, language=language, query=query)
