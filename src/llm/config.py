"""
FinTech AI - LLM provider configuration.

Settings are read from Streamlit secrets (``st.secrets``) first, then from
environment variables. The API key is never logged or echoed.

    [llm]    provider = "gemini" | "bedrock"      (env LLM_PROVIDER)
    [gemini] api_key  = "..."                     (env GEMINI_API_KEY)

Model roles
-----------
chat      - answers user questions and writes agent reports
grader    - independent validator (a *different* model than ``chat``)
hyde      - writes the hypothetical passage used for HyDE retrieval
embedding - document / query embeddings for the FAISS index
"""

from __future__ import annotations

import os
from typing import Optional

DEFAULT_PROVIDER = "gemini"

# The spec'd gemini-2.5-flash / gemini-2.5-flash-lite return HTTP 404 "no longer
# available to new users" for new API keys (checked 2026-09-29), and the free tier
# of gemini-3.5-flash allows only 20 requests/day/project (HTTP 429 observed at 20),
# which a public demo + eval cannot fit. Defaults are therefore two different
# Flash-Lite generations: answers on gemini-3.5-flash-lite, the independent grader
# and HyDE writer on gemini-3.1-flash-lite (quotas are per model). Override per role with
# st.secrets["llm"]["chat_model" | "grader_model" | "hyde_model" | "embedding_model"]
# or env LLM_CHAT_MODEL / LLM_GRADER_MODEL / LLM_HYDE_MODEL / LLM_EMBEDDING_MODEL.
# The grader must stay a different model from the chat model (independence).
GEMINI_MODELS = {
    "chat": "gemini-3.5-flash-lite",
    "grader": "gemini-3.1-flash-lite",
    "hyde": "gemini-3.1-flash-lite",
    "embedding": "gemini-embedding-001",
}

# Production architecture (see infra/). Not used by the free demo.
BEDROCK_MODELS = {
    "chat": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "grader": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "hyde": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "embedding": "amazon.titan-embed-text-v1",  # 1536-dim
}

# Gemini embedding task types
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

# Generation settings (kept in config so they are part of the traced setup)
CHAT_TEMPERATURE = 0.1
GRADER_TEMPERATURE = 0.0
HYDE_TEMPERATURE = 0.3
REQUEST_TIMEOUT_S = 90

FREE_TIER_LABEL = (
    "Demo runs on Gemini 2.5 Flash (free tier) + Gemini embeddings; production "
    "architecture targets Amazon Bedrock (Claude) + Titan — see infra/"
)


def _read_secret(section: str, key: str) -> Optional[str]:
    """Read ``st.secrets[section][key]`` if Streamlit and the secret exist."""
    try:
        import streamlit as st  # local import: CLI/API may not run Streamlit

        value = st.secrets[section][key]
        return str(value) if value else None
    except Exception:
        return None


def get_provider() -> str:
    provider = _read_secret("llm", "provider") or os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER
    provider = provider.strip().lower()
    return provider if provider in ("gemini", "bedrock") else DEFAULT_PROVIDER


def get_api_key() -> Optional[str]:
    """Gemini API key from st.secrets["gemini"]["api_key"] or env GEMINI_API_KEY."""
    return _read_secret("gemini", "api_key") or os.environ.get("GEMINI_API_KEY") or None


def has_credentials() -> bool:
    """True when a live call is possible for the configured provider."""
    if get_provider() == "gemini":
        return bool(get_api_key())
    # Bedrock relies on the standard AWS credential chain; we cannot cheaply
    # verify it without a network call, so treat it as available.
    return True


def model_id(role: str) -> str:
    override = _read_secret("llm", f"{role}_model") or os.environ.get(f"LLM_{role.upper()}_MODEL")
    if override:
        return override.strip()
    models = GEMINI_MODELS if get_provider() == "gemini" else BEDROCK_MODELS
    return models[role]


def display_name(mid: str) -> str:
    """'gemini-3.5-flash' -> 'Gemini 3.5 Flash'."""
    return " ".join(p if p[:1].isdigit() else p.capitalize() for p in mid.split("-"))


def free_tier_label() -> str:
    return (
        f"Demo runs on {display_name(model_id('chat'))} (free tier) + Gemini embeddings; production "
        "architecture targets Amazon Bedrock (Claude) + Titan \u2014 see infra/"
    )


def all_model_ids() -> dict:
    return {role: model_id(role) for role in ("chat", "grader", "hyde", "embedding")}
