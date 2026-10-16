"""
FinTech AI - LLM provider configuration.

Provider chain for every LLM call (answers, agents, HyDE):
    1. Amazon Bedrock  - Claude Haiku 4.5 via the us.* inference profile (primary)
    2. Google Gemini   - free tier (fallback)
    3. demo mode       - deterministic, data-derived templates (last resort)

Credentials (never logged or echoed):
    Bedrock  - inside Streamlit: st.secrets["aws"] = {access_key_id, secret_access_key, region}
               ONLY (no fallthrough to instance or other credentials on Cloud);
               CLI / eval / API: the default boto3 credential chain.
    Gemini   - st.secrets["gemini"]["api_key"] or env GEMINI_API_KEY.

Embeddings always use gemini-embedding-001 (the FAISS index is built on it).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

PROVIDERS = ("bedrock", "gemini")
DEFAULT_PROVIDER_ORDER = ("bedrock", "gemini")

HAIKU_45 = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
HAIKU_45_FOUNDATION_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_DEFAULT_REGION = "us-east-1"

# The spec'd gemini-2.5-flash / gemini-2.5-flash-lite return HTTP 404 "no longer
# available to new users" for new API keys (checked 2026-09-29), and the free tier
# of gemini-3.5-flash allows only 20 requests/day/project. Fallback defaults are two
# different Flash-Lite generations. Override per role with st.secrets["llm"]["<role>_model"]
# or env LLM_<ROLE>_MODEL (Gemini roles only).
GEMINI_MODELS = {
    "chat": "gemini-3.5-flash-lite",
    "grader": "gemini-3.1-flash-lite",
    "hyde": "gemini-3.1-flash-lite",
    "embedding": "gemini-embedding-001",
}

# Always the us.* inference profile (the app's IAM policy allows bedrock:InvokeModel only on this
# profile and its us-east-1/us-east-2/us-west-2 foundation-model ARNs; streaming and the global.*
# profile are denied, so calls are non-streaming Converse only).
BEDROCK_MODELS = {
    "chat": HAIKU_45,
    "grader": HAIKU_45,
    "hyde": HAIKU_45,
}

# Max output tokens on every call (Gemini caps are larger because thinking tokens count).
MAX_OUTPUT_TOKENS = {
    "bedrock": {"chat": 1024, "hyde": 256, "grader": 512},
    "gemini": {"chat": 2048, "hyde": 512, "grader": 1024},
}

# Claude Haiku 4.5, us.* inference profile, on-demand (AWS Pricing API), USD per 1M tokens.
BEDROCK_PRICE_PER_1M = {"input": 1.10, "output": 5.50}

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

CHAT_TEMPERATURE = 0.1
GRADER_TEMPERATURE = 0.0
HYDE_TEMPERATURE = 0.3
REQUEST_TIMEOUT_S = 90

DISPLAY_NAMES = {HAIKU_45: "Claude Haiku 4.5"}


def _read_secret(section: str, key: str) -> Optional[str]:
    """Read ``st.secrets[section][key]`` if Streamlit and the secret exist."""
    try:
        import streamlit as st  # local import: CLI/API may not run Streamlit

        value = st.secrets[section][key]
        return str(value) if value else None
    except Exception:
        return None


def _in_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime import exists

        return bool(exists())
    except Exception:
        return False


# -- Gemini ----------------------------------------------------------------------

def get_api_key() -> Optional[str]:
    """Gemini API key from st.secrets["gemini"]["api_key"] or env GEMINI_API_KEY."""
    return _read_secret("gemini", "api_key") or os.environ.get("GEMINI_API_KEY") or None


def gemini_available() -> bool:
    return bool(get_api_key())


# -- Bedrock ---------------------------------------------------------------------

def aws_streamlit_secrets() -> Optional[Dict[str, str]]:
    key_id = _read_secret("aws", "access_key_id")
    secret = _read_secret("aws", "secret_access_key")
    if key_id and secret:
        return {"aws_access_key_id": key_id, "aws_secret_access_key": secret,
                "region_name": _read_secret("aws", "region") or BEDROCK_DEFAULT_REGION}
    return None


def bedrock_client_kwargs() -> Optional[Dict[str, str]]:
    """
    boto3 client kwargs for Bedrock, or None when Bedrock must be skipped.
    Inside Streamlit only [aws] secrets are used (never instance / other credentials).
    """
    secrets = aws_streamlit_secrets()
    if secrets:
        return secrets
    if _in_streamlit_runtime():
        return None
    try:
        import boto3

        if boto3.Session().get_credentials() is None:
            return None
    except Exception:  # boto3 not installed or credential chain error
        return None
    return {"region_name": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            or BEDROCK_DEFAULT_REGION}


def bedrock_available() -> bool:
    return bedrock_client_kwargs() is not None


# -- Chain -----------------------------------------------------------------------

def provider_order() -> List[str]:
    raw = _read_secret("llm", "providers") or os.environ.get("LLM_PROVIDERS")
    if raw:
        order = [p.strip().lower() for p in raw.split(",") if p.strip().lower() in PROVIDERS]
        if order:
            return order
    return list(DEFAULT_PROVIDER_ORDER)


def provider_available(provider: str) -> bool:
    return bedrock_available() if provider == "bedrock" else gemini_available() if provider == "gemini" else False


def available_providers() -> List[str]:
    return [p for p in provider_order() if provider_available(p)]


def has_credentials() -> bool:
    """True when at least one live provider is configured."""
    return bool(available_providers())


def get_provider() -> str:
    """Primary provider that is configured right now ("none" when neither is)."""
    avail = available_providers()
    return avail[0] if avail else "none"


def model_id(role: str, provider: Optional[str] = None) -> str:
    """Model id for ``role``. Embeddings are always Gemini; provider defaults to gemini."""
    if role == "embedding":
        provider = "gemini"  # the FAISS index is built on gemini-embedding-001
    provider = provider or "gemini"
    if provider == "gemini":
        override = _read_secret("llm", f"{role}_model") or os.environ.get(f"LLM_{role.upper()}_MODEL")
        if override:
            return override.strip()
        return GEMINI_MODELS[role]
    return BEDROCK_MODELS[role]


def max_output_tokens(role: str, provider: str) -> int:
    return MAX_OUTPUT_TOKENS[provider].get(role, MAX_OUTPUT_TOKENS[provider]["chat"])


def display_name(mid: str) -> str:
    """'gemini-3.5-flash-lite' -> 'Gemini 3.5 Flash Lite'; Haiku profile -> 'Claude Haiku 4.5'."""
    if mid in DISPLAY_NAMES:
        return DISPLAY_NAMES[mid]
    return " ".join(p if p[:1].isdigit() else p.capitalize() for p in mid.split("-"))


def provider_label(provider: Optional[str], mid: Optional[str]) -> str:
    """'Bedrock · Claude Haiku 4.5' / 'Gemini · gemini-3.5-flash-lite'."""
    if provider == "bedrock":
        return f"Bedrock · {display_name(mid or HAIKU_45)}"
    if provider == "gemini":
        return f"Gemini · {mid}"
    return f"{provider} · {mid}"


def free_tier_label() -> str:
    return (
        f"Live AI: Amazon Bedrock {display_name(HAIKU_45)} (primary) -> Gemini free tier "
        f"{GEMINI_MODELS['chat']} (fallback) -> demo mode; embeddings {GEMINI_MODELS['embedding']}. "
        "Bedrock Guardrails, KMS and CloudTrail are defined in infra/ (CDK) only and are not used by this live demo."
    )


def all_model_ids(provider: Optional[str] = None) -> dict:
    provider = provider or "gemini"
    return {role: model_id(role, provider) for role in ("chat", "grader", "hyde")} | {"embedding": model_id("embedding")}
