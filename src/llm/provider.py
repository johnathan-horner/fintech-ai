"""
FinTech AI - LLM provider factory.

``get_chat_model(role, provider)`` returns a chat model with ``.invoke(messages)``:
  * provider "bedrock" - ``BedrockConverseChat``: boto3 ``converse`` on the Claude Haiku 4.5
    us.* inference profile (boto3 is imported lazily; token usage is recorded for cost reporting)
  * provider "gemini"  - ``ChatGoogleGenerativeAI`` (langchain-google-genai)
Every call carries a max-output-tokens cap (config.MAX_OUTPUT_TOKENS).
``get_embeddings()`` is always Gemini (gemini-embedding-001): the FAISS index is built on it.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from src.llm import config


class MissingCredentialsError(RuntimeError):
    """Raised when no credentials are configured for the selected provider."""


# -- Bedrock token usage (for cost accounting in the eval harness) -------------------

_USAGE_LOCK = threading.Lock()
USAGE_LOG: List[dict] = []


def record_usage(entry: dict) -> None:
    with _USAGE_LOCK:
        USAGE_LOG.append(entry)


def usage_since(mark: int) -> List[dict]:
    with _USAGE_LOCK:
        return list(USAGE_LOG[mark:])


def usage_mark() -> int:
    with _USAGE_LOCK:
        return len(USAGE_LOG)


def bedrock_cost_usd(input_tokens: int, output_tokens: int) -> float:
    p = config.BEDROCK_PRICE_PER_1M
    return round(input_tokens / 1e6 * p["input"] + output_tokens / 1e6 * p["output"], 6)


_CLIENTS: dict = {}


def _bedrock_client(kwargs: dict):
    import boto3
    from botocore.config import Config

    key = tuple(sorted(kwargs.items()))
    if key not in _CLIENTS:
        _CLIENTS[key] = boto3.client(
            "bedrock-runtime",
            config=Config(retries={"max_attempts": 2, "mode": "standard"}, connect_timeout=10,
                          read_timeout=config.REQUEST_TIMEOUT_S),
            **kwargs,
        )
    return _CLIENTS[key]


def _to_converse(messages):
    """str | PromptValue | [BaseMessage] -> (system blocks, converse messages)."""
    if hasattr(messages, "to_messages"):
        messages = messages.to_messages()
    if isinstance(messages, str):
        return [], [{"role": "user", "content": [{"text": messages}]}]
    system, conv = [], []
    for m in messages:
        mtype = getattr(m, "type", "human")
        text = m.content if isinstance(m.content, str) else str(m.content)
        if mtype == "system":
            system.append({"text": text})
        else:
            conv.append({"role": "assistant" if mtype == "ai" else "user", "content": [{"text": text}]})
    return system, conv


class BedrockConverseChat:
    """
    Minimal chat model over the NON-streaming Bedrock ``converse`` API (authorized by
    bedrock:InvokeModel) on the Claude Haiku 4.5 us.* inference profile. There is deliberately
    no stream() method: InvokeModelWithResponseStream / ConverseStream are denied by the app's policy.
    """

    provider = "bedrock"

    def __init__(self, model_id: str, max_tokens: int, temperature: float, client_kwargs: dict, client=None):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client_kwargs = client_kwargs
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = _bedrock_client(self._client_kwargs)
        return self._client

    def invoke(self, messages, **_):
        from langchain_core.messages import AIMessage  # lazy: the MCP env has no langchain

        system, conv = _to_converse(messages)
        req = {"modelId": self.model_id, "messages": conv,
               "inferenceConfig": {"maxTokens": self.max_tokens, "temperature": self.temperature}}
        if system:
            req["system"] = system
        resp = self.client.converse(**req)
        text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
        usage = resp.get("usage", {})
        entry = {"provider": "bedrock", "model_id": self.model_id,
                 "input_tokens": int(usage.get("inputTokens", 0)), "output_tokens": int(usage.get("outputTokens", 0))}
        record_usage(entry)
        return AIMessage(
            content=text,
            usage_metadata={"input_tokens": entry["input_tokens"], "output_tokens": entry["output_tokens"],
                            "total_tokens": entry["input_tokens"] + entry["output_tokens"]},
            response_metadata={"model_id": self.model_id, "stop_reason": resp.get("stopReason")},
        )


def get_chat_model(role: str = "chat", provider: Optional[str] = None):
    """Return a chat model for ``role`` in {chat, grader, hyde} on ``provider`` (default gemini)."""
    provider = provider or "gemini"
    temperature = {
        "chat": config.CHAT_TEMPERATURE,
        "grader": config.GRADER_TEMPERATURE,
        "hyde": config.HYDE_TEMPERATURE,
    }.get(role, config.CHAT_TEMPERATURE)
    model = config.model_id(role, provider)
    max_tokens = config.max_output_tokens(role, provider)

    if provider == "bedrock":
        kwargs = config.bedrock_client_kwargs()
        if kwargs is None:
            raise MissingCredentialsError("No AWS credentials for Bedrock")
        return BedrockConverseChat(model, max_tokens, temperature, kwargs)

    key = config.get_api_key()
    if not key:
        raise MissingCredentialsError("No Gemini API key configured")
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs = dict(
        model=model,
        google_api_key=key,
        # max_retries=1 disables SDK retries; src/llm/guard.call_llm retries 429 once itself.
        max_retries=1,
        timeout=config.REQUEST_TIMEOUT_S,
        max_output_tokens=max_tokens,
    )
    if not model.startswith("gemini-3"):
        # Gemini 3.x models use fixed sampling defaults and ignore temperature.
        kwargs["temperature"] = temperature
    if role == "grader":
        kwargs["response_mime_type"] = "application/json"
    return ChatGoogleGenerativeAI(**kwargs)


def get_embeddings():
    """Gemini embeddings (gemini-embedding-001) - the FAISS index is built on them."""
    key = config.get_api_key()
    if not key:
        raise MissingCredentialsError("No Gemini API key configured (needed for gemini-embedding-001)")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=config.model_id("embedding"), google_api_key=key)


class Embedder:
    """
    Task-aware wrapper around a LangChain embeddings object.

    Gemini: documents (and HyDE passages) use RETRIEVAL_DOCUMENT, plain
    queries use RETRIEVAL_QUERY. Other providers ignore the task type.
    """

    def __init__(self, embeddings=None, model_name: str | None = None):
        self._embeddings = embeddings
        self.model_name = model_name or config.model_id("embedding")

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def _supports_task_type(self) -> bool:
        return type(self.embeddings).__name__ == "GoogleGenerativeAIEmbeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._supports_task_type():
            return self.embeddings.embed_documents(texts, task_type=config.TASK_DOCUMENT)
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if self._supports_task_type():
            return self.embeddings.embed_query(text, task_type=config.TASK_QUERY)
        return self.embeddings.embed_query(text)

    def embed_passage(self, text: str) -> List[float]:
        """Embed a HyDE passage as a *document* (RETRIEVAL_DOCUMENT)."""
        return self.embed_documents([text])[0]


def message_text(message) -> str:
    """Extract plain text from an AIMessage (str or list-of-parts content)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)
