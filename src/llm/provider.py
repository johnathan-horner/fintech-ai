"""
FinTech AI - LLM provider factory.

``get_chat_model(role)`` and ``get_embeddings()`` return LangChain objects.
Gemini (``langchain-google-genai``) is the default. The Bedrock path
(``langchain-aws``) is imported lazily and is NOT in the Cloud requirements.
"""

from __future__ import annotations

from typing import List

from src.llm import config


class MissingCredentialsError(RuntimeError):
    """Raised when no API key is configured for the selected provider."""


def get_chat_model(role: str = "chat"):
    """Return a LangChain chat model for ``role`` in {chat, grader, hyde}."""
    provider = config.get_provider()
    temperature = {
        "chat": config.CHAT_TEMPERATURE,
        "grader": config.GRADER_TEMPERATURE,
        "hyde": config.HYDE_TEMPERATURE,
    }.get(role, config.CHAT_TEMPERATURE)
    model = config.model_id(role)

    if provider == "bedrock":
        from langchain_aws import ChatBedrockConverse  # optional dependency

        return ChatBedrockConverse(model=model, temperature=temperature)

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
    )
    if not model.startswith("gemini-3"):
        # Gemini 3.x models use fixed sampling defaults and ignore temperature.
        kwargs["temperature"] = temperature
    if role == "grader":
        kwargs["response_mime_type"] = "application/json"
    return ChatGoogleGenerativeAI(**kwargs)


def get_embeddings():
    """Return a LangChain embeddings object for the configured provider."""
    provider = config.get_provider()
    model = config.model_id("embedding")
    if provider == "bedrock":
        from langchain_aws import BedrockEmbeddings  # optional dependency

        return BedrockEmbeddings(model_id=model)

    key = config.get_api_key()
    if not key:
        raise MissingCredentialsError("No Gemini API key configured")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=key)


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
