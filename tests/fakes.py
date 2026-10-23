"""Test doubles: fake chat model, embedder and cross-encoder (no network)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Callable, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def prompt_text(messages) -> str:
    if isinstance(messages, str):
        return messages
    if hasattr(messages, "to_string"):
        return messages.to_string()
    return "\n".join(getattr(m, "content", str(m)) for m in messages)


def _first_record(text: str, header: str) -> str:
    part = text.split(header, 1)[1] if header in text else text
    m = re.search(r"\[([^\]]+)\] ([^\n]+)", part)
    if not m:
        return "The indexed data does not contain this."
    sentence = m.group(2).split(". ")[0].rstrip(".")
    return f"{sentence} [{m.group(1)}]."


def default_responder(text: str) -> str:
    if "independent model validator" in text:
        return json.dumps({"groundedness": 0.95, "numeric_consistency": 1.0,
                           "unsupported_claims": [], "verdict": "PASS"})
    if "Passage:" in text and "hedge fund's" in text:
        q = text.rsplit("Question:", 1)[-1].split("Passage:")[0].strip()
        return f"Record: {q} Position ticker sector market value weight beta P&L risk score."
    if "SOURCES:" in text:
        return _first_record(text, "SOURCES:")
    if "DATA:" in text:
        return _first_record(text, "DATA:")
    return "ok"


class FakeChatModel(BaseChatModel):
    """A real LangChain chat model (Runnable, usable in LCEL pipes) with scripted replies."""

    responder: Callable[[str], str] = Field(default=default_responder)
    errors: List[Exception] = Field(default_factory=list)
    calls: List[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = prompt_text(messages)
        self.calls.append(text)
        if self.errors:
            raise self.errors.pop(0)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.responder(text)))])


class FakeEmbedder:
    """Deterministic hashed bag-of-words embeddings; records every call."""

    model_name = "fake-embedding"

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.calls: List[tuple] = []

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        if not any(v):
            v[0] = 1.0
        return v

    def embed_documents(self, texts):
        self.calls.append(("document", list(texts)))
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        self.calls.append(("query", text))
        return self._vec(text)

    def embed_passage(self, text):
        self.calls.append(("passage", text))
        return self._vec(text)


class FakeCrossEncoder:
    """Word-overlap scorer with the fastembed ``rerank`` interface."""

    def rerank(self, query, documents):
        q = set(re.findall(r"\w+", query.lower()))
        for d in documents:
            yield float(len(q & set(re.findall(r"\w+", d.lower()))))


class RateLimitError(Exception):
    """Mimics google ResourceExhausted / HTTP 429."""

    code = 429

    def __init__(self, msg="429 RESOURCE_EXHAUSTED: quota exceeded"):
        super().__init__(msg)


def fake_deps_factory(components, index_dir: Optional[str] = None, provider: str = "gemini") -> dict:
    import tempfile

    from src.rag.reranker import CrossEncoderReranker
    from src.rag.vector_store import build_index

    deps = {"llm": FakeChatModel(), "grader_llm": FakeChatModel(), "hyde_llm": FakeChatModel(),
            "provider": provider, "grader_provider": "gemini", "grader_model": "fake-grader"}
    if "MRM-001" in components:
        emb = FakeEmbedder()
        deps["embedder"] = emb
        deps["index"] = build_index(emb, path=index_dir or tempfile.mkdtemp(prefix="fake_index_"))
        deps["reranker"] = CrossEncoderReranker(encoder=FakeCrossEncoder())
    return deps
