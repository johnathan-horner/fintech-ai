"""
FinTech AI - HyDE (Hypothetical Document Embeddings) retrieval.

1. ``generate_hypothetical_answer(question)``: an LLM (config role "hyde")
   writes a short, plausible *answer passage* in the style of the indexed records.
2. That passage is embedded as a document (RETRIEVAL_DOCUMENT) and used to
   search FAISS for the top k=12 chunks.
3. Additionally the raw question is embedded (RETRIEVAL_QUERY) and its top-12
   hits are merged in (deduplicated). This hybrid step is optional
   (``merge_raw_query``, default True) and improves recall for exact ticker /
   metric lookups; the cross-encoder re-ranker then orders the union.
"""

from __future__ import annotations

from typing import List, Optional

from src.llm.guard import call_llm, call_with_retry
from src.llm.provider import message_text
from src.prompts import registry

HYDE_TOP_K = 12


def _embed(fn, embedder):
    """Embedding calls never fall back to another provider: failures surface as EmbeddingUnavailable."""
    from src.llm.guard import sanitize_error
    from src.llm.router import EmbeddingUnavailable

    try:
        return call_with_retry(fn)
    except Exception as exc:
        name = getattr(embedder, "model_name", "embedding model")
        raise EmbeddingUnavailable(f"query embedding ({name}) failed: {sanitize_error(exc)}") from exc


def generate_hypothetical_answer(question: str, llm=None, provider: str = "gemini") -> str:
    """LLM (config role "hyde" on ``provider``) writes a short hypothetical answer passage."""
    if llm is None:
        from src.llm.provider import get_chat_model

        llm = get_chat_model("hyde", provider)
    prompt = registry.get("hyde_passage").format(question=question)
    return message_text(call_llm(llm, prompt)).strip()


def hyde_retrieve(
    question: str,
    index,
    embedder,
    llm=None,
    k: int = HYDE_TOP_K,
    merge_raw_query: bool = True,
    passage: Optional[str] = None,
) -> dict:
    """
    Returns {"passage", "candidates": [doc + faiss_score + retrieved_by]}.
    The HyDE search uses the *passage* embedding.
    """
    passage = passage if passage is not None else generate_hypothetical_answer(question, llm)
    passage_vec = _embed(lambda: embedder.embed_passage(passage), embedder)
    hits = index.search_by_vector(passage_vec, k=k)
    candidates: List[dict] = []
    seen = set()
    for doc, score in hits:
        seen.add(doc["source_id"])
        candidates.append({**doc, "faiss_score": score, "retrieved_by": "hyde"})
    if merge_raw_query:
        q_vec = _embed(lambda: embedder.embed_query(question), embedder)
        for doc, score in index.search_by_vector(q_vec, k=k):
            if doc["source_id"] not in seen:
                seen.add(doc["source_id"])
                candidates.append({**doc, "faiss_score": score, "retrieved_by": "query"})
    return {"passage": passage, "candidates": candidates}
