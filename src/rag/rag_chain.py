"""
FinTech AI - RAG chain (inventory MRM-001).

question -> approval gate -> HyDE retrieval (FAISS, k=12, + raw-query merge)
-> cross-encoder re-rank (top 4) -> answer (chat model, cites source ids)
-> independent grader -> TraceRecord.

Conversational memory: the last 5 turns of the session history are included
in the answer prompt.
"""

from __future__ import annotations

import time
from typing import List, Optional

from src.llm import config
from src.llm.guard import call_llm
from src.llm.provider import Embedder, get_chat_model, message_text
from src.mrm.grader import format_sources, grade_output
from src.mrm.model_inventory import current_version, require_approved
from src.mrm.trace import TraceRecord, log_trace
from src.prompts import registry
from src.rag.query_transform import hyde_retrieve
from src.rag.reranker import RERANK_TOP_N, get_reranker

INVENTORY_ID = "MRM-001"
MEMORY_TURNS = 5


def _format_history(history: Optional[List[dict]]) -> str:
    if not history:
        return "(none)"
    turns = history[-2 * MEMORY_TURNS:]
    return "\n".join(f"{m['role']}: {m['content'][:600]}" for m in turns)


def answer_question(
    question: str,
    history: Optional[List[dict]] = None,
    *,
    chat_llm=None,
    hyde_llm=None,
    grader_llm=None,
    embedder=None,
    index=None,
    reranker=None,
    approvals: Optional[dict] = None,
    enforce_approval: bool = True,
) -> dict:
    started = time.perf_counter()
    cur = current_version(INVENTORY_ID)
    if enforce_approval:
        approval = require_approved(INVENTORY_ID, cur["model_id"], cur["prompt_versions"], approvals)
    else:
        approval = {"status": "EVALUATION_RUN", "reason": "approval gate bypassed by the eval harness"}

    embedder = embedder or Embedder()
    if index is None:
        from src.rag.vector_store import get_index

        index = get_index(embedder)

    retrieval = hyde_retrieve(question, index, embedder, llm=hyde_llm or get_chat_model("hyde"))
    candidates = retrieval["candidates"]
    rerank_note = None
    try:
        ranked = (reranker or get_reranker()).rerank(question, candidates, top_n=RERANK_TOP_N)
    except Exception as exc:  # model download blocked etc. - keep FAISS order, but say so
        rerank_note = f"re-ranker unavailable ({type(exc).__name__}); FAISS order used"
        ranked = [(c, None) for c in sorted(candidates, key=lambda c: -c["faiss_score"])[:RERANK_TOP_N]]

    sources = [
        {"source_id": d["source_id"], "text": d["text"], "rerank_score": s,
         "faiss_score": round(d["faiss_score"], 4), "retrieved_by": d["retrieved_by"]}
        for d, s in ranked
    ]
    prompt = registry.get("rag_answer").format(
        history=_format_history(history), sources=format_sources(sources), question=question
    )
    answer = message_text(call_llm(chat_llm or get_chat_model("chat"), prompt)).strip()
    grader = grade_output(answer, sources, llm=grader_llm)

    trace = TraceRecord(
        inventory_id=INVENTORY_ID,
        provider=cur["provider"],
        model_id=cur["model_id"],
        prompt_versions=cur["prompt_versions"],
        mode="live",
        embedding_model=embedder.model_name,
        index_manifest_hash=getattr(index, "manifest_hash", None),
        source_ids=[s["source_id"] for s in sources],
        grader={k: grader.get(k) for k in ("groundedness", "numeric_consistency", "verdict", "llm_verdict",
                                           "unsupported_claims", "grader_model", "grader_prompt")}
        | {"numeric_mismatches": grader["numeric_check"]["mismatches"], "grader_error": grader.get("grader_error")},
        approval_status=approval["status"],
        approval_reason=approval["reason"],
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        extra={"hyde_model": config.model_id("hyde"), "candidates": len(candidates),
               "reranker": rerank_note or "Xenova/ms-marco-MiniLM-L-6-v2"},
    )
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "hypothetical_passage": retrieval["passage"],
        "grader": grader,
        "trace": log_trace(trace),
    }


def query(question: str, history: Optional[List[dict]] = None) -> dict:
    """Backwards-compatible entry point used by the API."""
    return answer_question(question, history)
