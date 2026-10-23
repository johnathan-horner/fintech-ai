"""
FinTech AI - TraceRecord: version tracing for every AI output.

Each record captures what produced an output (provider, model, prompt
versions, embedding model, index manifest hash), what it was grounded on
(source ids), how it was graded, and the approval status at run time.
Records are kept in a bounded in-process log (used by the API and by
monitoring) and, in Streamlit, in the per-session log.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Deque, List, Optional

INVENTORY_IDS = {
    "rag": "MRM-001",
    "market": "MRM-002",
    "risk": "MRM-003",
    "portfolio": "MRM-004",
    "orchestrator": "MRM-005",
}

PROCESS_TRACE_LOG: Deque[dict] = deque(maxlen=500)


@dataclass
class TraceRecord:
    inventory_id: str
    provider: str
    model_id: str
    prompt_versions: List[str]
    mode: str = "live"                      # live | demo | blocked
    answer_model: Optional[str] = None      # model that actually produced the output
    fallback_reason: Optional[str] = None   # why earlier providers in the chain were not used
    embedding_model: Optional[str] = None
    index_manifest_hash: Optional[str] = None
    source_ids: List[str] = field(default_factory=list)
    grader: Optional[dict] = None
    approval_status: Optional[str] = None
    approval_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)


def log_trace(record: "TraceRecord | dict") -> dict:
    d = record.to_dict() if isinstance(record, TraceRecord) else dict(record)
    PROCESS_TRACE_LOG.append(d)
    return d


def recent_traces(limit: int = 100) -> List[dict]:
    return list(PROCESS_TRACE_LOG)[-limit:]


def attach_demo_trace(inventory_id: str, payload: dict, reason: str, blocked: bool) -> None:
    """Give a demo payload ({"text", "sources"}) a numeric-check grade and a demo/blocked TraceRecord."""
    if not isinstance(payload, dict) or "text" not in payload or "trace" in payload:
        return
    from src.mrm.grader import deterministic_grade

    grade = deterministic_grade(payload["text"], payload.get("sources", []))
    payload["grader"] = grade
    payload["trace"] = log_trace(TraceRecord(
        inventory_id=inventory_id, provider="demo-template", model_id="deterministic-template",
        prompt_versions=[], mode="blocked" if blocked else "demo",
        source_ids=[s["source_id"] for s in payload.get("sources", [])], grader=grade,
        approval_status="BLOCKED" if blocked else None, approval_reason=reason,
        error=None if blocked else reason,
    ))
