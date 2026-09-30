"""
FinTech AI - shared agent call: approval gate -> chat model -> independent grader -> trace.
"""

from __future__ import annotations

import time
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.guard import call_llm
from src.llm.provider import get_chat_model, message_text
from src.mrm.grader import deterministic_grade, format_sources, grade_output
from src.mrm.model_inventory import current_version, require_approved
from src.mrm.trace import TraceRecord, log_trace
from src.prompts import registry

SYSTEM_MESSAGE = (
    "You are part of an internal hedge fund analytics demo that runs on synthetic data. "
    "Only use numbers provided in the user message."
)


def run_llm_agent(
    inventory_id: str,
    prompt_name: str,
    sources: List[dict],
    *,
    llm=None,
    grader_llm=None,
    approvals: Optional[dict] = None,
    enforce_approval: bool = True,
    grade: bool = True,
    **prompt_kwargs,
) -> dict:
    started = time.perf_counter()
    cur = current_version(inventory_id)
    if enforce_approval:
        approval = require_approved(inventory_id, cur["model_id"], cur["prompt_versions"], approvals)
    else:
        approval = {"status": "EVALUATION_RUN", "reason": "approval gate bypassed by the eval harness"}

    prompt = registry.get(prompt_name)
    text = prompt.format(data=format_sources(sources), **prompt_kwargs)
    msg = call_llm(llm or get_chat_model("chat"), [SystemMessage(content=SYSTEM_MESSAGE), HumanMessage(content=text)])
    output = message_text(msg).strip()
    grader = grade_output(output, sources, llm=grader_llm) if grade else deterministic_grade(output, sources)
    trace = TraceRecord(
        inventory_id=inventory_id,
        provider=cur["provider"],
        model_id=cur["model_id"],
        prompt_versions=[prompt.ref],
        mode="live",
        source_ids=[s["source_id"] for s in sources],
        grader={k: grader.get(k) for k in ("groundedness", "numeric_consistency", "verdict", "llm_verdict",
                                           "unsupported_claims", "grader_model", "grader_prompt")}
        | {"numeric_mismatches": grader["numeric_check"]["mismatches"], "grader_error": grader.get("grader_error")},
        approval_status=approval["status"],
        approval_reason=approval["reason"],
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return {"text": output, "grader": grader, "trace": log_trace(trace), "sources": sources}
