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


GRADER_TRACE_KEYS = ("groundedness", "numeric_consistency", "verdict", "llm_verdict", "unsupported_claims",
                     "grader_provider", "grader_model", "grader_prompt", "grader_error")


def grader_for_trace(grader: dict) -> dict:
    return {k: grader.get(k) for k in GRADER_TRACE_KEYS} | {"numeric_mismatches": grader["numeric_check"]["mismatches"]}


def run_llm_agent(
    inventory_id: str,
    prompt_name: str,
    sources,
    *,
    provider: str = "gemini",
    llm=None,
    grader_llm=None,
    grader_provider: Optional[str] = None,
    grader_model: Optional[str] = None,
    approvals: Optional[dict] = None,
    enforce_approval: bool = True,
    grade: bool = True,
    **prompt_kwargs,
) -> dict:
    """
    One agent call on ONE provider: approval gate -> model -> independent grader -> trace.
    ``prompt_name`` is the base name; the provider's variant is selected (e.g. *_haiku on Bedrock).
    ``sources`` may be a list or a callable(provider) -> list (provider-specific context).
    """
    from src.prompts.templates import prompt_name_for

    started = time.perf_counter()
    if callable(sources):
        sources = sources(provider)
    prompt_name = prompt_name_for(prompt_name, provider)
    cur = current_version(inventory_id, provider)
    if enforce_approval:
        approval = require_approved(inventory_id, cur["model_id"], cur["prompt_versions"], approvals, provider=provider)
    else:
        approval = {"status": "EVALUATION_RUN", "reason": "approval gate bypassed by the eval harness"}

    prompt = registry.get(prompt_name)
    text = prompt.format(data=format_sources(sources), **prompt_kwargs)
    msg = call_llm(llm or get_chat_model("chat", provider),
                   [SystemMessage(content=SYSTEM_MESSAGE), HumanMessage(content=text)])
    output = message_text(msg).strip()
    grader = (grade_output(output, sources, llm=grader_llm, answer_provider=provider, answer_model=cur["model_id"],
                           grader_provider=grader_provider, grader_model=grader_model)
              if grade else deterministic_grade(output, sources))
    trace = TraceRecord(
        inventory_id=inventory_id,
        provider=provider,
        model_id=cur["model_id"],
        answer_model=cur["model_id"],
        prompt_versions=[prompt.ref],
        mode="live",
        source_ids=[s["source_id"] for s in sources],
        grader=grader_for_trace(grader),
        approval_status=approval["status"],
        approval_reason=approval["reason"],
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return {"text": output, "grader": grader, "trace": log_trace(trace), "sources": sources,
            "provider": provider, "answer_model": cur["model_id"]}


def run_agent_component(
    inventory_id: str,
    prompt_name: str,
    sources,
    *,
    provider: Optional[str] = None,
    providers: Optional[List[str]] = None,
    llm=None,
    **kw,
) -> dict:
    """
    Provider chain for an agent (Bedrock Haiku 4.5 -> Gemini). With an injected ``llm``
    (tests / eval harness) exactly one provider is used - no fallback.
    """
    if llm is not None or provider is not None:
        return run_llm_agent(inventory_id, prompt_name, sources, provider=provider or "gemini", llm=llm, **kw)
    from src.llm.router import run_component

    return run_component(
        inventory_id,
        lambda prov: run_llm_agent(inventory_id, prompt_name, sources, provider=prov, **kw),
        providers=providers, approvals=kw.get("approvals"), enforce_approval=kw.get("enforce_approval", True),
    )
