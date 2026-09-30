"""
FinTech AI - Multi-Agent Orchestrator (inventory MRM-005).

LangGraph StateGraph:  market_analysis -> risk_assessment -> portfolio_management -> END

The orchestrator is gated as MRM-005 and every agent is additionally gated by
its own approval record. If an agent is blocked or fails, its section falls
back to a deterministic, data-derived template and the reason is recorded;
the pipeline continues.
"""

from __future__ import annotations

import time
from typing import List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.market_agent import run_market_analysis
from src.agents.portfolio_agent import run_portfolio_management
from src.agents.risk_agent import compute_risk_state, run_risk_assessment
from src.agents.context import action_items, technical_alerts
from src.demo.templates import demo_market_report, demo_portfolio_recommendations, demo_risk_report
from src.llm.guard import GovernanceBlocked, is_rate_limit_error, sanitize_error
from src.mrm.grader import deterministic_grade
from src.mrm.model_inventory import current_version, require_approved
from src.mrm.trace import TraceRecord, log_trace
from src.utils.metrics import portfolio_kpis

INVENTORY_ID = "MRM-005"
SECTION_KEYS = ("market", "risk", "portfolio")


class OrchestratorState(TypedDict, total=False):
    portfolio_data: dict
    market: dict
    risk: dict
    portfolio: dict
    errors: List[str]


def _fallback_section(inventory_id: str, demo: dict, reason: str, blocked: bool) -> dict:
    grade = deterministic_grade(demo["text"], demo["sources"])
    trace = TraceRecord(
        inventory_id=inventory_id, provider="demo-template", model_id="deterministic-template",
        prompt_versions=[], mode="blocked" if blocked else "demo",
        source_ids=[s["source_id"] for s in demo["sources"]], grader=grade,
        approval_status="BLOCKED" if blocked else None, approval_reason=reason,
        error=None if blocked else reason,
    )
    return {"mode": "blocked" if blocked else "demo", "reason": reason, "text": demo["text"],
            "sources": demo["sources"], "grader": grade, "trace": log_trace(trace)}


def _live_section(result: dict, text_key: str) -> dict:
    return {"mode": "live", "reason": None, "text": result[text_key], "sources": result["sources"],
            "grader": result["grader"], "trace": result["trace"]}


def _run_section(inventory_id: str, live_fn, demo_fn, errors: List[str]) -> dict:
    try:
        return live_fn()
    except GovernanceBlocked as exc:
        errors.append(f"{inventory_id}: blocked - {exc}")
        return _fallback_section(inventory_id, demo_fn(), f"Blocked by model governance: {exc}", True)
    except Exception as exc:
        reason = ("Gemini free-tier rate limit / quota reached (HTTP 429, retried once)"
                  if is_rate_limit_error(exc) else f"Live AI call failed ({sanitize_error(exc)})")
        errors.append(f"{inventory_id}: {reason}")
        return _fallback_section(inventory_id, demo_fn(), reason, False)


def build_orchestrator(llm=None, grader_llm=None, approvals: Optional[dict] = None, enforce_approval: bool = True):
    kw = dict(llm=llm, grader_llm=grader_llm, approvals=approvals, enforce_approval=enforce_approval)

    def market_node(state: OrchestratorState) -> dict:
        errors = list(state.get("errors") or [])
        p = state["portfolio_data"]
        section = _run_section(
            "MRM-002",
            lambda: _live_section(run_market_analysis(p, **kw), "market_analysis"),
            lambda: demo_market_report(p), errors,
        )
        section["sector_breakdown"] = portfolio_kpis(p)["sector_weights"]
        section["technical_alerts"] = technical_alerts(p)
        return {"market": section, "errors": errors}

    def risk_node(state: OrchestratorState) -> dict:
        errors = list(state.get("errors") or [])
        p = state["portfolio_data"]
        market_text = state["market"]["text"] if state.get("market") else ""
        section = _run_section(
            "MRM-003",
            lambda: _live_section(run_risk_assessment(p, market_text, **kw), "risk_report"),
            lambda: demo_risk_report(p), errors,
        )
        section.update(compute_risk_state(p))
        return {"risk": section, "errors": errors}

    def portfolio_node(state: OrchestratorState) -> dict:
        errors = list(state.get("errors") or [])
        p = state["portfolio_data"]
        section = _run_section(
            "MRM-004",
            lambda: _live_section(
                run_portfolio_management(p, state["market"]["text"], state["risk"]["text"], **kw),
                "portfolio_recommendations"),
            lambda: demo_portfolio_recommendations(p), errors,
        )
        section["action_items"] = action_items(p)
        return {"portfolio": section, "errors": errors}

    graph = StateGraph(OrchestratorState)
    graph.add_node("market_analysis", market_node)
    graph.add_node("risk_assessment", risk_node)
    graph.add_node("portfolio_management", portfolio_node)
    graph.add_edge(START, "market_analysis")
    graph.add_edge("market_analysis", "risk_assessment")
    graph.add_edge("risk_assessment", "portfolio_management")
    graph.add_edge("portfolio_management", END)
    return graph.compile()


def run_full_analysis(
    portfolio_data: dict,
    *,
    llm=None,
    grader_llm=None,
    approvals: Optional[dict] = None,
    enforce_approval: bool = True,
) -> dict:
    """Run the pipeline. Raises GovernanceBlocked if MRM-005 itself is not approved."""
    started = time.perf_counter()
    cur = current_version(INVENTORY_ID)
    if enforce_approval:
        approval = require_approved(INVENTORY_ID, cur["model_id"], cur["prompt_versions"], approvals)
    else:
        approval = {"status": "EVALUATION_RUN", "reason": "approval gate bypassed by the eval harness"}

    graph = build_orchestrator(llm, grader_llm, approvals, enforce_approval)
    result = graph.invoke({"portfolio_data": portfolio_data, "errors": []})
    sections = {k: result[k] for k in SECTION_KEYS}
    trace = TraceRecord(
        inventory_id=INVENTORY_ID, provider=cur["provider"], model_id=cur["model_id"],
        prompt_versions=cur["prompt_versions"], mode="live",
        source_ids=sorted({sid for s in sections.values() for sid in s["trace"]["source_ids"]}),
        grader={"sections": {k: s["grader"].get("verdict") for k, s in sections.items()}},
        approval_status=approval["status"], approval_reason=approval["reason"],
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        extra={"section_modes": {k: s["mode"] for k, s in sections.items()},
               "section_traces": [s["trace"]["trace_id"] for s in sections.values()]},
    )
    return {
        **sections,
        "errors": result.get("errors") or [],
        "trace": log_trace(trace),
        # flat keys kept for API compatibility
        "market_analysis": sections["market"]["text"],
        "sector_breakdown": sections["market"]["sector_breakdown"],
        "technical_alerts": sections["market"]["technical_alerts"],
        "risk_report": sections["risk"]["text"],
        "portfolio_var": sections["risk"]["portfolio_var"],
        "position_risk_scores": sections["risk"]["position_risk_scores"],
        "critical_positions": sections["risk"]["critical_positions"],
        "portfolio_recommendations": sections["portfolio"]["text"],
        "action_items": sections["portfolio"]["action_items"],
    }


def demo_full_analysis(portfolio_data: dict, reason: str) -> dict:
    """All three sections from deterministic templates (used when MRM-005 is blocked/unavailable)."""
    blocked = reason.startswith("Blocked by model governance")
    sections = {
        "market": _fallback_section("MRM-002", demo_market_report(portfolio_data), reason, blocked),
        "risk": _fallback_section("MRM-003", demo_risk_report(portfolio_data), reason, blocked),
        "portfolio": _fallback_section("MRM-004", demo_portfolio_recommendations(portfolio_data), reason, blocked),
    }
    sections["market"]["sector_breakdown"] = portfolio_kpis(portfolio_data)["sector_weights"]
    sections["market"]["technical_alerts"] = technical_alerts(portfolio_data)
    sections["risk"].update(compute_risk_state(portfolio_data))
    sections["portfolio"]["action_items"] = action_items(portfolio_data)
    return {**sections, "errors": [reason]}
