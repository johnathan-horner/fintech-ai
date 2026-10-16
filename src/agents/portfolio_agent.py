"""
FinTech AI - Portfolio Management Agent (inventory MRM-004).

Suggestions for human review only. Action items are computed
deterministically from score_position_risk; the LLM writes the memo.
"""

from __future__ import annotations

from typing import Optional

from src.agents.base import run_agent_component
from src.agents.context import action_items, portfolio_sources

INVENTORY_ID = "MRM-004"


def run_portfolio_management(
    portfolio_data: dict,
    market_analysis: str = "",
    risk_report: str = "",
    *,
    llm=None,
    grader_llm=None,
    approvals: Optional[dict] = None,
    enforce_approval: bool = True,
    provider: Optional[str] = None,
    providers: Optional[list] = None,
    grader_provider: Optional[str] = None,
    grader_model: Optional[str] = None,
    grade: bool = True,
    **_ignored,
) -> dict:
    res = run_agent_component(
        INVENTORY_ID, "portfolio_recommendations", lambda prov: portfolio_sources(portfolio_data, prov),
        llm=llm, grader_llm=grader_llm, approvals=approvals, enforce_approval=enforce_approval,
        provider=provider, providers=providers, grader_provider=grader_provider, grader_model=grader_model,
        grade=grade,
    )
    return {
        "portfolio_recommendations": res["text"],
        "action_items": action_items(portfolio_data),
        "grader": res["grader"],
        "trace": res["trace"],
        "sources": res["sources"],
        "error": None,
        "provider": res.get("provider"),
        "answer_model": res.get("answer_model"),
        "fallback_reason": res.get("fallback_reason"),
    }
