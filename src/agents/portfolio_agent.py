"""
FinTech AI - Portfolio Management Agent (inventory MRM-004).

Suggestions for human review only. Action items are computed
deterministically from score_position_risk; the LLM writes the memo.
"""

from __future__ import annotations

from typing import Optional

from src.agents.base import run_llm_agent
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
    grade: bool = True,
    **_ignored,
) -> dict:
    sources = portfolio_sources(portfolio_data)
    res = run_llm_agent(
        INVENTORY_ID, "portfolio_recommendations", sources,
        llm=llm, grader_llm=grader_llm, approvals=approvals, enforce_approval=enforce_approval,
        grade=grade,
    )
    return {
        "portfolio_recommendations": res["text"],
        "action_items": action_items(portfolio_data),
        "grader": res["grader"],
        "trace": res["trace"],
        "sources": sources,
        "error": None,
    }
