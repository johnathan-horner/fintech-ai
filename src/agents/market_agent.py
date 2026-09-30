"""
FinTech AI - Market Analysis Agent (inventory MRM-002).

Input: bundled positions + computed sector weights, technical signals, latest
earnings and macro records (src/agents/context.py). Output: an LLM market
report (graded + traced) plus deterministic sector breakdown and alerts.
"""

from __future__ import annotations

from typing import Optional

from src.agents.base import run_llm_agent
from src.agents.context import market_sources, technical_alerts
from src.utils.metrics import portfolio_kpis

INVENTORY_ID = "MRM-002"


def run_market_analysis(
    portfolio_data: dict,
    *,
    llm=None,
    grader_llm=None,
    approvals: Optional[dict] = None,
    enforce_approval: bool = True,
) -> dict:
    sources = market_sources(portfolio_data)
    res = run_llm_agent(
        INVENTORY_ID, "market_analysis", sources,
        llm=llm, grader_llm=grader_llm, approvals=approvals, enforce_approval=enforce_approval,
    )
    return {
        "market_analysis": res["text"],
        "sector_breakdown": portfolio_kpis(portfolio_data)["sector_weights"],
        "technical_alerts": technical_alerts(portfolio_data),
        "grader": res["grader"],
        "trace": res["trace"],
        "sources": sources,
        "error": None,
    }
