"""
FinTech AI - Market Analysis Agent (inventory MRM-002).

Input: bundled positions + computed sector weights, technical signals, latest
earnings and macro records (src/agents/context.py). Output: an LLM market
report (graded + traced) plus deterministic sector breakdown and alerts.
"""

from __future__ import annotations

from typing import Optional

from src.agents.base import run_agent_component
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
    provider: Optional[str] = None,
    providers: Optional[list] = None,
    grader_provider: Optional[str] = None,
    grader_model: Optional[str] = None,
) -> dict:
    res = run_agent_component(
        INVENTORY_ID, "market_analysis", lambda prov: market_sources(portfolio_data, prov),
        llm=llm, grader_llm=grader_llm, approvals=approvals, enforce_approval=enforce_approval,
        provider=provider, providers=providers, grader_provider=grader_provider, grader_model=grader_model,
    )
    return {
        "market_analysis": res["text"],
        "sector_breakdown": portfolio_kpis(portfolio_data)["sector_weights"],
        "technical_alerts": technical_alerts(portfolio_data),
        "grader": res["grader"],
        "trace": res["trace"],
        "sources": res["sources"],
        "error": None,
        "provider": res.get("provider"),
        "answer_model": res.get("answer_model"),
        "fallback_reason": res.get("fallback_reason"),
    }
