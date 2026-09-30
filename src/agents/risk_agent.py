"""
FinTech AI - Risk Assessment Agent (inventory MRM-003).

All numbers in the prompt come from src/utils/metrics.py (parametric VaR,
Monte Carlo CVaR, stress tests, concentration, position risk scores).
"""

from __future__ import annotations

from typing import Optional

from src.agents.base import run_llm_agent
from src.agents.context import risk_sources
from src.utils.metrics import full_risk_report, score_position_risk

INVENTORY_ID = "MRM-003"


def compute_risk_state(portfolio_data: dict) -> dict:
    positions = portfolio_data.get("positions", [])
    scored = sorted((score_position_risk(p) for p in positions), key=lambda x: x["risk_score"], reverse=True)
    report = full_risk_report(positions, confidence=0.95)
    portfolio_var = {
        **report["var"],
        "cvar_1d": report["cvar"]["cvar_1d"],
        "cvar_1d_pct": report["cvar"]["cvar_1d_pct"],
        "cvar_10d": report["cvar"]["cvar_10d"],
        "tail_ratio": report["cvar"]["tail_ratio"],
        "tail_ratio_interpretation": report["cvar"]["tail_ratio_interpretation"],
        "stress_worst_case_scenario": report["stress_tests"]["summary"]["worst_case_scenario"],
        "stress_worst_case_pnl": report["stress_tests"]["summary"]["worst_case_pnl"],
        "stress_worst_case_pnl_pct": report["stress_tests"]["summary"]["worst_case_pnl_pct"],
        "stress_catastrophic_count": report["stress_tests"]["summary"]["catastrophic_scenarios"],
        "stress_scenarios": report["stress_tests"]["scenarios"],
        "regulatory_note": report["cvar"]["regulatory_note"],
    }
    return {
        "position_risk_scores": scored,
        "portfolio_var": portfolio_var,
        "critical_positions": [s["ticker"] for s in scored if s["risk_level"] in ("CRITICAL", "HIGH")],
    }


def run_risk_assessment(
    portfolio_data: dict,
    market_analysis: str = "",
    *,
    llm=None,
    grader_llm=None,
    approvals: Optional[dict] = None,
    enforce_approval: bool = True,
) -> dict:
    state = compute_risk_state(portfolio_data)
    sources = risk_sources(portfolio_data)
    res = run_llm_agent(
        INVENTORY_ID, "risk_report", sources,
        llm=llm, grader_llm=grader_llm, approvals=approvals, enforce_approval=enforce_approval,
    )
    return {"risk_report": res["text"], **state, "grader": res["grader"], "trace": res["trace"],
            "sources": sources, "error": None}
