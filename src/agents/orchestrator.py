"""
FinTech AI - Multi-Agent Orchestrator
LangGraph StateGraph that chains:
  Market Analysis Agent -> Risk Assessment Agent -> Portfolio Management Agent

Mirrors EduAI's orchestrator.py pattern.
"""

import json
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from src.agents.market_agent import run_market_analysis
from src.agents.risk_agent import run_risk_assessment
from src.agents.portfolio_agent import run_portfolio_management


class OrchestratorState(TypedDict):
    # Input
    portfolio_data: dict

    # Market Agent outputs
    market_analysis: Optional[str]
    sector_breakdown: Optional[dict]
    technical_alerts: Optional[List[str]]

    # Risk Agent outputs
    risk_report: Optional[str]
    portfolio_var: Optional[dict]
    position_risk_scores: Optional[List[dict]]
    critical_positions: Optional[List[str]]

    # Portfolio Agent outputs
    portfolio_recommendations: Optional[str]
    action_items: Optional[List[dict]]

    # Meta
    errors: Optional[List[str]]


def run_market_node(state: OrchestratorState) -> OrchestratorState:
    print("[SEARCH] Running Market Analysis Agent...")
    result = run_market_analysis(state["portfolio_data"])
    state["market_analysis"] = result.get("market_analysis")
    state["sector_breakdown"] = result.get("sector_breakdown")
    state["technical_alerts"] = result.get("technical_alerts", [])
    if result.get("error"):
        state["errors"] = (state.get("errors") or []) + [f"MarketAgent: {result['error']}"]
    return state


def run_risk_node(state: OrchestratorState) -> OrchestratorState:
    print("[WARN]  Running Risk Assessment Agent...")
    result = run_risk_assessment(
        portfolio_data=state["portfolio_data"],
        market_analysis=state.get("market_analysis") or "",
    )
    state["risk_report"] = result.get("risk_report")
    state["portfolio_var"] = result.get("portfolio_var")
    state["position_risk_scores"] = result.get("position_risk_scores", [])
    state["critical_positions"] = result.get("critical_positions", [])
    if result.get("error"):
        state["errors"] = (state.get("errors") or []) + [f"RiskAgent: {result['error']}"]
    return state


def run_portfolio_node(state: OrchestratorState) -> OrchestratorState:
    print("[CHART] Running Portfolio Management Agent...")
    result = run_portfolio_management(
        portfolio_data=state["portfolio_data"],
        market_analysis=state.get("market_analysis") or "",
        risk_report=state.get("risk_report") or "",
        portfolio_var=state.get("portfolio_var") or {},
        critical_positions=state.get("critical_positions") or [],
        position_risk_scores=state.get("position_risk_scores") or [],
    )
    state["portfolio_recommendations"] = result.get("portfolio_recommendations")
    state["action_items"] = result.get("action_items", [])
    if result.get("error"):
        state["errors"] = (state.get("errors") or []) + [f"PortfolioAgent: {result['error']}"]
    return state


def build_orchestrator():
    """Build the full multi-agent LangGraph orchestrator."""
    graph = StateGraph(OrchestratorState)

    graph.add_node("market_analysis", run_market_node)
    graph.add_node("risk_assessment", run_risk_node)
    graph.add_node("portfolio_management", run_portfolio_node)

    graph.set_entry_point("market_analysis")
    graph.add_edge("market_analysis", "risk_assessment")
    graph.add_edge("risk_assessment", "portfolio_management")
    graph.add_edge("portfolio_management", END)

    return graph.compile()


_orchestrator = None


def run_full_analysis(portfolio_data: dict) -> dict:
    """
    Run the complete multi-agent analysis pipeline.
    Returns a unified result dict for the API/dashboard.
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = build_orchestrator()

    initial_state = OrchestratorState(
        portfolio_data=portfolio_data,
        market_analysis=None,
        sector_breakdown=None,
        technical_alerts=None,
        risk_report=None,
        portfolio_var=None,
        position_risk_scores=None,
        critical_positions=None,
        portfolio_recommendations=None,
        action_items=None,
        errors=None,
    )

    print("\n[START] Starting FinTech AI Multi-Agent Pipeline...")
    result = _orchestrator.invoke(initial_state)
    print("[PASS] Pipeline complete.\n")

    return {
        "market_analysis": result.get("market_analysis"),
        "sector_breakdown": result.get("sector_breakdown"),
        "technical_alerts": result.get("technical_alerts", []),
        "risk_report": result.get("risk_report"),
        "portfolio_var": result.get("portfolio_var"),
        "position_risk_scores": result.get("position_risk_scores", []),
        "critical_positions": result.get("critical_positions", []),
        "portfolio_recommendations": result.get("portfolio_recommendations"),
        "action_items": result.get("action_items", []),
        "errors": result.get("errors"),
    }
