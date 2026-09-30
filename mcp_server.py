#!/usr/bin/env python3
"""
MCP Server for FinTech AI - Hedge Fund Intelligence Platform

Exposes tools, resources, and prompts for AI assistant integration. Every
tool returns values computed from the bundled data/*.json files,
src/utils/metrics.py and src/mrm (see src/mcp_tools.py) - no LLM calls, nothing
is randomly generated, and the same input always returns the same output.

Run with:  python mcp_server.py        (deps: requirements-mcp.txt)
"""

import logging
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from src import mcp_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("FinTech AI Platform")


@mcp.tool()
def analyze_portfolio_risk(
    portfolio_symbols: Optional[List[str]] = None,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Portfolio risk for the bundled positions (optionally filtered by symbol):
    parametric VaR, seeded Monte Carlo CVaR/Expected Shortfall, five stress
    scenarios and position risk scores - all from src/utils/metrics.py.
    """
    return mcp_tools.analyze_portfolio_risk(portfolio_symbols, confidence_level)


@mcp.tool()
def generate_market_insights(sector: str, time_horizon: str = "1_week") -> Dict[str, Any]:
    """
    Sector analysis for a held sector: weight, P&L, price performance over the horizon
    (1_day, 1_week, 1_month, 3_month) and technical status, computed from data/market_data.json.
    Deterministic (no LLM).
    """
    return mcp_tools.generate_market_insights(sector, time_horizon)


@mcp.tool()
def optimize_portfolio_allocation(
    current_positions: Dict[str, float],
    target_risk_level: str = "moderate",
) -> Dict[str, Any]:
    """
    Long-only minimum-variance allocation (modern portfolio theory) with a
    per-name cap set by target_risk_level, using the covariance of the
    synthetic daily price history.
    """
    return mcp_tools.optimize_portfolio_allocation(current_positions, target_risk_level)


@mcp.tool()
def analyze_stock_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Position data (P/E, 52-week range, beta, rating, P&L), earnings history with revenue/EPS
    surprises and latest technicals for one ticker. Deterministic (no LLM).
    """
    return mcp_tools.analyze_stock_fundamentals(symbol)


@mcp.tool()
def get_market_regime_analysis() -> Dict[str, Any]:
    """Rule-based market regime classification from the latest macro indicators (VIX, 10Y-2Y slope, PMI, CPI trend)."""
    return mcp_tools.get_market_regime_analysis()


@mcp.tool()
def generate_compliance_report(
    portfolio_data: Optional[Dict[str, Any]] = None,
    report_type: str = "daily",
) -> Dict[str, Any]:
    """
    MRM report from real state: model inventory + approval evidence, VaR
    backtest (Kupiec / Basel zone), returns PSI drift, concentration checks and
    governance summary.
    """
    return mcp_tools.generate_compliance_report(portfolio_data, report_type)


@mcp.resource("system://rag_pipeline")
def get_rag_pipeline_info() -> Dict[str, Any]:
    """RAG architecture as configured in code (embedding model, chunking, HyDE, re-ranker, data files)."""
    return mcp_tools.rag_pipeline_info()


@mcp.resource("data://market_data")
def get_market_data_sources() -> Dict[str, Any]:
    """Bundled data sources and the audit retention actually configured in infra/."""
    return mcp_tools.market_data_sources()


@mcp.prompt("portfolio_analysis")
def portfolio_analysis_prompt() -> str:
    """Complete portfolio analysis workflow using the FinTech AI tools."""
    return """
    You are a senior portfolio manager reviewing the (synthetic) Horner Capital Alpha Fund.

    **Analysis Workflow:**
    1. Use analyze_portfolio_risk() for VaR, CVaR, stress tests and risk scores
    2. Use analyze_stock_fundamentals() for the largest and highest-risk holdings
    3. Use generate_market_insights() for the largest sectors
    4. Use optimize_portfolio_allocation() to see a minimum-variance alternative
    5. Use get_market_regime_analysis() for the macro regime
    6. Use generate_compliance_report() to check model governance status

    Only quote numbers returned by the tools. Present findings for human review;
    this is analysis, not investment advice.
    """


@mcp.prompt("market_research")
def market_research_prompt() -> str:
    """Market research workflow prompt for an AI assistant that calls these tools."""
    return """
    You are conducting market research on the (synthetic) FinTech AI dataset.

    1. Use get_market_regime_analysis() to understand the macro backdrop
    2. Use generate_market_insights() across the held sectors
    3. Use analyze_stock_fundamentals() for sector leaders
    4. Use generate_compliance_report() to confirm which models are approved

    Only quote numbers returned by the tools; flag anything the data does not cover.
    """


if __name__ == "__main__":
    mcp.run()
