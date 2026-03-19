"""
FinTech AI - Portfolio Management Agent
LangGraph agent that recommends rebalancing, hedges, entries, and exits.
Third agent in the chain - runs after Market + Risk agents.
"""

import json
import boto3
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_aws import ChatBedrock
from langchain.schema import HumanMessage, SystemMessage

REGION = "us-east-1"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

PORTFOLIO_PROMPT = """You are a Portfolio Manager at a top-tier hedge fund.

Given the market analysis and risk assessment below, generate specific, actionable portfolio recommendations:

1. IMMEDIATE ACTIONS (within 24-48 hours)
   - Positions to exit or reduce (with target size)
   - Stop-loss levels for high-risk positions
   - Hedges to implement (options, inverse ETFs, sector shorts)

2. REBALANCING PLAN (1-2 week horizon)
   - Concentration adjustments (over-weight sectors to trim)
   - New positions to consider (with entry thesis)
   - Target portfolio weights by sector

3. MACRO POSITIONING (1-3 month view)
   - Rate sensitivity adjustments
   - Defensive vs. growth positioning
   - Cash allocation recommendation

4. PERFORMANCE OPTIMIZATION
   - Tax-loss harvesting opportunities (positions with large unrealized losses)
   - Dividend and income optimization
   - Fee/cost minimization

Be specific. Give target allocations in percentages. Name specific tickers for each recommendation.

Market Analysis:
{market_analysis}

Risk Assessment:
{risk_assessment}

Portfolio Summary:
{portfolio_summary}

Portfolio Recommendations:"""


class PortfolioManagementState(TypedDict):
    portfolio_data: dict
    market_analysis: str
    risk_report: str
    portfolio_var: dict
    critical_positions: List[str]
    position_risk_scores: List[dict]
    portfolio_recommendations: Optional[str]
    action_items: Optional[List[dict]]
    error: Optional[str]


def generate_recommendations(state: PortfolioManagementState) -> PortfolioManagementState:
    """Generate portfolio recommendations using all upstream agent data."""
    try:
        bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)
        llm = ChatBedrock(
            client=bedrock_client,
            model_id=MODEL_ID,
            model_kwargs={"max_tokens": 2500, "temperature": 0.1},
        )

        portfolio = state["portfolio_data"]
        portfolio_summary = json.dumps({
            "fund_name": portfolio.get("fund_name"),
            "aum": portfolio.get("aum"),
            "positions": portfolio.get("positions", [])[:15],
        }, indent=2)

        prompt = PORTFOLIO_PROMPT.format(
            market_analysis=state.get("market_analysis", "")[:1500],
            risk_assessment=state.get("risk_report", "")[:1500],
            portfolio_summary=portfolio_summary,
        )

        messages = [
            SystemMessage(content="You are a senior portfolio manager at a hedge fund."),
            HumanMessage(content=prompt),
        ]

        response = llm(messages)
        state["portfolio_recommendations"] = response.content

        # Build structured action items for the dashboard
        action_items = []
        for pos in state.get("position_risk_scores", [])[:5]:
            if pos["risk_level"] in ["CRITICAL", "HIGH"]:
                action_items.append({
                    "ticker": pos["ticker"],
                    "action": pos["recommended_action"],
                    "risk_level": pos["risk_level"],
                    "risk_score": pos["risk_score"],
                    "flags": pos["flags"],
                    "priority": "URGENT" if pos["risk_level"] == "CRITICAL" else "HIGH",
                })

        state["action_items"] = action_items

    except Exception as e:
        state["error"] = f"Portfolio recommendations failed: {e}"
        state["portfolio_recommendations"] = "Recommendations unavailable."

    return state


def build_portfolio_agent():
    graph = StateGraph(PortfolioManagementState)
    graph.add_node("generate_recommendations", generate_recommendations)
    graph.set_entry_point("generate_recommendations")
    graph.add_edge("generate_recommendations", END)
    return graph.compile()


_agent = None


def run_portfolio_management(
    portfolio_data: dict,
    market_analysis: str = "",
    risk_report: str = "",
    portfolio_var: dict = None,
    critical_positions: List[str] = None,
    position_risk_scores: List[dict] = None,
) -> dict:
    global _agent
    if _agent is None:
        _agent = build_portfolio_agent()

    initial_state = PortfolioManagementState(
        portfolio_data=portfolio_data,
        market_analysis=market_analysis,
        risk_report=risk_report,
        portfolio_var=portfolio_var or {},
        critical_positions=critical_positions or [],
        position_risk_scores=position_risk_scores or [],
        portfolio_recommendations=None,
        action_items=None,
        error=None,
    )
    result = _agent.invoke(initial_state)
    return {
        "portfolio_recommendations": result.get("portfolio_recommendations"),
        "action_items": result.get("action_items", []),
        "error": result.get("error"),
    }
