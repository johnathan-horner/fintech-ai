"""
FinTech AI - Market Analysis Agent
LangGraph agent that analyzes price trends, technical indicators, and sector momentum.
Mirrors EduAI's grading_agent.py pattern.
"""

import json
import boto3
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_aws import ChatBedrock
from langchain.schema import HumanMessage, SystemMessage
from src.rag.vector_store import load_vector_store

REGION = "us-east-1"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

MARKET_ANALYSIS_PROMPT = """You are a senior market analyst at a top-tier hedge fund.

Analyze the following market data and identify:
1. Key price trends (bullish/bearish/neutral) for each position
2. Technical signals (RSI overbought/oversold, SMA crossovers, MACD)
3. Sector-level momentum (which sectors are outperforming/underperforming)
4. Earnings beat/miss patterns from recent transcripts
5. Macro factors (rates, VIX, PMI) impacting the portfolio

Be quantitative. Call out specific numbers. Flag any position showing concerning technical deterioration.

Market Context:
{context}

Provide a structured Market Analysis Report."""


class MarketAnalysisState(TypedDict):
    portfolio_data: dict
    market_context: str
    market_analysis: Optional[str]
    sector_breakdown: Optional[dict]
    technical_alerts: Optional[List[str]]
    error: Optional[str]


def retrieve_market_context(state: MarketAnalysisState) -> MarketAnalysisState:
    """Retrieve relevant market data from FAISS."""
    try:
        vectorstore = load_vector_store()
        tickers = [p["ticker"] for p in state["portfolio_data"].get("positions", [])]
        query = f"market trends technical indicators sector momentum {' '.join(tickers[:5])}"
        docs = vectorstore.similarity_search(query, k=12)
        state["market_context"] = "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        state["error"] = f"Context retrieval failed: {e}"
        state["market_context"] = "No market context available."
    return state


def analyze_market(state: MarketAnalysisState) -> MarketAnalysisState:
    """Run Claude market analysis on retrieved context."""
    try:
        bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)
        llm = ChatBedrock(
            client=bedrock_client,
            model_id=MODEL_ID,
            model_kwargs={"max_tokens": 2048, "temperature": 0.1},
        )

        prompt = MARKET_ANALYSIS_PROMPT.format(context=state["market_context"])
        portfolio_summary = json.dumps({
            "positions": state["portfolio_data"].get("positions", [])[:10]
        }, indent=2)

        messages = [
            SystemMessage(content="You are a quantitative hedge fund market analyst."),
            HumanMessage(content=f"Portfolio:\n{portfolio_summary}\n\n{prompt}"),
        ]

        response = llm(messages)
        state["market_analysis"] = response.content

        # Build sector breakdown
        positions = state["portfolio_data"].get("positions", [])
        sector_map = {}
        for p in positions:
            sector = p.get("sector", "Unknown")
            if sector not in sector_map:
                sector_map[sector] = {"tickers": [], "total_value": 0, "avg_pnl_pct": []}
            sector_map[sector]["tickers"].append(p["ticker"])
            sector_map[sector]["total_value"] += p.get("market_value", 0)
            sector_map[sector]["avg_pnl_pct"].append(p.get("pnl_pct", 0))

        for sector in sector_map:
            pnls = sector_map[sector]["avg_pnl_pct"]
            sector_map[sector]["avg_pnl_pct"] = round(sum(pnls) / len(pnls), 2) if pnls else 0

        state["sector_breakdown"] = sector_map

        # Technical alerts
        alerts = []
        for p in positions:
            if p.get("pnl_pct", 0) < -20:
                alerts.append(f"[WARN] {p['ticker']}: Down {p['pnl_pct']}% - near 52W low territory")
            if p.get("beta", 1.0) > 1.8:
                alerts.append(f"[HIGH] {p['ticker']}: Beta {p['beta']}x - high market sensitivity")
            if p.get("analyst_rating") in ["Sell", "Strong Sell"]:
                alerts.append(f"[DOWN] {p['ticker']}: Analyst rating = {p['analyst_rating']}")
        state["technical_alerts"] = alerts

    except Exception as e:
        state["error"] = f"Market analysis failed: {e}"
        state["market_analysis"] = "Analysis unavailable."

    return state


def build_market_agent():
    """Build the LangGraph Market Analysis Agent."""
    graph = StateGraph(MarketAnalysisState)
    graph.add_node("retrieve_context", retrieve_market_context)
    graph.add_node("analyze_market", analyze_market)
    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "analyze_market")
    graph.add_edge("analyze_market", END)
    return graph.compile()


# Singleton
_agent = None


def run_market_analysis(portfolio_data: dict) -> dict:
    global _agent
    if _agent is None:
        _agent = build_market_agent()

    initial_state = MarketAnalysisState(
        portfolio_data=portfolio_data,
        market_context="",
        market_analysis=None,
        sector_breakdown=None,
        technical_alerts=None,
        error=None,
    )
    result = _agent.invoke(initial_state)
    return {
        "market_analysis": result.get("market_analysis"),
        "sector_breakdown": result.get("sector_breakdown"),
        "technical_alerts": result.get("technical_alerts", []),
        "error": result.get("error"),
    }
