"""
FinTech AI - Risk Assessment Agent
LangGraph agent that calculates VaR, Sharpe, Beta, drawdown risk.
Mirrors EduAI's at_risk_agent.py.
"""

import json
import boto3
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_aws import ChatBedrock
from langchain.schema import HumanMessage, SystemMessage
from src.utils.metrics import calculate_portfolio_var, score_position_risk, calculate_portfolio_cvar, run_stress_tests, full_risk_report

REGION = "us-east-1"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

RISK_PROMPT = """You are a Chief Risk Officer at a hedge fund.

Based on the quantitative risk metrics provided, generate a comprehensive risk report:
1. Portfolio-level risk summary - VaR, CVaR/Expected Shortfall, tail ratio, weighted beta, overall risk level
2. CVaR vs VaR comparison - explain what the tail ratio means for this portfolio
3. Stress test results - which scenarios cause the most damage, and which positions drive it
4. Top 5 highest-risk positions with specific concerns
5. Concentration risk analysis (over-weight sectors or single names)
6. Recommended risk mitigation actions (hedges, position sizing, stop-losses)
7. Macro risk overlay (how current rates/VIX amplify portfolio risk)

Key distinction: CVaR (Expected Shortfall) is the Basel III FRTB-preferred measure.
Always report both VaR and CVaR. A high tail ratio (CVaR/VaR > 1.5) indicates fat-tail risk
that VaR alone would understate. Be direct. Flag CRITICAL risks first. Quantify everything.

Portfolio Risk Data:
{risk_data}

Macro Context:
{macro_context}

Risk Assessment Report:"""


class RiskAssessmentState(TypedDict):
    portfolio_data: dict
    market_analysis: Optional[str]
    position_risk_scores: Optional[List[dict]]
    portfolio_var: Optional[dict]
    macro_context: Optional[str]
    risk_report: Optional[str]
    critical_positions: Optional[List[str]]
    error: Optional[str]


def score_all_positions(state: RiskAssessmentState) -> RiskAssessmentState:
    """Score every position's risk and calculate portfolio VaR, CVaR, and stress tests."""
    try:
        positions = state["portfolio_data"].get("positions", [])
        scored = [score_position_risk(p) for p in positions]
        scored.sort(key=lambda x: x["risk_score"], reverse=True)
        state["position_risk_scores"] = scored

        # Full risk report: VaR + CVaR + stress tests in one call
        risk_report = full_risk_report(positions, confidence=0.95)
        state["portfolio_var"] = {
            **risk_report["var"],
            "cvar_1d": risk_report["cvar"]["cvar_1d"],
            "cvar_1d_pct": risk_report["cvar"]["cvar_1d_pct"],
            "cvar_10d": risk_report["cvar"]["cvar_10d"],
            "tail_ratio": risk_report["cvar"]["tail_ratio"],
            "tail_ratio_interpretation": risk_report["cvar"]["tail_ratio_interpretation"],
            "stress_worst_case_scenario": risk_report["stress_tests"]["summary"]["worst_case_scenario"],
            "stress_worst_case_pnl": risk_report["stress_tests"]["summary"]["worst_case_pnl"],
            "stress_worst_case_pnl_pct": risk_report["stress_tests"]["summary"]["worst_case_pnl_pct"],
            "stress_catastrophic_count": risk_report["stress_tests"]["summary"]["catastrophic_scenarios"],
            "stress_scenarios": risk_report["stress_tests"]["scenarios"],
            "regulatory_note": risk_report["cvar"]["regulatory_note"],
        }

        state["critical_positions"] = [
            s["ticker"] for s in scored if s["risk_level"] in ["CRITICAL", "HIGH"]
        ]
    except Exception as e:
        state["error"] = f"Risk scoring failed: {e}"
    return state


def retrieve_macro_context(state: RiskAssessmentState) -> RiskAssessmentState:
    """Retrieve macro indicator context from FAISS."""
    try:
        from src.rag.vector_store import load_vector_store
        vectorstore = load_vector_store()
        docs = vectorstore.similarity_search(
            "Federal Reserve interest rates VIX inflation GDP unemployment portfolio risk", k=6
        )
        state["macro_context"] = "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        state["macro_context"] = "Macro context unavailable."
    return state


def generate_risk_report(state: RiskAssessmentState) -> RiskAssessmentState:
    """Generate a comprehensive risk report using Claude."""
    try:
        bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)
        llm = ChatBedrock(
            client=bedrock_client,
            model_id=MODEL_ID,
            model_kwargs={"max_tokens": 2048, "temperature": 0.05},
        )

        risk_data = json.dumps({
            "portfolio_var": {
                k: v for k, v in state["portfolio_var"].items()
                if k != "stress_scenarios"  # keep prompt concise
            },
            "stress_test_summary": {
                "worst_case": state["portfolio_var"].get("stress_worst_case_scenario"),
                "worst_case_pnl_pct": state["portfolio_var"].get("stress_worst_case_pnl_pct"),
                "catastrophic_scenario_count": state["portfolio_var"].get("stress_catastrophic_count"),
            },
            "top_risk_positions": state["position_risk_scores"][:8],
            "critical_tickers": state["critical_positions"],
            "market_analysis_summary": (state.get("market_analysis") or "")[:500],
        }, indent=2)

        prompt = RISK_PROMPT.format(
            risk_data=risk_data,
            macro_context=state.get("macro_context", "")
        )

        messages = [
            SystemMessage(content="You are a quantitative risk officer at a hedge fund."),
            HumanMessage(content=prompt),
        ]

        response = llm(messages)
        state["risk_report"] = response.content

    except Exception as e:
        state["error"] = f"Risk report generation failed: {e}"
        state["risk_report"] = "Risk report unavailable."

    return state


def build_risk_agent():
    graph = StateGraph(RiskAssessmentState)
    graph.add_node("score_positions", score_all_positions)
    graph.add_node("retrieve_macro", retrieve_macro_context)
    graph.add_node("generate_report", generate_risk_report)
    graph.set_entry_point("score_positions")
    graph.add_edge("score_positions", "retrieve_macro")
    graph.add_edge("retrieve_macro", "generate_report")
    graph.add_edge("generate_report", END)
    return graph.compile()


_agent = None


def run_risk_assessment(portfolio_data: dict, market_analysis: str = "") -> dict:
    global _agent
    if _agent is None:
        _agent = build_risk_agent()

    initial_state = RiskAssessmentState(
        portfolio_data=portfolio_data,
        market_analysis=market_analysis,
        position_risk_scores=None,
        portfolio_var=None,
        macro_context=None,
        risk_report=None,
        critical_positions=None,
        error=None,
    )
    result = _agent.invoke(initial_state)
    return {
        "risk_report": result.get("risk_report"),
        "portfolio_var": result.get("portfolio_var"),
        "position_risk_scores": result.get("position_risk_scores", []),
        "critical_positions": result.get("critical_positions", []),
        "error": result.get("error"),
    }
