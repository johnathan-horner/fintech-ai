"""
FinTech AI - FastAPI Backend
Endpoints for chat, market analysis, risk assessment, and portfolio management.
Mirrors EduAI's main.py API pattern.
"""

import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.rag.rag_chain import query as rag_query
from src.agents.orchestrator import run_full_analysis
from src.agents.market_agent import run_market_analysis
from src.agents.risk_agent import run_risk_assessment
from src.agents.portfolio_agent import run_portfolio_management
from src.guardrails.bedrock_guardrails import check_compliance, apply_guardrails

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")

app = FastAPI(
    title="FinTech AI - Hedge Fund Intelligence Platform",
    description="AI-powered market analysis, risk assessment, and portfolio management.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#  Request / Response Models 

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list


class AnalysisResponse(BaseModel):
    success: bool
    data: dict
    error: Optional[str] = None


#  Helpers 

def load_portfolio():
    path = os.path.join(DATA_DIR, "portfolio.json")
    with open(path) as f:
        return json.load(f)


#  Routes 

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "FinTech AI", "version": "1.0.0"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Natural language Q&A about markets, positions, earnings, and macro themes.
    Uses RAG pipeline with conversational memory.
    """
    # Compliance check
    compliance = check_compliance(request.question)
    if not compliance["compliant"]:
        raise HTTPException(status_code=400, detail=compliance["reason"])

    try:
        result = rag_query(request.question)
        answer = apply_guardrails(result["answer"])
        return ChatResponse(
            question=request.question,
            answer=answer,
            sources=result.get("sources", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insights/market", response_model=AnalysisResponse)
def market_insights():
    """
    Run the Market Analysis Agent on the current portfolio.
    Returns price trends, technical alerts, and sector breakdown.
    """
    try:
        portfolio = load_portfolio()
        result = run_market_analysis(portfolio)
        result["answer"] = apply_guardrails(result.get("market_analysis", ""), add_disclaimer=False)
        return AnalysisResponse(success=True, data=result)
    except Exception as e:
        return AnalysisResponse(success=False, data={}, error=str(e))


@app.post("/insights/risk", response_model=AnalysisResponse)
def risk_insights():
    """
    Run the Risk Assessment Agent.
    Returns VaR, position risk scores, and risk report.
    """
    try:
        portfolio = load_portfolio()
        result = run_risk_assessment(portfolio)
        result["risk_report"] = apply_guardrails(result.get("risk_report", ""))
        return AnalysisResponse(success=True, data=result)
    except Exception as e:
        return AnalysisResponse(success=False, data={}, error=str(e))


@app.post("/insights/portfolio", response_model=AnalysisResponse)
def portfolio_insights():
    """
    Run all three agents sequentially via the orchestrator.
    Returns full market analysis, risk assessment, and portfolio recommendations.
    """
    try:
        portfolio = load_portfolio()
        result = run_full_analysis(portfolio)
        if result.get("portfolio_recommendations"):
            result["portfolio_recommendations"] = apply_guardrails(
                result["portfolio_recommendations"]
            )
        return AnalysisResponse(success=True, data=result)
    except Exception as e:
        return AnalysisResponse(success=False, data={}, error=str(e))


@app.get("/portfolio/summary")
def portfolio_summary():
    """Return current portfolio data for dashboard display."""
    try:
        return load_portfolio()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  Mount MRM Router 
from src.api.mrm_routes import router as mrm_router
app.include_router(mrm_router)
