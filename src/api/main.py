"""
FinTech AI - FastAPI Backend

Same provider / approval gate / grader / trace code as the Streamlit app.
Importable without AWS or an API key (providers are imported lazily). Every
AI endpoint returns ``mode`` ("live" | "demo"), ``reason``, ``grader`` and
``trace``; without a key (or when governance blocks) it returns the
deterministic, data-derived demo output instead of failing.

Run:  uvicorn src.api.main:app --reload     (deps: requirements-api.txt)
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.guardrails.bedrock_guardrails import apply_guardrails, check_compliance
from src.llm.guard import run_guarded
from src.mrm.trace import attach_demo_trace, recent_traces
from src.utils.data import load_portfolio

app = FastAPI(
    title="FinTech AI - Hedge Fund Intelligence Platform",
    description="AI-powered market analysis, risk assessment, and portfolio management (synthetic data).",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatTurn]] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list
    mode: str
    reason: Optional[str] = None
    grader: Optional[dict] = None
    trace: Optional[dict] = None


class AnalysisResponse(BaseModel):
    success: bool
    mode: str
    reason: Optional[str] = None
    data: dict
    error: Optional[str] = None


def _guarded(inventory_id: str, live_fn, demo_fn) -> dict:
    res = run_guarded(live_fn, demo_fn)
    if res["mode"] != "live":
        attach_demo_trace(inventory_id, res["output"], res["reason"], res["blocked"])
    return res


@app.get("/health")
def health_check():
    from src.llm import config

    return {"status": "healthy", "service": "FinTech AI", "version": "2.0.0",
            "provider": config.get_provider(), "live_ai_configured": config.has_credentials()}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """RAG Q&A (MRM-001): HyDE -> FAISS -> cross-encoder re-rank -> answer -> grader."""
    compliance = check_compliance(request.question)
    if not compliance["compliant"]:
        raise HTTPException(status_code=400, detail=compliance["reason"])

    def live():
        from src.rag.rag_chain import answer_question

        history = [t.model_dump() for t in (request.history or [])]
        return answer_question(request.question, history)

    def demo(reason):
        from src.demo.templates import demo_chat_answer

        return demo_chat_answer(request.question)

    res = _guarded("MRM-001", live, demo)
    out = res["output"]
    answer = out.get("answer") or out.get("text", "")
    return ChatResponse(
        question=request.question, answer=apply_guardrails(answer), sources=out.get("sources", []),
        mode=res["mode"], reason=res["reason"], grader=out.get("grader"), trace=out.get("trace"),
    )


@app.post("/insights/market", response_model=AnalysisResponse)
def market_insights():
    """Market Analysis Agent (MRM-002)."""
    portfolio = load_portfolio()

    def live():
        from src.agents.market_agent import run_market_analysis

        return run_market_analysis(portfolio)

    def demo(reason):
        from src.agents.context import technical_alerts
        from src.demo.templates import demo_market_report
        from src.utils.metrics import portfolio_kpis

        d = demo_market_report(portfolio)
        return {**d, "market_analysis": d["text"], "sector_breakdown": portfolio_kpis(portfolio)["sector_weights"],
                "technical_alerts": technical_alerts(portfolio)}

    res = _guarded("MRM-002", live, demo)
    data = res["output"]
    data["market_analysis"] = apply_guardrails(data.get("market_analysis", ""), add_disclaimer=False)
    return AnalysisResponse(success=True, mode=res["mode"], reason=res["reason"], data=data)


@app.post("/insights/risk", response_model=AnalysisResponse)
def risk_insights():
    """Risk Assessment Agent (MRM-003) over metrics computed by MRM-006."""
    portfolio = load_portfolio()

    def live():
        from src.agents.risk_agent import run_risk_assessment

        return run_risk_assessment(portfolio)

    def demo(reason):
        from src.agents.risk_agent import compute_risk_state
        from src.demo.templates import demo_risk_report

        d = demo_risk_report(portfolio)
        return {**d, "risk_report": d["text"], **compute_risk_state(portfolio)}

    res = _guarded("MRM-003", live, demo)
    data = res["output"]
    data["risk_report"] = apply_guardrails(data.get("risk_report", ""))
    return AnalysisResponse(success=True, mode=res["mode"], reason=res["reason"], data=data)


@app.post("/insights/portfolio", response_model=AnalysisResponse)
def portfolio_insights():
    """LangGraph orchestrator (MRM-005): market -> risk -> portfolio."""
    portfolio = load_portfolio()

    def live():
        from src.agents.orchestrator import run_full_analysis

        return run_full_analysis(portfolio)

    def demo(reason):
        from src.agents.orchestrator import demo_full_analysis

        return demo_full_analysis(portfolio, reason)

    res = _guarded("MRM-005", live, demo)
    data = res["output"]
    if data.get("portfolio", {}).get("text"):
        data["portfolio"]["text"] = apply_guardrails(data["portfolio"]["text"])
    return AnalysisResponse(success=True, mode=res["mode"], reason=res["reason"], data=data)


@app.get("/portfolio/summary")
def portfolio_summary():
    """Bundled (synthetic) portfolio data plus computed KPIs."""
    from src.utils.metrics import portfolio_kpis

    pf = load_portfolio()
    return {**pf, "kpis": portfolio_kpis(pf)}


@app.get("/traces/recent")
def traces_recent(limit: int = 50):
    """TraceRecords produced by this API process (most recent last)."""
    return recent_traces(max(1, min(limit, 500)))


#  Mount MRM Router
from src.api.mrm_routes import router as mrm_router  # noqa: E402

app.include_router(mrm_router)


def list_endpoints(application: FastAPI = app) -> list:
    """(METHOD, path) for every API endpoint, including included routers (docs routes excluded)."""
    out = []

    def walk(routes, prefix=""):
        for r in routes:
            original = getattr(r, "original_router", None)
            if original is not None:  # FastAPI >= 0.13x wraps include_router() results
                ctx = getattr(r, "include_context", None)
                walk(original.routes, prefix + (getattr(ctx, "prefix", "") or ""))
                continue
            path = getattr(r, "path", None)
            methods = getattr(r, "methods", None)
            if not path or not methods or path.startswith(("/docs", "/redoc", "/openapi")):
                continue
            for m in sorted(methods - {"HEAD", "OPTIONS"}):
                out.append((m, prefix + path))

    walk(application.routes)
    return sorted(set(out), key=lambda x: (x[1], x[0]))
