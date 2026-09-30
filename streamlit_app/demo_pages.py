"""
FinTech AI - Dashboard pages.

Every KPI is computed from the bundled (synthetic) data by src/utils/metrics.py.
AI outputs come from the live Gemini pipeline when available (with a
"Live AI" badge, grader score and trace) and otherwise from deterministic
templates filled with computed values (with a "Demo mode" badge + reason).
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import metrics  # noqa: E402
from src.utils.data import load_portfolio as _load_portfolio  # noqa: E402
from streamlit_app.formatting import md_escape, md_money  # noqa: E402

# -- Global CSS ----------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background-color: #080d1a;
    color: #cbd5e1;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a1020 100%);
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    padding: 6px 0;
    font-size: 0.88rem;
    letter-spacing: 0.01em;
}

/* Streamlit overrides */
.stMetric { background: #0d1829; border: 1px solid #1a2740; border-radius: 8px; padding: 16px !important; }
.stMetric label { color: #475569 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
.stMetric [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-family: 'JetBrains Mono', monospace; }
div[data-testid="stExpander"] { background: #0d1829; border: 1px solid #1a2740; border-radius: 8px; margin-bottom: 8px; }
div[data-testid="stExpander"] summary { color: #cbd5e1; }
.stButton > button { background: #1e3a5f; color: #93c5fd; border: 1px solid #2d4a7a; border-radius: 6px; font-size: 0.82rem; font-weight: 500; padding: 6px 14px; transition: all 0.15s; }
.stButton > button:hover { background: #2d4a7a; color: #f1f5f9; border-color: #3b82f6; }
.stTabs [data-baseweb="tab-list"] { background: #0d1829; border-bottom: 1px solid #1a2740; }
.stTabs [data-baseweb="tab"] { color: #64748b; font-size: 0.82rem; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #3b82f6 !important; border-bottom: 2px solid #3b82f6; }
.stChatMessage { background: #0d1829; border: 1px solid #1a2740; border-radius: 10px; }
div[data-testid="stDataFrame"] { background: #0d1829; border: 1px solid #1a2740; border-radius: 8px; }
div[data-testid="stAlert"] { border-radius: 8px; }
.stProgress > div > div > div { background: #1a2740; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# -- Data (all computed) ------------------------------------------------------------

@st.cache_data
def load_portfolio():
    return _load_portfolio()


@st.cache_data
def load_kpis():
    return metrics.portfolio_kpis(load_portfolio())


@st.cache_data
def load_risk_metrics():
    pos = load_portfolio()["positions"]
    return {
        "var": metrics.calculate_portfolio_var(pos, 0.95),
        "cvar": metrics.calculate_portfolio_cvar(pos, 0.95),
        "stress": metrics.run_stress_tests(pos),
        "scored": sorted((metrics.score_position_risk(p) for p in pos), key=lambda x: x["risk_score"], reverse=True),
        "concentration": metrics.concentration_check(load_portfolio()),
    }


@st.cache_data
def load_signals():
    from src.utils.data import load_market_data

    held = sorted(p["ticker"] for p in load_portfolio()["positions"])
    return metrics.technical_signals(load_market_data(), held)


@st.cache_data
def load_history_metrics():
    from src.utils.data import load_macro, load_price_history

    rf = sorted(load_macro(), key=lambda m: m["date"])[-1]["Federal Funds Rate"] / 100
    return metrics.history_metrics(load_price_history(), load_portfolio()["positions"], rf)


@st.cache_data
def load_backtest():
    from src.mrm.validation.model_validator import portfolio_backtest

    return portfolio_backtest()


def _m(x: float) -> str:
    """Compact currency for st.metric values (plain text, not markdown)."""
    a = abs(x)
    s = f"${a / 1e6:.2f}M" if a >= 1e6 else f"${a / 1e3:.0f}K" if a >= 1e3 else f"${a:,.0f}"
    return f"-{s}" if x < 0 else s


# -- Multi-agent run shared by Market / Risk / Portfolio Management -------------

def _orchestrator_result():
    return st.session_state.get("orchestrator_result")


def _run_orchestrator():
    from src.agents.orchestrator import demo_full_analysis, run_full_analysis
    from streamlit_app.live_ai import run_ai_request

    portfolio = load_portfolio()
    with st.spinner("Running LangGraph pipeline: market -> risk -> portfolio (3 model calls + 3 grader calls)..."):
        st.session_state["orchestrator_result"] = run_ai_request(
            "MRM-005",
            lambda: run_full_analysis(portfolio),
            lambda reason: demo_full_analysis(portfolio, reason),
        )


def _orchestrator_controls(key: str):
    from streamlit_app.live_ai import MAX_LIVE_REQUESTS_PER_SESSION, requests_used

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Run multi-agent analysis", key=key):
            _run_orchestrator()
    with c2:
        st.caption(
            f"One run = 1 live AI request ({requests_used()}/{MAX_LIVE_REQUESTS_PER_SESSION} used this session). "
            "Result is shared by Market Analysis, Risk Assessment and Portfolio Management."
        )


def _render_section(section_key: str, placeholder_demo):
    from streamlit_app.live_ai import render_output

    res = _orchestrator_result()
    if res is None:
        demo = placeholder_demo()
        from src.mrm.grader import deterministic_grade

        render_output(demo["text"], "demo", "multi-agent analysis not run yet in this session",
                      deterministic_grade(demo["text"], demo["sources"]), None)
        return
    out = res["output"]
    section = out.get(section_key, {})
    if res["mode"] == "live":
        mode, reason = section.get("mode", "live"), section.get("reason")
    else:
        mode, reason = ("blocked" if res.get("blocked") else "demo"), res.get("reason")
    render_output(section.get("text", ""), mode, reason, section.get("grader"), section.get("trace"))
    if res["mode"] == "live" and out.get("trace"):
        with st.expander(f"Orchestrator trace {out['trace']['trace_id']} · MRM-005"):
            st.json(out["trace"])


# -- Pages ---------------------------------------------------------------------

def page_portfolio():
    inject_css()
    portfolio = load_portfolio()
    k = load_kpis()
    conc = load_risk_metrics()["concentration"]
    positions = portfolio.get("positions", [])

    st.title(k["fund_name"])
    st.caption(f"Portfolio Overview  --  As of {k['as_of_date']}  --  synthetic data")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total AUM", _m(k["aum"]))
    col2.metric("Unrealized P&L", _m(k["total_unrealized_pnl"]), delta=f"{k['total_unrealized_pnl']:+,.0f}")
    col3.metric("Positions", str(k["n_positions"]))
    col4.metric("Weighted Beta", f"{k['weighted_beta']:.2f}x",
                delta="Above market (beta > 1)" if k["weighted_beta"] > 1 else "At/below market (beta <= 1)",
                delta_color="off")
    tech_breach = k["tech_weight_pct"] > conc["sector_limit_pct"]
    col5.metric("Tech Weight", f"{k['tech_weight_pct']:.1f}%",
                delta=f"{'Above' if tech_breach else 'Within'} {conc['sector_limit_pct']:.0f}% sector limit",
                delta_color="inverse" if tech_breach else "off")

    st.divider()
    st.subheader("Sector Allocation")
    cols = st.columns(len(k["sector_weights"]))
    for i, (sector, info) in enumerate(k["sector_weights"].items()):
        cols[i].metric(sector, f"{info['weight_pct']:.1f}%", delta=f"{info['avg_pnl_pct']:+.1f}% avg P&L")

    st.divider()
    st.subheader("Positions")
    sorted_pos = sorted(positions, key=lambda x: x.get("market_value", 0), reverse=True)
    st.dataframe(pd.DataFrame([{
        "Ticker": p["ticker"],
        "Sector": p["sector"],
        "Price": f"${p['current_price']:,.2f}",
        "Market Value": f"${p['market_value']:,.0f}",
        "P&L %": f"{p.get('pnl_pct', 0):+.1f}%",
        "Weight %": f"{p.get('weight_pct', 0):.1f}%",
        "Beta": f"{p['beta']:.2f}",
        "Rating": p.get("analyst_rating", ""),
    } for p in sorted_pos]), width="stretch", hide_index=True)

    for p in sorted_pos:
        pnl = p.get("pnl_pct", 0)
        weight = p.get("weight_pct", 0)
        with st.expander(f"{p['ticker']} - {p['sector']} - {pnl:+.1f}%"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"${p['current_price']:,.2f}")
            c1.metric("Market Value", f"${p['market_value']:,.0f}")
            c2.metric("Weight", f"{weight:.1f}%")
            c2.metric("Beta", f"{p['beta']:.2f}")
            c3.metric("P&L %", f"{pnl:+.1f}%")
            c3.metric("Rating", p.get("analyst_rating", ""))
            st.caption(f"Portfolio weight vs {metrics.SINGLE_NAME_LIMIT_PCT:.0f}% single-name limit:")
            st.progress(min(weight / metrics.SINGLE_NAME_LIMIT_PCT, 1.0))


def page_market():
    inject_css()
    from src.demo.templates import demo_market_report

    st.title("Market Analysis")
    st.caption("Sector performance and technicals computed from the bundled synthetic data; "
               "report written by the LangGraph market agent (MRM-002)")

    k = load_kpis()
    st.subheader("Sector Performance")
    cols = st.columns(len(k["sector_weights"]))
    for i, (sector, info) in enumerate(k["sector_weights"].items()):
        cols[i].metric(sector, _m(info["value"]), delta=f"{info['avg_pnl_pct']:+.1f}% avg P&L")

    st.divider()
    st.subheader("Technical Alerts")
    sig = load_signals()
    st.caption(f"Latest row per held ticker in market_data.json (as of {max(s['date'] for s in sig.values())}). "
               "ALERT = RSI extreme and SMA-50 below SMA-200; WATCH = one of those.")
    ok = []
    for t, s in sorted(sig.items(), key=lambda kv: {"ALERT": 0, "WATCH": 1, "OK": 2}[kv[1]["status"]]):
        line = md_money(f"**{t} -- {s['status']}**  \n" + "; ".join(s["signals"]) +
                        f"; close ${s['close']:,.2f} on {s['date']}")
        if s["status"] == "ALERT":
            st.error(line)
        elif s["status"] == "WATCH":
            st.warning(line)
        else:
            ok.append(t)
    if ok:
        st.success(f"No RSI extreme and SMA-50 above SMA-200: {', '.join(ok)}")

    st.divider()
    st.subheader("Analysis Report  --  multi-agent orchestrator (market agent section)")
    _orchestrator_controls("run_orch_market")
    _render_section("market", lambda: demo_market_report(load_portfolio()))


def page_risk():
    inject_css()
    from src.demo.templates import demo_risk_report

    st.title("Risk Assessment")
    st.caption("VaR, Monte Carlo CVaR/Expected Shortfall, stress tests and position risk scores "
               "computed by src/utils/metrics.py (MRM-006)")

    rm = load_risk_metrics()
    var, cvar, stress, scored = rm["var"], rm["cvar"], rm["stress"], rm["scored"]

    st.subheader("Portfolio Risk Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("1-Day VaR (95%)", _m(abs(var["portfolio_var_1d"])), delta="Parametric: beta x 1.2% vol", delta_color="off")
    col2.metric("1-Day CVaR / ES (95%)", _m(abs(cvar["cvar_1d"])),
                delta=f"Monte Carlo, {cvar['n_simulations']:,} paths", delta_color="off")
    col3.metric("10-Day CVaR", _m(abs(cvar["cvar_10d"])), delta="Sqrt-of-time scaled", delta_color="off")
    col4.metric("Tail Ratio (CVaR/VaR)", f"{cvar['tail_ratio']}x", delta=cvar["tail_ratio_interpretation"], delta_color="off")
    col5.metric("Risk Level (CVaR)", cvar["risk_level"], delta=f"Weighted beta {var['weighted_beta']}x", delta_color="off")

    try:
        hm = load_history_metrics()
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Sharpe Ratio (ann.)", f"{hm['sharpe']:.2f}", delta=f"rf = latest Fed Funds {hm['risk_free_rate']:.2%}",
                  delta_color="off")
        h2.metric("Sortino Ratio (ann.)", f"{hm['sortino']:.2f}")
        h3.metric("Annualized Volatility", f"{hm['annualized_vol_pct']:.1f}%")
        h4.metric("Max Drawdown", f"{hm['max_drawdown_pct']:.1f}%")
        st.caption(f"Current positions held over the synthetic price history {hm['period']} ({hm['n_days']} days).")
    except Exception as exc:
        st.info(f"History metrics unavailable: {type(exc).__name__}")

    positions = load_portfolio()["positions"]
    total = sum(p["market_value"] for p in positions)
    contrib = sorted(positions, key=lambda p: -p["market_value"] * p["beta"])[:2]
    drivers = ", ".join(
        f"{p['ticker']} (beta {p['beta']}, {p['market_value'] * p['beta'] / sum(q['market_value'] * q['beta'] for q in positions) * 100:.1f}% of beta-weighted exposure)"
        for p in contrib
    )
    st.warning(md_money(
        f"**CVaR vs VaR -- Basel III alignment**  \n"
        f"Monte Carlo CVaR of ${abs(cvar['cvar_1d']):,.0f} is {cvar['tail_ratio']}x the Monte Carlo VaR of "
        f"${abs(cvar['var_1d']):,.0f} (parametric VaR: ${abs(var['portfolio_var_1d']):,.0f}). "
        f"On days that breach VaR, average losses are higher; Basel III FRTB uses Expected Shortfall for this reason.  \n"
        f"Largest beta-weighted exposures: {drivers}. Portfolio value ${total:,.0f}."
    ))

    st.divider()
    st.subheader("Out-of-sample VaR Backtest (99%, rolling 250-day window)")
    try:
        bt = load_backtest()
        st.caption(md_money(f"Synthetic price history {bt['period']}; {bt['n_pnl_days']} daily P&L observations of the "
                            f"current positions. Overall Basel zone: **{bt['overall_zone']}** ({bt['zone_rule']})."))
        st.dataframe(pd.DataFrame([{
            "Method": name,
            "Test days": r["test_observations"],
            "Exceptions": r["exceptions"],
            "Exceptions (last 250)": r["exceptions_last_250"],
            "Basel zone": r["basel_zone"],
            "Kupiec LR": r["kupiec_lr"],
            "Kupiec p-value": r["kupiec_p_value"],
            "Mean VaR": f"${r['mean_var']:,.0f}",
        } for name, r in bt["methods"].items()]), width="stretch", hide_index=True)
    except Exception as exc:  # e.g. price_history.json missing
        st.info(f"Backtest unavailable: {type(exc).__name__}")

    st.divider()
    st.subheader(f"Stress Test Scenarios  --  {len(stress['scenarios'])} Historical Events")
    scenarios = stress.get("scenarios", {})
    cols = st.columns(len(scenarios))
    for i, (key, s) in enumerate(scenarios.items()):
        cols[i].metric(s["scenario_name"], f"{s['portfolio_pnl_pct']:.1f}%", delta=_m(s["portfolio_pnl"]))
        sev = s.get("severity", "MODERATE")
        (cols[i].error if sev == "CATASTROPHIC" else cols[i].warning if sev == "SEVERE" else cols[i].info)(f"Severity: {sev}")

    st.divider()
    st.subheader("Position Risk Scores")
    for s in scored[:6]:
        lvl, score = s["risk_level"], s["risk_score"]
        with st.expander(f"{s['ticker']}  --  Score: {score}/100  --  {lvl}"):
            c1, c2 = st.columns(2)
            c1.metric("Risk Score", f"{score}/100")
            {"CRITICAL": c1.error, "HIGH": c1.warning, "MEDIUM": c1.info}.get(lvl, c1.success)(f"Level: {lvl}")
            c2.caption("Recommended Action:")
            c2.write(s["recommended_action"])
            st.progress(score / 100.0)
            for flag in s.get("flags", []):
                st.write(md_money(f"- {flag}"))

    st.divider()
    st.subheader("AI Risk Narrative  --  risk agent (MRM-003)")
    _orchestrator_controls("run_orch_risk")
    _render_section("risk", lambda: demo_risk_report(load_portfolio()))


def page_portfolio_management():
    inject_css()
    from src.agents.context import action_items
    from src.demo.templates import demo_portfolio_recommendations

    st.title("Portfolio Management")
    st.caption("Action items and concentration checks computed from the data; recommendation memo written by the "
               "portfolio agent (MRM-004) for human review")

    portfolio = load_portfolio()
    k = load_kpis()
    conc = load_risk_metrics()["concentration"]

    st.subheader("Immediate Action Items (from position risk scores)")
    items = action_items(portfolio)
    by_ticker = {p["ticker"]: p for p in portfolio["positions"]}
    if not items:
        st.success("No CRITICAL or HIGH risk positions.")
    for it in items:
        p = by_ticker[it["ticker"]]
        body = md_money(
            f"**{it['action'].upper()}: {it['ticker']}  --  {it['risk_level']} (score {it['risk_score']}/100)**  \n"
            f"Current: ${p['market_value']:,.0f} / {p['weight_pct']}% weight, beta {p['beta']}, P&L {p['pnl_pct']}%.  \n"
            f"Flags: {'; '.join(it['flags'])}"
        )
        (st.error if it["risk_level"] == "CRITICAL" else st.warning)(body)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"**Sector weights vs {conc['sector_limit_pct']:.0f}% policy limit (illustrative)**")
        st.dataframe(pd.DataFrame([
            [s, f"{v['weight_pct']:.2f}%", f"{conc['sector_limit_pct']:.0f}%",
             "Reduce" if v["weight_pct"] > conc["sector_limit_pct"] else "Within limit"]
            for s, v in k["sector_weights"].items()
        ], columns=["Sector", "Current", "Limit", "Status"]), width="stretch", hide_index=True)
    with col2:
        st.caption("**Tax-loss harvesting candidates (unrealized losses)**")
        st.dataframe(pd.DataFrame([
            [x["ticker"], f"${x['unrealized_pnl']:,.0f}", f"{x['pnl_pct']:.2f}%"] for x in k["loss_positions"]
        ], columns=["Ticker", "Unrealized P&L", "P&L %"]), width="stretch", hide_index=True)

    st.divider()
    st.subheader("AI Recommendations  --  portfolio agent (MRM-004)")
    _orchestrator_controls("run_orch_pm")
    _render_section("portfolio", lambda: demo_portfolio_recommendations(portfolio))


def page_chat():
    inject_css()
    from src.demo.templates import demo_chat_answer
    from src.guardrails.bedrock_guardrails import check_compliance
    from src.llm import config
    from src.rag.rag_chain import answer_question
    from src.utils.data import data_sha256
    from streamlit_app.live_ai import (
        MAX_LIVE_REQUESTS_PER_SESSION, cached_index, cached_reranker, render_output, requests_used, run_ai_request,
    )

    st.title("AI Chat")
    st.caption("RAG over the bundled data: HyDE retrieval (FAISS) -> cross-encoder re-rank -> "
               f"{config.model_id('chat')} answer -> independent grader ({config.model_id('grader')})")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    def ask(question: str):
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": question})
        compliance = check_compliance(question)
        if not compliance["compliant"]:
            st.session_state.messages.append({"role": "assistant", "content": compliance["reason"],
                                              "mode": "demo", "reason": "Blocked by compliance guardrail"})
            return

        def live():
            index = cached_index(config.model_id("embedding"), data_sha256())
            return answer_question(question, history, index=index, reranker=cached_reranker())

        with st.spinner("Retrieving (HyDE) and re-ranking..."):
            res = run_ai_request("MRM-001", live, lambda reason: demo_chat_answer(question))
        out = res["output"]
        st.session_state.messages.append({
            "role": "assistant",
            "content": out.get("answer") or out.get("text", ""),
            "mode": res["mode"] if not res.get("blocked") else "blocked",
            "reason": res["reason"],
            "grader": out.get("grader"),
            "trace": out.get("trace"),
            "sources": out.get("sources", []),
        })

    st.subheader("Suggested Queries")
    examples = [
        "What are my highest risk positions?",
        "What is the portfolio's technology weight?",
        "Should I be worried about TSLA?",
        "What does my CVaR tell me?",
    ]
    cols = st.columns(4)
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}"):
            ask(ex)
    st.caption(f"Live AI requests this session: {requests_used()}/{MAX_LIVE_REQUESTS_PER_SESSION}")
    st.divider()

    if prompt := st.chat_input("Ask about your portfolio or markets..."):
        ask(prompt)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(md_escape(msg["content"]))
                continue
            render_output(msg["content"], msg.get("mode", "demo"), msg.get("reason"), msg.get("grader"), msg.get("trace"))
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        score = s.get("rerank_score")
                        head = f"`{s['source_id']}`" + (f" · rerank {score:.3f}" if isinstance(score, (int, float)) else "")
                        if isinstance(s.get("faiss_score"), (int, float)):
                            head += f" · faiss {s['faiss_score']:.3f} ({s.get('retrieved_by')})"
                        st.markdown(md_money(head))
                        st.caption(md_money(s["text"][:400]))


def page_mrm():
    inject_css()
    from streamlit_app.mrm_dashboard import render_mrm_page

    render_mrm_page()
