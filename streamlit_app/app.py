"""
FinTech AI - Analyst Dashboard (Streamlit)
Hedge fund intelligence interface: portfolio overview, market analysis,
risk assessment, portfolio recommendations, and natural language chat.
"""

import streamlit as st
import requests
import json
import os

API_URL = os.getenv("FINTECH_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FinTech AI - Hedge Fund Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ?? Custom CSS 
st.markdown("""
<style>
    .main { background-color: #0a0e1a; }
    .stApp { background-color: #0a0e1a; color: #e2e8f0; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .risk-critical { color: #ef4444; font-weight: bold; }
    .risk-high { color: #f97316; font-weight: bold; }
    .risk-medium { color: #eab308; font-weight: bold; }
    .risk-low { color: #22c55e; font-weight: bold; }
    .action-card {
        background: #1e293b;
        border-left: 4px solid #ef4444;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
    }
    h1, h2, h3 { color: #f1f5f9 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ?? Sidebar 
with st.sidebar:
    st.markdown("#  FinTech AI")
    st.markdown("**Hedge Fund Intelligence Platform**")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [" Portfolio Overview", "Search Market Analysis", "[WARN] Risk Assessment",
         " Portfolio Management", " AI Chat", "MRM Model Risk Management"]
    )
    st.markdown("---")
    st.caption("Powered by Amazon Bedrock ? Claude 3 Sonnet")
    st.caption("LangGraph ? FAISS ? RAG")


# ?? Data Loaders 

@st.cache_data(ttl=300)
def load_portfolio():
    try:
        r = requests.get(f"{API_URL}/portfolio/summary", timeout=10)
        return r.json()
    except Exception:
        # Load from file directly if API is not running
        data_path = os.path.join(os.path.dirname(__file__), "../data/portfolio.json")
        if os.path.exists(data_path):
            with open(data_path) as f:
                return json.load(f)
        return {}


def call_api(endpoint: str) -> dict:
    try:
        r = requests.post(f"{API_URL}{endpoint}", timeout=120)
        return r.json()
    except Exception as e:
        return {"success": False, "data": {}, "error": str(e)}


def risk_color(level: str) -> str:
    return {
        "CRITICAL": "risk-critical",
        "HIGH": "risk-high",
        "MEDIUM": "risk-medium",
        "LOW": "risk-low",
    }.get(level, "")


# ?? Page: Portfolio Overview 

def page_portfolio():
    st.title(" Portfolio Overview")
    portfolio = load_portfolio()

    if not portfolio:
        st.error("Could not load portfolio data. Ensure the API is running.")
        return

    positions = portfolio.get("positions", [])
    total_aum = portfolio.get("aum", 0)

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total AUM", f"${total_aum:,.0f}")
    col2.metric("Positions", len(positions))
    total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
    col3.metric("Total Unrealized PnL", f"${total_pnl:,.0f}",
                delta=f"{'^' if total_pnl > 0 else 'v'} ${abs(total_pnl):,.0f}")
    avg_beta = round(sum(p.get("beta", 1) for p in positions) / len(positions), 2) if positions else 0
    col4.metric("Avg Portfolio Beta", avg_beta)

    st.markdown("---")
    st.subheader(" Positions")

    # Sort by market value
    positions_sorted = sorted(positions, key=lambda x: x.get("market_value", 0), reverse=True)

    for p in positions_sorted:
        pnl = p.get("pnl_pct", 0)
        pnl_color = "#22c55e" if pnl >= 0 else "#ef4444"
        pnl_arrow = "^" if pnl >= 0 else "v"

        with st.expander(
            f"**{p['ticker']}** - ${p['market_value']:,.0f} | "
            f"{pnl_arrow} {abs(pnl):.1f}% | ? {p['beta']} | {p['sector']}"
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Shares", f"{p['shares']:,}")
            c2.metric("Avg Cost", f"${p['avg_cost']}")
            c3.metric("Current Price", f"${p['current_price']}")
            c4.metric("Weight", f"{p['weight_pct']}%")
            c1.metric("52W High", f"${p['52w_high']}")
            c2.metric("52W Low", f"${p['52w_low']}")
            c3.metric("P/E Ratio", p.get('pe_ratio', 'N/A'))
            c4.metric("Analyst Rating", p.get("analyst_rating", "N/A"))


# ?? Page: Market Analysis 

def page_market():
    st.title("Search Market Analysis")
    st.markdown("AI-powered market trend analysis using RAG over earnings transcripts and market data.")

    if st.button(" Run Market Analysis"):
        with st.spinner("Analyzing market trends..."):
            result = call_api("/insights/market")

        if result.get("success"):
            data = result["data"]

            if data.get("technical_alerts"):
                st.subheader("? Technical Alerts")
                for alert in data["technical_alerts"]:
                    st.warning(alert)

            if data.get("sector_breakdown"):
                st.subheader(" Sector Breakdown")
                cols = st.columns(min(len(data["sector_breakdown"]), 4))
                for i, (sector, info) in enumerate(data["sector_breakdown"].items()):
                    col = cols[i % len(cols)]
                    pnl = info.get("avg_pnl_pct", 0)
                    col.metric(
                        sector,
                        f"${info.get('total_value', 0):,.0f}",
                        delta=f"{pnl:+.1f}% avg PnL"
                    )

            if data.get("market_analysis"):
                st.subheader(" Market Analysis Report")
                st.markdown(data["market_analysis"])
        else:
            st.error(f"Analysis failed: {result.get('error')}")


# ?? Page: Risk Assessment 

def page_risk():
    st.title("[WARN] Risk Assessment")
    st.markdown("Quantitative risk scoring: VaR, Beta, drawdown, and concentration risk.")

    if st.button("[HIGH] Run Risk Assessment"):
        with st.spinner("Calculating portfolio risk..."):
            result = call_api("/insights/risk")

        if result.get("success"):
            data = result["data"]

            # VaR + CVaR Summary
            var = data.get("portfolio_var", {})
            if var:
                st.subheader("[DOWN] Portfolio VaR & CVaR (Expected Shortfall)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("1-Day VaR (95%)", f"${abs(var.get('portfolio_var_1d', 0)):,.0f}")
                c2.metric("1-Day CVaR/ES (95%)", f"${abs(var.get('cvar_1d', 0)):,.0f}",
                          delta=f"Tail ratio: {var.get('tail_ratio', 'N/A')}x",
                          delta_color="inverse")
                c3.metric("10-Day CVaR", f"${abs(var.get('cvar_10d', 0)):,.0f}")
                risk_level = var.get("risk_level", "Unknown")
                c4.metric("Portfolio Risk Level", risk_level)

                tail_interp = var.get("tail_ratio_interpretation", "")
                if tail_interp:
                    st.info(f" **Tail Risk:** {tail_interp}. {var.get('regulatory_note', '')}")

                # Stress Test Results
                stress_scenarios = var.get("stress_scenarios", {})
                if stress_scenarios:
                    st.subheader("[STRESS] Stress Test Scenarios")
                    worst_scenario = var.get("stress_worst_case_scenario", "")
                    st.warning(
                        f"[HIGH] **Worst Case:** {worst_scenario} - "
                        f"Portfolio loss: {var.get('stress_worst_case_pnl_pct', 0):.1f}% "
                        f"(${abs(var.get('stress_worst_case_pnl', 0)):,.0f})"
                    )

                    scenario_cols = st.columns(min(len(stress_scenarios), 3))
                    for i, (key, scenario) in enumerate(stress_scenarios.items()):
                        col = scenario_cols[i % 3]
                        severity = scenario.get("severity", "")
                        severity_icon = {
                            "CATASTROPHIC": "[CRITICAL]", "SEVERE": "[HIGH]",
                            "SIGNIFICANT": "?", "MODERATE": "[MED]", "MILD": "[LOW]"
                        }.get(severity, "?")
                        col.metric(
                            f"{severity_icon} {scenario['scenario_name']}",
                            f"{scenario.get('portfolio_pnl_pct', 0):.1f}%",
                            delta=f"${scenario.get('portfolio_pnl', 0):,.0f}",
                            delta_color="inverse"
                        )

            # Position Risk Scores
            scores = data.get("position_risk_scores", [])
            if scores:
                st.subheader(" Position Risk Scores")
                for s in scores[:10]:
                    level = s.get("risk_level", "LOW")
                    css = risk_color(level)
                    with st.expander(
                        f"**{s['ticker']}** - Risk Score: {s['risk_score']}/100 | "
                        f"Level: {level}"
                    ):
                        st.markdown(f"**Action:** {s['recommended_action']}")
                        for flag in s.get("flags", []):
                            st.markdown(f"? {flag}")

            if data.get("risk_report"):
                st.subheader(" Full Risk Report")
                st.markdown(data["risk_report"])
        else:
            st.error(f"Risk assessment failed: {result.get('error')}")


# ?? Page: Portfolio Management 

def page_portfolio_management():
    st.title(" Portfolio Management")
    st.markdown("Full AI pipeline: Market -> Risk -> Portfolio recommendations.")
    st.warning("?? This runs all three agents sequentially. May take 60-90 seconds.")

    if st.button("? Run Full AI Analysis"):
        with st.spinner("Running multi-agent pipeline (Market -> Risk -> Portfolio)..."):
            result = call_api("/insights/portfolio")

        if result.get("success"):
            data = result["data"]

            # Action Items
            actions = data.get("action_items", [])
            if actions:
                st.subheader("? Immediate Action Items")
                for action in actions:
                    priority = action.get("priority", "HIGH")
                    border_color = "#ef4444" if priority == "URGENT" else "#f97316"
                    st.markdown(
                        f"""<div class="action-card">
                        <strong>{action['ticker']}</strong> - 
                        {action.get('action', '')} 
                        <span style="float:right; color:{border_color}">
                        {action.get('risk_level')} Risk
                        </span></div>""",
                        unsafe_allow_html=True
                    )

            # Recommendations
            if data.get("portfolio_recommendations"):
                st.subheader(" Portfolio Recommendations")
                st.markdown(data["portfolio_recommendations"])

            # Errors
            if data.get("errors"):
                with st.expander(" Pipeline Warnings"):
                    for err in data["errors"]:
                        st.warning(err)
        else:
            st.error(f"Pipeline failed: {result.get('error')}")


# ?? Page: AI Chat 

def page_chat():
    st.title(" AI Chat - Ask Anything")
    st.markdown("Natural language Q&A about your portfolio, markets, earnings, and macro themes.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Example queries
    st.markdown("**Example queries:**")
    examples = [
        "What are my highest risk positions?",
        "How is the tech sector performing?",
        "Summarize NVDA's last earnings call",
        "What does the current Fed rate environment mean for my portfolio?",
        "Which positions have the highest beta?",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}"):
            st.session_state.messages.append({"role": "user", "content": ex})

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask about your portfolio or markets..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    r = requests.post(
                        f"{API_URL}/chat",
                        json={"question": prompt},
                        timeout=60,
                    )
                    data = r.json()
                    answer = data.get("answer", "No answer returned.")
                except Exception as e:
                    answer = f"Error: {e}"
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


# ?? Router 

if page == " Portfolio Overview":
    page_portfolio()
elif page == "Search Market Analysis":
    page_market()
elif page == "[WARN] Risk Assessment":
    page_risk()
elif page == " Portfolio Management":
    page_portfolio_management()
elif page == " AI Chat":
    page_chat()
elif page == "MRM Model Risk Management":
    from streamlit_app.mrm_dashboard import render_mrm_page
    render_mrm_page()
