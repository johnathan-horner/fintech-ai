"""
FinTech AI - Demo Pages
Professional dark-theme analyst dashboard.
Runs on synthetic data with no AWS credentials required.
"""

import streamlit as st
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

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

/* Page header bar */
.page-header {
    background: linear-gradient(135deg, #0f1e3d 0%, #0d1829 100%);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 24px 28px 20px;
    margin-bottom: 28px;
}
.page-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9 !important;
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}
.page-header p {
    font-size: 0.85rem;
    color: #64748b;
    margin: 0;
}

/* KPI cards */
.kpi-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.kpi-card {
    flex: 1;
    min-width: 160px;
    background: #0d1829;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 18px 20px;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.kpi-delta-pos {
    font-size: 0.78rem;
    color: #22c55e;
    margin-top: 6px;
    font-weight: 500;
}
.kpi-delta-neg {
    font-size: 0.78rem;
    color: #ef4444;
    margin-top: 6px;
    font-weight: 500;
}
.kpi-delta-neu {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 6px;
    font-weight: 500;
}
.kpi-card.accent-blue  { border-top: 3px solid #3b82f6; }
.kpi-card.accent-green { border-top: 3px solid #22c55e; }
.kpi-card.accent-red   { border-top: 3px solid #ef4444; }
.kpi-card.accent-amber { border-top: 3px solid #f59e0b; }
.kpi-card.accent-indigo{ border-top: 3px solid #6366f1; }

/* Section label */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e2d4a;
}

/* Position table rows */
.pos-row {
    display: flex;
    align-items: center;
    background: #0d1829;
    border: 1px solid #1a2740;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
    gap: 16px;
    transition: border-color 0.15s;
}
.pos-row:hover { border-color: #2d4a7a; }
.pos-ticker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    color: #f1f5f9;
    min-width: 56px;
}
.pos-sector {
    font-size: 0.72rem;
    color: #475569;
    min-width: 120px;
}
.pos-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    color: #cbd5e1;
    flex: 1;
}
.pos-pnl-pos {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    font-weight: 600;
    color: #22c55e;
    min-width: 80px;
    text-align: right;
}
.pos-pnl-neg {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    font-weight: 600;
    color: #ef4444;
    min-width: 80px;
    text-align: right;
}
.pos-badge {
    font-size: 0.68rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-buy   { background: #14532d; color: #4ade80; }
.badge-sbuy  { background: #1e3a5f; color: #60a5fa; }
.badge-hold  { background: #292524; color: #a8a29e; }
.badge-sell  { background: #450a0a; color: #f87171; }

/* Risk badge */
.risk-critical { background: #450a0a; color: #f87171; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:4px; text-transform:uppercase; }
.risk-high     { background: #431407; color: #fb923c; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:4px; text-transform:uppercase; }
.risk-medium   { background: #422006; color: #fbbf24; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:4px; text-transform:uppercase; }
.risk-low      { background: #14532d; color: #4ade80; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:4px; text-transform:uppercase; }

/* Alert cards */
.alert-critical {
    background: #1c0505;
    border: 1px solid #7f1d1d;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.alert-warning {
    background: #1c1005;
    border: 1px solid #78350f;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.alert-info {
    background: #05101c;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.alert-success {
    background: #05130a;
    border: 1px solid #14532d;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.alert-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 4px;
}
.alert-body {
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.5;
}

/* Stress scenario cards */
.stress-card {
    background: #0d1829;
    border: 1px solid #1a2740;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.stress-scenario {
    font-size: 0.72rem;
    color: #64748b;
    margin-bottom: 8px;
    font-weight: 500;
}
.stress-pnl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.stress-severity {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 2px 8px;
    border-radius: 4px;
}
.sev-catastrophic { background:#450a0a; color:#f87171; }
.sev-severe       { background:#431407; color:#fb923c; }
.sev-significant  { background:#422006; color:#fbbf24; }
.sev-moderate     { background:#1a2020; color:#94a3b8; }
.sev-mild         { background:#14532d; color:#4ade80; }

/* Progress bar for weight */
.weight-bar-track {
    background: #1a2740;
    border-radius: 4px;
    height: 4px;
    margin-top: 6px;
}
.weight-bar-fill {
    height: 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, #3b82f6, #6366f1);
}

/* Chat */
.chat-prompt-btn {
    background: #0d1829;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #94a3b8;
    cursor: pointer;
    width: 100%;
    text-align: left;
    margin-bottom: 8px;
    transition: border-color 0.15s, color 0.15s;
}
.chat-prompt-btn:hover { border-color: #3b82f6; color: #e2e8f0; }

/* Demo tag */
.demo-tag {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-left: 10px;
    vertical-align: middle;
}

/* Divider */
.divider {
    height: 1px;
    background: #1a2740;
    margin: 24px 0;
}

/* Streamlit overrides */
.stMetric { background: #0d1829; border: 1px solid #1a2740; border-radius: 8px; padding: 16px !important; }
.stMetric label { color: #475569 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
.stMetric [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-family: 'JetBrains Mono', monospace; }
div[data-testid="stExpander"] { background: #0d1829; border: 1px solid #1a2740; border-radius: 8px; }
div[data-testid="stExpander"] summary { color: #cbd5e1; }
.stButton > button { background: #1e3a5f; color: #93c5fd; border: 1px solid #2d4a7a; border-radius: 6px; font-size: 0.82rem; font-weight: 500; padding: 6px 14px; transition: all 0.15s; }
.stButton > button:hover { background: #2d4a7a; color: #f1f5f9; border-color: #3b82f6; }
.stTabs [data-baseweb="tab-list"] { background: #0d1829; border-bottom: 1px solid #1a2740; }
.stTabs [data-baseweb="tab"] { color: #64748b; font-size: 0.82rem; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #3b82f6 !important; border-bottom: 2px solid #3b82f6; }
.stChatMessage { background: #0d1829; border: 1px solid #1a2740; border-radius: 10px; }
</style>
"""

def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)

def page_header(title, subtitle):
    st.markdown(f"""
    <div class="page-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def section_label(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)

def divider():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

def kpi_card(label, value, delta=None, delta_type="neu", accent="blue"):
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta-{delta_type}">{delta}</div>'
    return f"""
    <div class="kpi-card accent-{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>"""

def alert(title, body, level="info"):
    return f"""
    <div class="alert-{level}">
        <div class="alert-title">{title}</div>
        <div class="alert-body">{body}</div>
    </div>"""

def rating_badge(rating):
    cls = {"Strong Buy": "sbuy", "Buy": "buy", "Hold": "hold", "Sell": "sell", "Strong Sell": "sell"}.get(rating, "hold")
    return f'<span class="pos-badge badge-{cls}">{rating}</span>'

# -- Data ----------------------------------------------------------------------

@st.cache_data
def load_portfolio():
    path = os.path.join(DATA_DIR, "portfolio.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "fund_name": "Horner Capital Alpha Fund",
        "aum": 5_420_000,
        "as_of_date": "2025-03-15",
        "positions": [
            {"ticker": "AAPL",  "sector": "Technology",             "shares": 2500, "avg_cost": 162.40, "current_price": 189.30, "market_value": 473250,  "unrealized_pnl":  67250, "pnl_pct":  16.6, "weight_pct":  8.73, "beta": 1.21, "52w_high": 199.62, "52w_low": 143.90, "pe_ratio": 29.4, "analyst_rating": "Buy"},
            {"ticker": "NVDA",  "sector": "Technology",             "shares":  800, "avg_cost": 495.20, "current_price": 875.40, "market_value": 700320,  "unrealized_pnl": 304160, "pnl_pct":  76.8, "weight_pct": 12.92, "beta": 1.94, "52w_high": 974.00, "52w_low": 402.18, "pe_ratio": 68.2, "analyst_rating": "Strong Buy"},
            {"ticker": "JPM",   "sector": "Financials",             "shares": 3200, "avg_cost": 148.60, "current_price": 198.70, "market_value": 635840,  "unrealized_pnl": 160320, "pnl_pct":  33.7, "weight_pct": 11.73, "beta": 1.12, "52w_high": 207.46, "52w_low": 135.19, "pe_ratio": 11.8, "analyst_rating": "Buy"},
            {"ticker": "TSLA",  "sector": "Consumer Disc",          "shares": 1500, "avg_cost": 248.90, "current_price": 177.80, "market_value": 266700,  "unrealized_pnl":-106650, "pnl_pct": -28.6, "weight_pct":  4.92, "beta": 2.18, "52w_high": 299.29, "52w_low": 138.80, "pe_ratio": 47.1, "analyst_rating": "Hold"},
            {"ticker": "MSFT",  "sector": "Technology",             "shares": 1800, "avg_cost": 310.20, "current_price": 415.50, "market_value": 747900,  "unrealized_pnl": 189540, "pnl_pct":  33.9, "weight_pct": 13.80, "beta": 0.92, "52w_high": 430.82, "52w_low": 309.45, "pe_ratio": 36.8, "analyst_rating": "Strong Buy"},
            {"ticker": "XOM",   "sector": "Energy",                 "shares": 4000, "avg_cost":  96.40, "current_price": 112.30, "market_value": 449200,  "unrealized_pnl":  63600, "pnl_pct":  16.5, "weight_pct":  8.29, "beta": 0.78, "52w_high": 123.75, "52w_low":  94.10, "pe_ratio": 13.2, "analyst_rating": "Hold"},
            {"ticker": "META",  "sector": "Technology",             "shares":  900, "avg_cost": 298.50, "current_price": 485.20, "market_value": 436680,  "unrealized_pnl": 168030, "pnl_pct":  62.6, "weight_pct":  8.06, "beta": 1.38, "52w_high": 531.49, "52w_low": 274.38, "pe_ratio": 25.3, "analyst_rating": "Buy"},
            {"ticker": "GS",    "sector": "Financials",             "shares": 1100, "avg_cost": 342.80, "current_price": 448.60, "market_value": 493460,  "unrealized_pnl": 116380, "pnl_pct":  30.9, "weight_pct":  9.10, "beta": 1.35, "52w_high": 471.45, "52w_low": 312.77, "pe_ratio": 14.6, "analyst_rating": "Buy"},
        ]
    }

MOCK_CHAT = {
    "risk":    "Highest-risk positions: TSLA (score 72/100, beta 2.18, down 28.6%) rated CRITICAL, and NVDA (score 55/100, beta 1.94, 12.9% weight) rated HIGH. Portfolio 1-day CVaR is $156,200 at 95% confidence -- tail ratio 1.44x indicates fat-tail exposure beyond what VaR captures alone.",
    "tech":    "Technology represents 43.5% of portfolio across AAPL, NVDA, MSFT, META. NVDA leads with +76.8% PnL on strong AI infrastructure demand, though P/E 68x is stretched. MSFT is highest quality -- beta 0.92, consistent earnings beats. META recovered +62.6%. Concentration above prudent limits -- recommend trimming to 30-35%.",
    "tsla":    "TSLA is rated CRITICAL. Down 28.6% from avg cost of $248.90, currently $177.80. Beta of 2.18 amplifies every market move. RSI at 31, death cross pattern on technicals. Recommend exit or reduce to max 1.5% weight with stop-loss at $165.",
    "cvar":    "CVaR (Conditional Value at Risk) is $156,200 versus VaR of $108,400 -- a tail ratio of 1.44x. This means on the days that breach your VaR threshold, average losses are 44% worse than VaR alone would suggest. Basel III uses CVaR as the primary risk measure for exactly this reason. Your tail risk is elevated, driven by TSLA and NVDA.",
    "default": "Portfolio summary: 8 positions, $5.42M AUM, weighted beta 1.31. Technology is 43.5% -- above prudent limits. Most urgent concern is TSLA: down 28.6%, beta 2.18, rated CRITICAL. Ask me about any specific position, sector, or risk metric.",
}

def get_mock_response(q):
    q = q.lower()
    if any(w in q for w in ["cvar", "expected shortfall", "tail", "var"]):
        return MOCK_CHAT["cvar"]
    if any(w in q for w in ["tsla", "tesla"]):
        return MOCK_CHAT["tsla"]
    if any(w in q for w in ["risk", "danger", "worst", "loss", "score"]):
        return MOCK_CHAT["risk"]
    if any(w in q for w in ["tech", "nvda", "aapl", "msft", "meta", "sector"]):
        return MOCK_CHAT["tech"]
    return MOCK_CHAT["default"]

# -- Pages ---------------------------------------------------------------------

def page_portfolio():
    inject_css()
    portfolio = load_portfolio()
    positions = portfolio.get("positions", [])
    total_aum = portfolio.get("aum", 0)
    total_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
    avg_beta  = sum(p.get("beta", 1) for p in positions) / len(positions) if positions else 0

    page_header(
        "Horner Capital Alpha Fund",
        f"Portfolio Overview  --  As of {portfolio.get('as_of_date', 'N/A')}  --  Demo Mode"
    )

    # KPI row
    pnl_sign = "+" if total_pnl >= 0 else ""
    kpis = [
        kpi_card("Total AUM",       f"${total_aum/1_000_000:.2f}M",   accent="blue"),
        kpi_card("Unrealized P&L",  f"${total_pnl/1_000:.0f}K",       delta=f"{pnl_sign}${total_pnl:,.0f}",  delta_type="pos" if total_pnl>=0 else "neg", accent="green" if total_pnl>=0 else "red"),
        kpi_card("Positions",       str(len(positions)),               accent="indigo"),
        kpi_card("Portfolio Beta",  f"{avg_beta:.2f}x",               delta="Above market", delta_type="neg", accent="amber"),
        kpi_card("Tech Weight",     "43.5%",                           delta="Concentration risk", delta_type="neg", accent="red"),
    ]
    st.markdown('<div class="kpi-row">' + "".join(kpis) + '</div>', unsafe_allow_html=True)

    # Sector summary
    divider()
    section_label("Sector Allocation")
    sector_map = {}
    for p in positions:
        s = p.get("sector", "Other")
        sector_map.setdefault(s, {"value": 0, "pnl": []})
        sector_map[s]["value"] += p.get("market_value", 0)
        sector_map[s]["pnl"].append(p.get("pnl_pct", 0))

    cols = st.columns(len(sector_map))
    for i, (sector, info) in enumerate(sector_map.items()):
        avg_pnl = sum(info["pnl"]) / len(info["pnl"])
        weight  = info["value"] / total_aum * 100
        cols[i].metric(sector, f"{weight:.1f}%", delta=f"{avg_pnl:+.1f}% avg P&L")

    # Positions table
    divider()
    section_label("Positions")

    sorted_pos = sorted(positions, key=lambda x: x.get("market_value", 0), reverse=True)
    for p in sorted_pos:
        pnl     = p.get("pnl_pct", 0)
        pnl_cls = "pos-pnl-pos" if pnl >= 0 else "pos-pnl-neg"
        pnl_str = f"+{pnl:.1f}%" if pnl >= 0 else f"{pnl:.1f}%"
        weight  = p.get("weight_pct", 0)
        badge   = rating_badge(p.get("analyst_rating", "Hold"))
        bar_w   = int(weight * 4)

        st.markdown(f"""
        <div class="pos-row">
            <div class="pos-ticker">{p['ticker']}</div>
            <div class="pos-sector">{p['sector']}</div>
            <div class="pos-value">
                <div>${p['current_price']:.2f}</div>
                <div class="weight-bar-track"><div class="weight-bar-fill" style="width:{bar_w}%"></div></div>
                <div style="font-size:0.68rem;color:#475569;margin-top:2px">{weight:.1f}% weight</div>
            </div>
            <div class="pos-value">${p['market_value']:,.0f}</div>
            <div class="{pnl_cls}">{pnl_str}</div>
            <div style="font-size:0.78rem;color:#64748b;">beta {p['beta']:.2f}</div>
            {badge}
        </div>""", unsafe_allow_html=True)


def page_market():
    inject_css()
    page_header("Market Analysis", "AI-powered trend analysis using RAG over earnings transcripts and market data  --  Demo Mode")

    portfolio = load_portfolio()
    positions = portfolio.get("positions", [])

    # Sector breakdown
    section_label("Sector Performance")
    sector_map = {}
    for p in positions:
        s = p.get("sector", "Other")
        sector_map.setdefault(s, {"value": 0, "pnl": []})
        sector_map[s]["value"] += p.get("market_value", 0)
        sector_map[s]["pnl"].append(p.get("pnl_pct", 0))

    cols = st.columns(len(sector_map))
    for i, (sector, info) in enumerate(sector_map.items()):
        avg_pnl = sum(info["pnl"]) / len(info["pnl"])
        cols[i].metric(sector, f"${info['value']/1000:.0f}K", delta=f"{avg_pnl:+.1f}%")

    divider()
    section_label("Technical Alerts")

    alerts_html = (
        alert("TSLA -- CRITICAL SIGNAL",
              "RSI 31 approaching oversold. SMA-50 crossed below SMA-200 (death cross). Bearish trend intact. Recommend review.",
              "critical") +
        alert("NVDA -- MONITOR",
              "RSI 72 approaching overbought. Strong uptrend but P/E 68x is stretched. Watch for reversal signal.",
              "warning") +
        alert("MSFT -- HEALTHY",
              "RSI 58, trading above both SMA-50 and SMA-200. No technical concerns. Hold.",
              "success")
    )
    st.markdown(alerts_html, unsafe_allow_html=True)

    divider()
    section_label("Analysis Report  --  Bedrock Claude 3 Sonnet")

    st.markdown("""
**Technology (43.5% weight):** Strong momentum across NVDA, MSFT, and META.
NVDA benefits from AI infrastructure demand with revenue guidance ahead of consensus.
MSFT Azure growth re-accelerating. META ad recovery intact. Concentration risk is elevated.

**Financials (20.8% weight):** JPM and GS benefiting from elevated rates and M&A reactivation.
Net interest margin expansion supporting earnings beats. Rate cut expectations reducing near-term tailwind.

**Energy (8.3% weight):** XOM trading below 52-week high on demand concerns.
OPEC+ discipline supporting price floor but China demand remains uncertain.

**Consumer Discretionary (4.9% weight):** TSLA is the critical concern -- down 28.6% from cost.
Margin compression, aggressive price cuts, and BYD competition creating sustained pressure.

**Key Risk -- Technology Concentration:** 43.5% in a single sector creates significant single-factor exposure.
A tech sector rotation or valuation multiple compression from a rate shock would disproportionately impact this portfolio.
Recommend reducing technology exposure toward 30-35% through selective trimming of NVDA and META.
    """)


def page_risk():
    inject_css()
    page_header("Risk Assessment", "VaR, CVaR/Expected Shortfall, Monte Carlo stress tests, position-level risk scoring  --  Demo Mode")

    portfolio = load_portfolio()
    positions = portfolio.get("positions", [])

    try:
        from src.utils.metrics import calculate_portfolio_var, calculate_portfolio_cvar, run_stress_tests, score_position_risk
        var    = calculate_portfolio_var(positions)
        cvar   = calculate_portfolio_cvar(positions, n_simulations=5000)
        stress = run_stress_tests(positions)
        scored = sorted([score_position_risk(p) for p in positions], key=lambda x: x["risk_score"], reverse=True)
    except Exception:
        var    = {"portfolio_var_1d": -108400, "weighted_beta": 1.31}
        cvar   = {"cvar_1d": -156200, "cvar_10d": -494000, "cvar_1d_pct": 2.88, "tail_ratio": 1.44, "tail_ratio_interpretation": "Elevated tail risk", "risk_level": "HIGH"}
        stress = {"summary": {"worst_case_scenario": "2008 Global Financial Crisis", "worst_case_pnl": -1724000, "worst_case_pnl_pct": -31.8, "catastrophic_scenarios": 2}, "scenarios": {
            "2008_gfc":        {"scenario_name": "2008 GFC",          "portfolio_pnl_pct": -31.8, "portfolio_pnl": -1724000, "severity": "CATASTROPHIC"},
            "covid_crash":     {"scenario_name": "COVID Crash",       "portfolio_pnl_pct": -27.0, "portfolio_pnl": -1461000, "severity": "SEVERE"},
            "rate_shock":      {"scenario_name": "Rate Shock +200bps","portfolio_pnl_pct": -11.3, "portfolio_pnl":  -614000, "severity": "SIGNIFICANT"},
            "tech_selloff":    {"scenario_name": "Tech Selloff -30%", "portfolio_pnl_pct": -16.4, "portfolio_pnl":  -892000, "severity": "SEVERE"},
            "flash_crash":     {"scenario_name": "Flash Crash",       "portfolio_pnl_pct":  -4.2, "portfolio_pnl":  -228000, "severity": "MODERATE"},
        }}
        scored = []

    # VaR / CVaR KPIs
    section_label("Portfolio Risk Metrics")
    kpis = [
        kpi_card("1-Day VaR (95%)",    f"${abs(var.get('portfolio_var_1d',0)):,.0f}",  delta="Parametric estimate",  delta_type="neu",  accent="amber"),
        kpi_card("1-Day CVaR / ES",    f"${abs(cvar.get('cvar_1d',0)):,.0f}",          delta="Basel III preferred",  delta_type="neg",  accent="red"),
        kpi_card("10-Day CVaR",        f"${abs(cvar.get('cvar_10d',0)):,.0f}",         delta="Sqrt-of-time scaled",  delta_type="neg",  accent="red"),
        kpi_card("Tail Ratio",         f"{cvar.get('tail_ratio','N/A')}x",             delta=cvar.get('tail_ratio_interpretation',''), delta_type="neg", accent="amber"),
        kpi_card("Risk Level",         cvar.get("risk_level","N/A"),                  delta="Weighted beta 1.31x",  delta_type="neg",  accent="red"),
    ]
    st.markdown('<div class="kpi-row">' + "".join(kpis) + '</div>', unsafe_allow_html=True)

    st.markdown(alert(
        "CVaR vs VaR -- Basel III Alignment",
        f"CVaR of ${abs(cvar.get('cvar_1d',0)):,.0f} is {cvar.get('tail_ratio','N/A')}x the VaR of ${abs(var.get('portfolio_var_1d',0)):,.0f}. "
        "On days that breach VaR, average losses are materially higher. "
        "Basel III FRTB uses Expected Shortfall as the primary measure for this reason. "
        "Tail risk is driven primarily by TSLA (beta 2.18) and NVDA (beta 1.94).",
        "warning"
    ), unsafe_allow_html=True)

    divider()
    section_label("Stress Test Scenarios  --  5 Historical Events")

    scenarios = stress.get("scenarios", {})
    sev_cls = {"CATASTROPHIC": "catastrophic", "SEVERE": "severe", "SIGNIFICANT": "significant", "MODERATE": "moderate", "MILD": "mild"}

    cols = st.columns(len(scenarios))
    for i, (key, s) in enumerate(scenarios.items()):
        pnl     = s.get("portfolio_pnl_pct", 0)
        pnl_clr = "#ef4444" if pnl < -20 else "#f59e0b" if pnl < -10 else "#fbbf24"
        sev     = s.get("severity","MODERATE")
        sc      = sev_cls.get(sev, "moderate")
        cols[i].markdown(f"""
        <div class="stress-card">
            <div class="stress-scenario">{s['scenario_name']}</div>
            <div class="stress-pnl" style="color:{pnl_clr}">{pnl:.1f}%</div>
            <div style="font-size:0.72rem;color:#64748b;margin-bottom:8px">${s.get('portfolio_pnl',0):,.0f}</div>
            <span class="stress-severity sev-{sc}">{sev}</span>
        </div>""", unsafe_allow_html=True)

    divider()
    section_label("Position Risk Scores")

    if not scored:
        scored = [
            {"ticker": "TSLA", "risk_score": 72, "risk_level": "CRITICAL", "recommended_action": "Consider immediate exit or hedge", "flags": ["Very high beta (2.18x)", "Severe drawdown (-28.6%)", "Moderate concentration (4.9%)"]},
            {"ticker": "NVDA", "risk_score": 55, "risk_level": "HIGH",     "recommended_action": "Review position size and add stop-loss", "flags": ["Elevated beta (1.94x)", "High concentration (12.9%)"]},
            {"ticker": "GS",   "risk_score": 28, "risk_level": "MEDIUM",   "recommended_action": "Monitor closely", "flags": ["Elevated beta (1.35x)"]},
            {"ticker": "META", "risk_score": 20, "risk_level": "MEDIUM",   "recommended_action": "Monitor closely", "flags": ["Elevated beta (1.38x)"]},
            {"ticker": "MSFT", "risk_score":  8, "risk_level": "LOW",      "recommended_action": "Hold -- within risk tolerance", "flags": []},
            {"ticker": "AAPL", "risk_score":  7, "risk_level": "LOW",      "recommended_action": "Hold -- within risk tolerance", "flags": []},
        ]

    for s in scored[:6]:
        lvl = s.get("risk_level", "LOW")
        lvl_cls = {"CRITICAL": "critical", "HIGH": "high", "MEDIUM": "medium", "LOW": "low"}.get(lvl, "low")
        score = s.get("risk_score", 0)
        bar_color = "#ef4444" if score >= 60 else "#f59e0b" if score >= 35 else "#fbbf24" if score >= 15 else "#22c55e"
        flags_html = "".join(f'<div style="font-size:0.75rem;color:#64748b;margin-top:3px">-- {f}</div>' for f in s.get("flags", []))

        with st.expander(f"{s['ticker']}  --  Score: {score}/100  --  {lvl}"):
            st.markdown(f"""
            <div style="margin-bottom:12px">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
                    <span class="risk-{lvl_cls}">{lvl}</span>
                    <span style="font-size:0.8rem;color:#94a3b8">{s['recommended_action']}</span>
                </div>
                <div style="background:#1a2740;border-radius:4px;height:6px;margin-bottom:10px">
                    <div style="width:{score}%;height:6px;border-radius:4px;background:{bar_color}"></div>
                </div>
                {flags_html}
            </div>""", unsafe_allow_html=True)


def page_portfolio_management():
    inject_css()
    page_header("Portfolio Management", "AI-generated rebalancing, hedges, and action items  --  Demo Mode")

    section_label("Immediate Action Items")
    st.markdown(
        alert("EXIT / REDUCE TSLA  --  CRITICAL",
              "Current: $266,700 / 4.9% weight. Target: exit entirely or reduce to max 1.5% ($82,000). "
              "Rationale: -28.6% drawdown, beta 2.18, deteriorating analyst consensus. Stop-loss if holding: $165.",
              "critical") +
        alert("TRIM NVDA  --  HIGH PRIORITY",
              "Current: $700,320 / 12.9% weight -> Target: 8% ($435,000). Sell approx. $265,000. "
              "Lock in +76.8% gain on trimmed shares. P/E 68x, RSI approaching overbought, concentration risk.",
              "warning") +
        alert("HEDGE TECH EXPOSURE",
              "Buy SPY put options (5% OTM, 90-day expiry). Cost: approx. 1.2% of AUM (~$65,000 premium). "
              "Protects against GFC/COVID-style scenario (-$1.7M unhedged).",
              "warning"),
        unsafe_allow_html=True
    )

    divider()
    section_label("Rebalancing Plan  --  1-2 Week Horizon")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Target Sector Weights**")
        targets = [
            ("Technology",      "43.5%", "30%",  "neg"),
            ("Financials",      "20.8%", "20.8%","neu"),
            ("Healthcare",      "0%",    "8%",   "pos"),
            ("Energy",          "8.3%",  "8%",   "neu"),
            ("Consumer Disc",   "4.9%",  "2%",   "neg"),
            ("Cash",            "0%",    "5%",   "pos"),
        ]
        for sector, curr, tgt, direction in targets:
            color = "#22c55e" if direction == "pos" else "#ef4444" if direction == "neg" else "#94a3b8"
            arrow = "+" if direction == "pos" else "->" if direction == "neg" else "="
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:8px 0;border-bottom:1px solid #1a2740;font-size:0.83rem">
                <span style="color:#cbd5e1">{sector}</span>
                <span style="color:#64748b">{curr}</span>
                <span style="color:{color};font-weight:600">{arrow} {tgt}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("**Macro Positioning  --  1-3 Month View**")
        st.markdown("""
        <div style="font-size:0.82rem;line-height:1.8;color:#94a3b8">
        <div style="margin-bottom:8px"><strong style="color:#cbd5e1">Rate sensitivity:</strong>
        Portfolio net short duration. Rate cuts benefit growth names. Hold current positioning.</div>
        <div style="margin-bottom:8px"><strong style="color:#cbd5e1">Defensive add:</strong>
        Consider UNH or JNJ to reduce weighted beta from 1.31 toward 1.10.</div>
        <div style="margin-bottom:8px"><strong style="color:#cbd5e1">Cash target:</strong>
        5% cash from TSLA/NVDA proceeds. Dry powder for re-entry opportunities.</div>
        <div><strong style="color:#cbd5e1">Tax-loss harvesting:</strong>
        TSLA unrealized loss of $106,650 available. Coordinate with CPA before year-end.</div>
        </div>""", unsafe_allow_html=True)


def page_chat():
    inject_css()
    page_header("AI Chat", "Natural language Q&A -- portfolio, risk, markets, macro  --  Demo Mode")

    section_label("Suggested Queries")
    examples = [
        "What are my highest risk positions?",
        "How is the tech sector performing?",
        "Should I be worried about TSLA?",
        "What does my CVaR tell me?",
    ]
    cols = st.columns(4)
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}"):
            if "messages" not in st.session_state:
                st.session_state.messages = []
            st.session_state.messages.append({"role": "user", "content": ex})
            st.session_state.messages.append({"role": "assistant", "content": get_mock_response(ex)})

    divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your portfolio or markets..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        response = get_mock_response(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


def page_mrm():
    inject_css()
    from streamlit_app.mrm_dashboard import render_mrm_page
    render_mrm_page()
