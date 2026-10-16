"""
FinTech AI -- Streamlit Cloud Entry Point

Quant metrics are always computed from the bundled synthetic data.
AI features: Amazon Bedrock Claude Haiku 4.5 (st.secrets["aws"]) first, Google Gemini
free tier (st.secrets["gemini"]["api_key"]) as fallback, labelled data-derived demo output
last - gated by model governance and capped at 10 live requests per session.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import config  # noqa: E402

st.set_page_config(
    page_title="FinTech AI -- Hedge Fund Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; color: #e2e8f0; }
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #7c3aed);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
    }
    h1, h2, h3 { color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

LIVE_AVAILABLE = config.has_credentials()

with st.sidebar:
    st.markdown("# FinTech AI")
    st.markdown("**Hedge Fund Intelligence Platform**")
    from streamlit_app.live_ai import provider_status_lines  # noqa: E402

    if LIVE_AVAILABLE:
        st.success("Live AI available")
    else:
        st.warning("Demo mode")
    for line in provider_status_lines():
        st.caption(line)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "Portfolio Overview",
            "Market Analysis",
            "Risk Assessment",
            "Portfolio Management",
            "AI Chat",
            "Model Risk Management",
        ],
        key="main_nav",
    )
    st.markdown("---")
    from streamlit_app.live_ai import MAX_LIVE_REQUESTS_PER_SESSION, requests_used  # noqa: E402

    with st.expander("System"):
        counter_slot = st.empty()  # filled after the page ran, so the count includes this run's request
        trace_slot = st.empty()
        st.caption(f"Provider chain: Bedrock {config.display_name(config.HAIKU_45)} -> Gemini "
                   f"{config.model_id('chat', 'gemini')} -> demo mode")
        st.caption(f"Embeddings: {config.model_id('embedding')} · re-ranker: ms-marco-MiniLM-L-6-v2 (local ONNX)")
        st.caption("Bedrock Guardrails, KMS and CloudTrail: defined in infra/ (CDK) only - not used by this live demo")
        st.caption(config.free_tier_label())
    st.caption("Built by Johnathan Horner")
    st.caption("LangGraph | FAISS | Bedrock Claude Haiku 4.5 | Gemini fallback")
    st.caption("SR 11-7 MRM Framework")

if LIVE_AVAILABLE:
    st.info("Synthetic portfolio data. AI outputs are labelled **Live AI** or **Demo mode**; "
            f"live use is gated by model approvals and capped at {MAX_LIVE_REQUESTS_PER_SESSION} requests per session.")
else:
    st.info("**Demo mode** -- synthetic portfolio data; quant metrics are computed live, AI sections show "
            "deterministic templates filled from the data (no Bedrock credentials and no Gemini API key configured).")

from streamlit_app.demo_pages import (  # noqa: E402
    page_chat, page_market, page_mrm, page_portfolio, page_portfolio_management, page_risk,
)

if page == "Portfolio Overview":
    page_portfolio()
elif page == "Market Analysis":
    page_market()
elif page == "Risk Assessment":
    page_risk()
elif page == "Portfolio Management":
    page_portfolio_management()
elif page == "AI Chat":
    page_chat()
elif page == "Model Risk Management":
    page_mrm()

st.markdown("---")
st.caption(config.free_tier_label())

counter_slot.caption(f"Live AI requests this session: {requests_used()}/{MAX_LIVE_REQUESTS_PER_SESSION}")
trace_slot.caption(f"Traced outputs this session: {len(st.session_state.get('trace_log', []))}")
