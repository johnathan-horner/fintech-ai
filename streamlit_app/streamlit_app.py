"""
FinTech AI -- Streamlit Cloud Entry Point
Demo mode: synthetic data + mock AI responses, no AWS needed.
Live mode: full Amazon Bedrock + FAISS pipeline.
"""

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

DEMO_MODE = os.getenv("FINTECH_DEMO_MODE", "true").lower() != "false"

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
    .demo-banner {
        background: linear-gradient(135deg, #1e3a5f, #1a1f3a);
        border: 1px solid #3b82f6; border-radius: 8px;
        padding: 0.6rem 1rem; margin-bottom: 1rem;
        font-size: 0.85rem; color: #93c5fd;
    }
    h1, h2, h3 { color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# FinTech AI")
    st.markdown("**Hedge Fund Intelligence Platform**")
    if DEMO_MODE:
        st.success("Demo Mode")
        st.caption("Synthetic data | Mock AI responses")
    else:
        st.info("Live Mode")
        st.caption("Amazon Bedrock | Real AI pipeline")
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
    st.caption("Built by Johnathan Horner")
    st.caption("Amazon Bedrock | LangGraph | FAISS")
    st.caption("SR 11-7 MRM Framework")

if DEMO_MODE:
    st.info("**[DEMO] Demo Mode** -- Running on synthetic portfolio data with mock AI responses. No AWS credentials required.")

from streamlit_app.demo_pages import (
    page_portfolio, page_market, page_risk,
    page_portfolio_management, page_chat, page_mrm,
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
