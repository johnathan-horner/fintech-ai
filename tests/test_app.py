"""Streamlit AppTest over every page with NO API key (demo mode)."""

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "streamlit_app", "streamlit_app.py"))
PAGES = ["Portfolio Overview", "Market Analysis", "Risk Assessment", "Portfolio Management", "AI Chat",
         "Model Risk Management"]


def _text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.label for e in at.expander] + [str(x.value) for x in at.info] + [str(x.value) for x in at.warning]
    parts += [str(x.value) for x in at.error] + [str(x.value) for x in at.success]
    return "\n".join(parts)


def _open(page):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio(key="main_nav").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_key(page):
    at = _open(page)
    txt = _text(at)
    assert "Demo mode" in txt
    assert "5.42M" not in txt and "43.5%" not in txt and "Weighted beta 1.31" not in txt
    assert "Amazon Bedrock Claude Haiku 4.5 (primary)" in txt and "not used by this live demo" in txt


def test_portfolio_kpis_are_computed():
    at = _open("Portfolio Overview")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Total AUM"] == "$32.36M"
    assert metrics["Positions"] == "15"
    assert metrics["Tech Weight"] == "47.0%"


def test_ai_sections_fall_back_with_labels():
    at = _open("Market Analysis")
    at.button(key="run_orch_market").click().run()
    assert not at.exception
    txt = _text(at)
    assert "Demo mode" in txt and "No LLM provider configured" in txt
    assert any(e.label.startswith("Trace ") for e in at.expander)


def test_chat_demo_answer_with_trace():
    at = _open("AI Chat")
    at.chat_input[0].set_value("What does my CVaR tell me? $5 test").run()
    assert not at.exception
    txt = _text(at)
    assert "No LLM provider configured" in txt
    assert "Grader: LLM grader not run (demo output)" in txt
    assert any(e.label.startswith("Trace ") for e in at.expander)


def test_mrm_run_all_models_and_real_data():
    at = _open("Model Risk Management")
    txt = _text(at)
    assert "MRM-006" in txt and "NOT YET VALIDATED" in txt
    at.checkbox(key="val_run_all").check().run()
    at.button(key="val_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [e.label for e in at.expander]
    for mid in ("MRM-001", "MRM-002", "MRM-003", "MRM-004", "MRM-005", "MRM-006"):
        assert any(label.startswith(f"{mid} - score") for label in labels), mid
    assert any("historical_rolling" in str(df.value) for df in at.dataframe)
