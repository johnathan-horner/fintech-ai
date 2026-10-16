"""
FinTech AI - live AI guard for Streamlit.

* Session cap: at most MAX_LIVE_REQUESTS_PER_SESSION (10) live AI *requests*
  per browser session, tracked in st.session_state. One user action is one
  request even when it makes several model calls (e.g. one chat question =
  HyDE passage + answer + grader; one multi-agent run = 3 agent calls + 3
  grader calls). Requests blocked by model governance before any model call
  are not counted.
* Fallback to deterministic demo output (with the reason shown) when the key
  is missing, the cap is reached, governance blocks the call, the Gemini free
  tier returns 429 / RESOURCE_EXHAUSTED (after one retry), or anything fails.
* Every output gets a visible "Live AI" or "Demo mode" badge, the grader
  result, and its TraceRecord in an expander. Never raises.
"""

from __future__ import annotations

import html
from typing import Callable, MutableMapping, Optional

import streamlit as st

from src.llm import config
from src.llm.guard import run_guarded
from src.mrm.trace import attach_demo_trace
from streamlit_app.formatting import md_money

MAX_LIVE_REQUESTS_PER_SESSION = 10
CAP_REASON = f"Session cap reached ({MAX_LIVE_REQUESTS_PER_SESSION} live AI requests per session)"


def _state(state: Optional[MutableMapping]) -> MutableMapping:
    state = st.session_state if state is None else state
    if "live_requests_used" not in state:
        state["live_requests_used"] = 0
    if "trace_log" not in state:
        state["trace_log"] = []
    return state


def requests_used(state: Optional[MutableMapping] = None) -> int:
    return _state(state)["live_requests_used"]


def _collect_traces(output) -> list:
    """Find TraceRecord dicts in an output payload (top level + sections)."""
    traces = []
    if isinstance(output, dict):
        if isinstance(output.get("trace"), dict):
            traces.append(output["trace"])
        for key in ("market", "risk", "portfolio"):
            sec = output.get(key)
            if isinstance(sec, dict) and isinstance(sec.get("trace"), dict):
                traces.append(sec["trace"])
    return traces


def run_ai_request(
    inventory_id: str,
    live_fn: Callable[[], dict],
    demo_fn: Callable[[str], dict],
    *,
    state: Optional[MutableMapping] = None,
    credentials_present: Optional[bool] = None,
) -> dict:
    """Run one user-initiated AI request under the session cap and guard."""
    state = _state(state)
    if credentials_present is None:
        credentials_present = config.has_credentials()

    if credentials_present and state["live_requests_used"] >= MAX_LIVE_REQUESTS_PER_SESSION:
        try:
            payload = demo_fn(CAP_REASON)
        except Exception as exc:  # pragma: no cover - demo must not crash
            payload = {"text": f"Demo output unavailable ({type(exc).__name__})."}
        result = {"mode": "demo", "reason": CAP_REASON, "blocked": False, "output": payload, "latency_ms": 0.0}
    else:
        if credentials_present:
            state["live_requests_used"] += 1
        result = run_guarded(live_fn, demo_fn, credentials_present=credentials_present)
        if result["blocked"] and credentials_present:
            state["live_requests_used"] -= 1  # blocked before any model call

    if result["mode"] != "live":
        attach_demo_trace(inventory_id, result["output"], result["reason"], result["blocked"])
    traces = _collect_traces(result["output"])
    state["trace_log"].extend(traces)
    result["inventory_id"] = inventory_id
    return result


# -- Rendering ------------------------------------------------------------------

def output_label(mode: str, reason: Optional[str] = None, trace: Optional[dict] = None,
                 fallback: Optional[str] = None) -> str:
    """'Live AI · Bedrock · Claude Haiku 4.5' / 'Live AI · Gemini · <model> (fallback: ...)' / 'Demo mode (...)'."""
    if mode == "live":
        trace = trace or {}
        model = trace.get("answer_model") or (trace.get("model_id") or "").split("+")[0]
        label = f"Live AI \u00b7 {config.provider_label(trace.get('provider'), model)}"
        fb = fallback or trace.get("fallback_reason")
        if fb:
            label += f" (fallback: {fb[:180] + ('...' if len(fb) > 180 else '')})"
        return label
    return "Demo mode" + (f" ({reason})" if reason else "")


def render_badge(mode: str, reason: Optional[str] = None, trace: Optional[dict] = None,
                 fallback: Optional[str] = None) -> None:
    text = html.escape(output_label(mode, reason, trace, fallback))
    if mode == "live":
        colors = ("#14532d", "#bbf7d0")
    elif mode == "blocked" or (reason or "").startswith("Blocked by model governance"):
        colors = ("#7f1d1d", "#fecaca")
    else:
        colors = ("#78350f", "#fde68a")
    st.markdown(
        f'<span style="background:{colors[0]};color:{colors[1]};border-radius:6px;padding:2px 10px;'
        f'font-size:0.78rem;font-weight:600;">{text}</span>',
        unsafe_allow_html=True,
    )


def grader_caption(grader: Optional[dict]) -> str:
    if not grader:
        return "Grader: not available"
    num = grader.get("numeric_check") or {}
    num_txt = f"numeric check {num.get('matched', 0)}/{num.get('checked', 0)} matched"
    if num.get("mismatches"):
        num_txt += f" (unmatched: {', '.join(num['mismatches'][:5])})"
    if grader.get("verdict") == "UNGRADED" or grader.get("grader_error"):
        return (f"Grader unavailable ({(grader.get('grader_error') or '')[:120]}) \u00b7 verdict **UNGRADED** "
                f"\u00b7 {num_txt}")
    if grader.get("grader_model"):
        g = grader.get("groundedness")
        nc = grader.get("numeric_consistency")
        g_txt = f"{g:.2f}" if isinstance(g, (int, float)) else "n/a"
        nc_txt = f"{nc:.2f}" if isinstance(nc, (int, float)) else "n/a"
        who = config.provider_label(grader.get("grader_provider"), grader["grader_model"])
        return (f"Grader ({who}): groundedness {g_txt} \u00b7 numeric consistency {nc_txt} "
                f"\u00b7 verdict **{grader.get('verdict')}** \u00b7 {num_txt}")
    return f"Grader: LLM grader not run (demo output) \u00b7 {num_txt} \u00b7 verdict **{grader.get('verdict')}**"


def render_output(text: str, mode: str, reason: Optional[str], grader: Optional[dict], trace: Optional[dict],
                  fallback: Optional[str] = None) -> None:
    """Badge (provider + model that actually answered) + body + grader + trace expander."""
    from src.guardrails.bedrock_guardrails import filter_pii

    render_badge(mode, reason, trace if mode == "live" else None, fallback)
    st.markdown(md_money(filter_pii(text or "")))
    st.caption(md_money(grader_caption(grader)))
    if trace:
        with st.expander(f"Trace {trace.get('trace_id', '')} \u00b7 {trace.get('inventory_id', '')}"):
            st.json(trace)


def provider_status_lines() -> list:
    """Sidebar lines: which providers are configured right now."""
    return [
        f"Bedrock ({config.display_name(config.HAIKU_45)}): "
        + ("configured" if config.bedrock_available() else "not configured"),
        f"Gemini ({config.model_id('chat', 'gemini')}): "
        + ("configured" if config.gemini_available() else "not configured"),
        "Demo mode: always available",
    ]


# -- Cached heavy resources -------------------------------------------------------

@st.cache_resource(show_spinner=False)
def cached_reranker():
    from src.rag.reranker import CrossEncoderReranker

    return CrossEncoderReranker()


@st.cache_resource(show_spinner=False)
def cached_index(embedding_model: str, data_hash: str):
    from src.llm.provider import Embedder
    from src.rag.vector_store import get_index

    return get_index(Embedder(model_name=embedding_model))
