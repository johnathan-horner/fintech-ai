"""Tests for round 2: precomputed agent inputs, 3-agent orchestrator, grader 503 fallback, eval-set v2."""

import pytest

from src.utils import metrics
from src.utils.data import load_macro, load_portfolio
from tests.fakes import FakeChatModel, RateLimitError


# -- derived values ---------------------------------------------------------------

def test_macro_derived_spread_and_regime():
    from src.utils.macro import macro_derived, market_regime

    d = macro_derived(load_macro())
    last = sorted(load_macro(), key=lambda m: m["date"])[-1]
    assert d["spread_10y_2y_pct"] == round(last["10Y Treasury Yield"] - last["2Y Treasury Yield"], 2)
    assert market_regime(load_macro())["regime"] in ("Risk-On", "Risk-Off", "Transitional")


def test_concentration_excess_and_volume():
    c = metrics.concentration_excess(load_portfolio())
    tech = [x for x in c["sector_excess"] if x["sector"] == "Technology"][0]
    assert tech["excess_pct_points"] == round(tech["weight_pct"] - metrics.SECTOR_LIMIT_PCT, 2)
    from src.utils.data import load_market_data
    v = metrics.volume_stats(load_market_data(), ["AAPL"])["AAPL"]
    assert v["volume_ratio_vs_20d"] == round(v["latest_volume"] / v["avg_volume_20d"], 2) or v["avg_volume_20d"]


def test_agent_inputs_contain_the_derived_values():
    from src.agents.context import market_sources, risk_sources

    src = market_sources(load_portfolio())
    ids = [s["source_id"] for s in src]
    assert "derived:macro" in ids and "derived:volume" in ids and "derived:earnings_vs_estimates" in ids
    macro = [s for s in src if s["source_id"] == "derived:macro"][0]["text"]
    assert "10Y-2Y spread -1.14" in macro  # the number MRM-005 previously computed by itself
    ids = [s["source_id"] for s in risk_sources(load_portfolio())]
    assert "derived:history" in ids and "derived:concentration" in ids


# -- bias / directions -------------------------------------------------------------

def _approve(*comps):
    from src.mrm.model_inventory import current_version

    out = {}
    for c in comps:
        cur = current_version(c)
        out[c] = {"status": "APPROVED", "model_id": cur["model_id"], "prompt_versions": cur["prompt_versions"]}
    return out


def test_orchestrator_three_sections_and_prompt_versions():
    from src.agents.orchestrator import SECTION_KEYS, run_full_analysis
    from src.mrm.model_inventory import current_version

    approvals = _approve("MRM-002", "MRM-003", "MRM-004", "MRM-005")
    res = run_full_analysis(load_portfolio(), llm=FakeChatModel(), grader_llm=FakeChatModel(), approvals=approvals)
    assert SECTION_KEYS == ("market", "risk", "portfolio")
    assert all(res[k]["mode"] == "live" for k in SECTION_KEYS)
    pv = current_version("MRM-005")["prompt_versions"]
    assert [p.split("#")[0] for p in pv] == ["market_analysis@1.2.0", "portfolio_recommendations@1.2.0",
                                              "risk_report@1.2.0"]


def test_agent_prompts_forbid_arithmetic():
    from src.prompts import registry

    for name in ("market_analysis", "risk_report", "portfolio_recommendations"):
        t = registry.get(name).template
        assert "Never add, subtract, multiply, divide" in t and "{upstream}" not in t and "{market_summary}" not in t


def test_grader_503_degrades_to_ungraded_without_dropping_output():
    from src.mrm.grader import grade_output

    class Unavailable(Exception):
        code = 503

    g = grade_output("AUM $32.36M", [{"source_id": "fund:summary", "text": "Total AUM: $32,355,808.19."}],
                     llm=FakeChatModel(errors=[Unavailable("503 UNAVAILABLE high demand")] * 3))
    assert g["verdict"] == "UNGRADED" and "503" in g["grader_error"]
    assert g["numeric_check"]["mismatches"] == []


def test_retry_delay_from_429_message():
    from src.llm.guard import call_with_retry, suggested_retry_delay

    assert suggested_retry_delay(Exception("Please retry in 12.2s.")) == pytest.approx(13.2)
    sleeps = []
    calls = []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise RateLimitError("429 RESOURCE_EXHAUSTED Please retry in 5s.")
        return "ok"

    assert call_with_retry(fn, sleep=sleeps.append) == "ok" and sleeps == [6.0]


# -- live data (mocked, no network) -------------------------------------------------

def test_eval_set_v2_sizes():
    from collections import Counter

    from src.eval.dataset import build_dataset

    c = Counter(i["component"] for i in build_dataset() if i["split"] == "heldout")
    for comp in ("MRM-001", "MRM-002", "MRM-003", "MRM-004", "MRM-005"):
        assert c[comp] >= 10, comp
    assert set(c) == {"MRM-001", "MRM-002", "MRM-003", "MRM-004", "MRM-005"}




def test_ungraded_caption_is_labelled():
    from streamlit_app.live_ai import grader_caption

    cap = grader_caption({"verdict": "UNGRADED", "grader_model": "m", "grader_error": "GoogleAPIError: 503",
                          "numeric_check": {"checked": 1, "matched": 1, "mismatches": []}})
    assert "Grader unavailable" in cap and "UNGRADED" in cap
