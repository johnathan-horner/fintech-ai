"""Per-provider prompt selection (Claude Haiku 4.5 variants) and Gemini prompt stability."""

import pytest

from src.mrm.model_inventory import current_version
from src.prompts import registry
from src.prompts.templates import component_prompts, prompt_name_for
from src.utils.data import load_portfolio
from tests.fakes import FakeChatModel

# Gemini-approved versions (data/mrm/approvals.json) - these must never change.
GEMINI_APPROVED = {
    "MRM-001": ["hyde_passage@1.0.0#abb1a9d5", "rag_answer@1.0.0#b6aacf6d"],
    "MRM-002": ["market_analysis@1.2.0#f046466c"],
    "MRM-003": ["risk_report@1.2.0#7ca70726"],
    "MRM-004": ["portfolio_recommendations@1.2.0#c5fa95b1"],
    "MRM-005": ["market_analysis@1.2.0#f046466c", "portfolio_recommendations@1.2.0#c5fa95b1",
                "risk_report@1.2.0#7ca70726"],
}


@pytest.mark.parametrize("comp", sorted(GEMINI_APPROVED))
def test_gemini_prompt_hashes_unchanged(comp):
    assert current_version(comp, "gemini")["prompt_versions"] == GEMINI_APPROVED[comp]
    assert registry.get("grader").ref == "grader@1.0.0#13d51998"


def test_bedrock_selects_haiku_variants():
    assert prompt_name_for("rag_answer", "bedrock") == "rag_answer_haiku"
    assert prompt_name_for("market_analysis", "bedrock") == "market_analysis_haiku"
    assert prompt_name_for("portfolio_recommendations", "bedrock") == "portfolio_recommendations_haiku"
    assert prompt_name_for("risk_report", "bedrock") == "risk_report"  # shared: MRM-003 approved on Haiku with it
    assert prompt_name_for("hyde_passage", "bedrock") == "hyde_passage"
    for base in ("rag_answer", "market_analysis", "portfolio_recommendations", "risk_report"):
        assert prompt_name_for(base, "gemini") == base
    assert component_prompts("MRM-005", "bedrock") == ["market_analysis_haiku", "risk_report",
                                                       "portfolio_recommendations_haiku"]
    b = current_version("MRM-002", "bedrock")["prompt_versions"]
    assert len(b) == 1 and b[0].startswith("market_analysis_haiku@1.0.0#")


def test_haiku_prompts_carry_the_strict_rules():
    for name in ("rag_answer_haiku", "market_analysis_haiku", "portfolio_recommendations_haiku"):
        t = registry.get(name).template
        for phrase in ("appear verbatim", "Never relabel a period", "Do not turn beta",
                       "include every matching item", "do not round".capitalize().split()[0]):
            assert phrase in t, (name, phrase)


def test_gemini_agent_inputs_unchanged_and_haiku_inputs_labelled():
    from src.agents.context import market_sources, portfolio_sources

    pf = load_portfolio()
    g = market_sources(pf)
    assert g == market_sources(pf, "gemini")
    assert not any("not year-to-date" in s["text"] for s in g)
    h = market_sources(pf, "bedrock")
    aapl = [s for s in h if s["source_id"] == "market:AAPL"][0]["text"]
    assert "Price change over the market-data window 2025-12-19 to 2026-03-18 (not year-to-date): -4.39% (down)" in aapl
    ev = [s for s in h if s["source_id"] == "derived:earnings_vs_estimates"][0]["text"]
    assert "revenue $100.2B was above the $94.0B estimate by $6.2B" in ev
    mc = [s for s in h if s["source_id"] == "derived:macro_changes"][0]["text"]
    assert "Federal Funds Rate increased from 4.63 (2026-01) to 5.22 (2026-02), a change of 0.59 percentage points" in mc
    pm = [s for s in portfolio_sources(pf, "bedrock") if s["source_id"] == "position:MA"][0]["text"]
    assert "Unrealized P&L since average cost (not a period return): -$587,687.32 (-31.65% return since average cost; loss)" in pm


def test_haiku_rag_companions_and_direction_words():
    from src.agents.haiku_context import augment_rag_sources

    srcs = [{"source_id": "macro:2025-08", "text": "Macro indicators for 2025-08 - ..."},
            {"source_id": "position:MA", "text": "Position MA. Unrealized P&L: -$587,687.32 (-31.65%)."}]
    out = augment_rag_sources(srcs)
    ids = [s["source_id"] for s in out]
    assert ids[:2] == ["macro:2025-08", "position:MA"]
    assert "derived:macro_changes:2025-08" in ids and "derived:portfolio_lists" in ids
    mc = [s for s in out if s["source_id"] == "derived:macro_changes:2025-08"][0]["text"]
    assert "Consumer Confidence Index" in mc and ("increased from" in mc or "decreased from" in mc)
    lists = [s for s in out if s["source_id"] == "derived:portfolio_lists"][0]["text"]
    for t in ("CVX", "MA", "XOM", "UNH", "JNJ"):
        assert t in lists


def test_agent_uses_provider_prompt_and_context(monkeypatch):
    from src.agents.market_agent import run_market_analysis

    llm = FakeChatModel()
    res = run_market_analysis(load_portfolio(), llm=llm, grader_llm=FakeChatModel(), provider="bedrock",
                              grader_provider="gemini", grader_model="gemini-3.1-flash-lite", enforce_approval=False)
    assert res["trace"]["prompt_versions"][0].startswith("market_analysis_haiku@1.0.0#")
    assert "Never relabel a period" in llm.calls[0] and "(not year-to-date)" in llm.calls[0]
    res_g = run_market_analysis(load_portfolio(), llm=FakeChatModel(), grader_llm=FakeChatModel(), provider="gemini",
                                enforce_approval=False)
    assert res_g["trace"]["prompt_versions"] == ["market_analysis@1.2.0#f046466c"]


def test_rag_uses_haiku_prompt_and_companions_on_bedrock(tmp_path):
    from src.rag.rag_chain import answer_question
    from src.rag.reranker import CrossEncoderReranker
    from src.rag.vector_store import build_index
    from tests.fakes import FakeCrossEncoder, FakeEmbedder

    emb = FakeEmbedder()
    vi = build_index(emb, path=str(tmp_path / "idx"))
    chat = FakeChatModel()
    res = answer_question("Which positions have unrealized losses?", chat_llm=chat, hyde_llm=FakeChatModel(),
                          grader_llm=FakeChatModel(), embedder=emb, index=vi, provider="bedrock",
                          reranker=CrossEncoderReranker(encoder=FakeCrossEncoder()), enforce_approval=False)
    assert "Never relabel a period" in chat.calls[0]
    assert "rag_answer_haiku@1.0.0" in " ".join(res["trace"]["prompt_versions"])
    assert len([s for s in res["sources"] if s.get("retrieved_by") != "derived"]) == 4
