import json
import os

import pytest

from src.llm.guard import GovernanceBlocked, call_llm, is_rate_limit_error, run_guarded
from src.mrm.grader import contains_value, deterministic_grade, grade_output, numeric_check, parse_grader_json
from tests.fakes import FakeChatModel, FakeCrossEncoder, FakeEmbedder, RateLimitError


# -- grader -----------------------------------------------------------------------

def test_parse_grader_json_variants():
    ok = parse_grader_json('```json\n{"groundedness": 0.9, "numeric_consistency": 1, '
                           '"unsupported_claims": [], "verdict": "pass"}\n```')
    assert ok["verdict"] == "PASS" and ok["groundedness"] == 0.9 and not ok["parse_error"]
    embedded = parse_grader_json('Here you go: {"groundedness": 2, "numeric_consistency": 0.5, "verdict": "x"} thanks')
    assert embedded["groundedness"] == 1.0 and embedded["verdict"] == "FLAG"
    bad = parse_grader_json("not json at all")
    assert bad["verdict"] == "FLAG" and bad["parse_error"] and bad["groundedness"] is None


def test_numeric_check_rounding_scales_and_mismatch():
    src = ["Total AUM: $32,355,808.19. Weight: 47.03%. Beta: 1.298. Revenue: $118.8B."]
    good = "AUM is $32.36M ($32,355,808), tech 47.0% of book, beta 1.30, revenue $118.8B."
    r = numeric_check(good, src)
    assert r["mismatches"] == [] and r["checked"] >= 5
    bad = numeric_check("AUM is $35.1M and tech is 52.3%.", src)
    assert set(bad["mismatches"]) == {"$35.1M", "52.3%"}
    # identifiers and plain integers are not treated as metrics
    assert numeric_check("See MRM-001, SMA-50 and Q1 2024 for 15 positions.", src)["checked"] == 0
    assert contains_value("The fund holds 15 positions.", 15)
    assert contains_value("VaR is $1.17M", 1172331.66)


def test_grade_output_combines_llm_and_numeric_check():
    sources = [{"source_id": "fund:summary", "text": "Total AUM: $32,355,808.19."}]
    llm = FakeChatModel()
    g = grade_output("AUM is $32.36M [fund:summary].", sources, llm=llm)
    assert g["verdict"] == "PASS" and g["llm_verdict"] == "PASS"
    assert "independent model validator" in llm.calls[0]
    g2 = grade_output("AUM is $99M.", sources, llm=FakeChatModel())
    assert g2["verdict"] == "FLAG" and g2["numeric_check"]["mismatches"] == ["$99M"]


# -- retrieval --------------------------------------------------------------------

@pytest.fixture(scope="module")
def fake_index(tmp_path_factory):
    from src.rag.vector_store import build_index

    emb = FakeEmbedder()
    return emb, build_index(emb, path=str(tmp_path_factory.mktemp("idx")))


def test_index_manifest_and_verification(fake_index, tmp_path):
    from src.rag.vector_store import IndexUnavailable, build_index, load_index

    emb, vi = fake_index
    m = vi.manifest
    assert m["embedding_model"] == "fake-embedding" and m["chunk_size"] == 1000 and m["chunk_overlap"] == 200
    assert m["n_chunks"] == len(vi.docs) and m["dim"] == 64 and len(m["data_sha256"]) == 64
    path = str(tmp_path / "idx")
    build_index(FakeEmbedder(), path=path)
    assert load_index("fake-embedding", path).manifest["n_chunks"] == m["n_chunks"]
    with pytest.raises(IndexUnavailable, match="embedding model"):
        load_index("gemini-embedding-001", path)
    mf = os.path.join(path, "manifest.json")
    data = json.load(open(mf))
    data["data_sha256"] = "0" * 64
    json.dump(data, open(mf, "w"))
    with pytest.raises(IndexUnavailable, match="data_sha256"):
        load_index("fake-embedding", path)


def test_hyde_searches_with_the_passage_embedding(fake_index):
    from src.rag.query_transform import hyde_retrieve

    _, vi = fake_index
    emb = FakeEmbedder()
    llm = FakeChatModel(responder=lambda t: "Position UNH (Healthcare). Market value: $6,494,098.40.")
    res = hyde_retrieve("biggest holding?", vi, emb, llm=llm, merge_raw_query=False)
    assert res["passage"].startswith("Position UNH")
    assert emb.calls[0] == ("passage", res["passage"])  # the passage is embedded, not the question
    assert len(res["candidates"]) == 12
    assert all(c["retrieved_by"] == "hyde" for c in res["candidates"])
    assert "hedge fund's" in llm.calls[0]  # HyDE prompt was used
    top3 = [c["source_id"] for c in res["candidates"][:3]]
    assert "position:UNH" in top3


def test_hyde_merge_adds_raw_query_hits(fake_index):
    from src.rag.query_transform import hyde_retrieve

    _, vi = fake_index
    emb = FakeEmbedder()
    res = hyde_retrieve("macro VIX", vi, emb, llm=FakeChatModel(), merge_raw_query=True)
    kinds = [c[0] for c in emb.calls]
    assert kinds == ["passage", "query"]
    ids = [c["source_id"] for c in res["candidates"]]
    assert len(ids) == len(set(ids)) >= 12


def test_reranker_orders_by_score():
    from src.rag.reranker import CrossEncoderReranker

    rr = CrossEncoderReranker(encoder=FakeCrossEncoder())
    docs = [{"text": "macro VIX reading", "source_id": "a"}, {"text": "UNH market value largest position", "source_id": "b"},
            {"text": "largest position", "source_id": "c"}]
    ranked = rr.rerank("largest position market value", docs, top_n=2)
    assert [d["source_id"] for d, _ in ranked] == ["b", "c"]
    assert ranked[0][1] > ranked[1][1]


def test_real_cross_encoder_orders_relevant_first():
    from src.rag.reranker import CrossEncoderReranker

    try:
        rr = CrossEncoderReranker()
        ranked = rr.rerank("What is UNH's market value?",
                           [{"text": "Macro indicators for 2026-02: VIX 22.1", "source_id": "m"},
                            {"text": "Position UNH (Healthcare). Market value: $6,494,098.40.", "source_id": "u"}], top_n=2)
    except Exception as exc:  # model not downloadable offline
        pytest.skip(f"cross-encoder model unavailable offline: {type(exc).__name__}")
    assert ranked[0][0]["source_id"] == "u"


def test_rag_chain_end_to_end_with_fakes(fake_index):
    from src.mrm.model_inventory import current_version
    from src.rag.rag_chain import answer_question
    from src.rag.reranker import CrossEncoderReranker

    emb, vi = fake_index
    cur = current_version("MRM-001")
    approvals = {"MRM-001": {"status": "APPROVED", "model_id": cur["model_id"],
                             "prompt_versions": cur["prompt_versions"], "evidence": "t"}}
    chat = FakeChatModel()
    res = answer_question("What is the market value of UNH?", [{"role": "user", "content": "hi"}],
                          chat_llm=chat, hyde_llm=FakeChatModel(), grader_llm=FakeChatModel(), embedder=emb,
                          index=vi, reranker=CrossEncoderReranker(encoder=FakeCrossEncoder()), approvals=approvals)
    assert len(res["sources"]) == 4
    scores = [s["rerank_score"] for s in res["sources"]]
    assert scores == sorted(scores, reverse=True)
    t = res["trace"]
    for key in ("trace_id", "inventory_id", "provider", "model_id", "prompt_versions", "embedding_model",
                "index_manifest_hash", "source_ids", "grader", "approval_status", "timestamp"):
        assert t.get(key) not in (None, []), key
    assert t["inventory_id"] == "MRM-001" and t["approval_status"] == "APPROVED"
    assert "user: hi" in chat.calls[0]  # conversational memory in prompt
    with pytest.raises(GovernanceBlocked):
        answer_question("x", chat_llm=chat, embedder=emb, index=vi, approvals={})


def test_orchestrator_graph_with_fakes_and_blocked_agent():
    from src.agents.orchestrator import run_full_analysis
    from src.mrm.model_inventory import current_version
    from src.utils.data import load_portfolio

    approvals = {}
    for c in ("MRM-002", "MRM-004", "MRM-005"):  # MRM-003 deliberately not approved
        cur = current_version(c)
        approvals[c] = {"status": "APPROVED", "model_id": cur["model_id"], "prompt_versions": cur["prompt_versions"]}
    res = run_full_analysis(load_portfolio(), llm=FakeChatModel(), grader_llm=FakeChatModel(), approvals=approvals)
    assert res["market"]["mode"] == "live"
    assert res["risk"]["mode"] == "blocked" and "not yet validated" in res["risk"]["reason"]
    assert res["portfolio"]["mode"] == "live"
    assert res["trace"]["inventory_id"] == "MRM-005"
    assert res["action_items"] and res["technical_alerts"] is not None
    with pytest.raises(GovernanceBlocked):
        run_full_analysis(load_portfolio(), llm=FakeChatModel(), approvals={})


# -- guard / session cap ----------------------------------------------------------

def test_missing_key_falls_back_to_demo():
    calls = []
    res = run_guarded(lambda: calls.append(1) or {"x": 1}, lambda reason: {"text": reason})
    assert res["mode"] == "demo" and "No Gemini API key" in res["reason"] and not calls


def test_429_falls_back_to_demo():
    def live():
        raise RateLimitError()

    res = run_guarded(live, lambda r: {"text": "demo"}, credentials_present=True)
    assert res["mode"] == "demo" and "429" in res["reason"]
    assert is_rate_limit_error(RuntimeError("wrapped")) is False
    try:
        try:
            raise RateLimitError()
        except RateLimitError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as outer:
        assert is_rate_limit_error(outer)


def test_call_llm_retries_429_once():
    sleeps = []
    llm = FakeChatModel(responder=lambda t: "ok", errors=[RateLimitError()])
    assert call_llm(llm, "hi", sleep=sleeps.append).content == "ok"
    assert len(llm.calls) == 2 and len(sleeps) == 1
    llm2 = FakeChatModel(errors=[RateLimitError(), RateLimitError()])
    with pytest.raises(RateLimitError):
        call_llm(llm2, "hi", sleep=lambda s: None)
    assert len(llm2.calls) == 2


def test_call_llm_retries_transient_5xx_then_raises():
    class Unavailable(Exception):
        code = 503

    sleeps = []
    llm = FakeChatModel(responder=lambda t: "ok", errors=[Unavailable("503 UNAVAILABLE high demand")])
    assert call_llm(llm, "hi", sleep=sleeps.append).content == "ok"
    assert sleeps == [10.0]
    llm2 = FakeChatModel(errors=[Unavailable("503 UNAVAILABLE")] * 3)
    with pytest.raises(Unavailable):
        call_llm(llm2, "hi", sleep=lambda s: None)
    assert len(llm2.calls) == 3


def test_governance_block_and_generic_error():
    def blocked():
        raise GovernanceBlocked("MRM-001 is not yet validated")

    res = run_guarded(blocked, lambda r: {"text": "d"}, credentials_present=True)
    assert res["blocked"] and res["reason"].startswith("Blocked by model governance:")
    res = run_guarded(lambda: 1 / 0, lambda r: {"text": "d"}, credentials_present=True)
    assert res["mode"] == "demo" and "ZeroDivisionError" in res["reason"]


def test_session_cap_eleventh_request_falls_back():
    from streamlit_app.live_ai import MAX_LIVE_REQUESTS_PER_SESSION, run_ai_request

    state = {}
    live_calls = []
    for i in range(MAX_LIVE_REQUESTS_PER_SESSION):
        r = run_ai_request("MRM-001", lambda: live_calls.append(1) or {"text": "live"},
                           lambda reason: {"text": "demo", "sources": []}, state=state, credentials_present=True)
        assert r["mode"] == "live"
    r11 = run_ai_request("MRM-001", lambda: live_calls.append(1) or {"text": "live"},
                         lambda reason: {"text": "demo", "sources": []}, state=state, credentials_present=True)
    assert r11["mode"] == "demo" and "Session cap" in r11["reason"]
    assert len(live_calls) == 10 and state["live_requests_used"] == 10
    assert r11["output"]["trace"]["mode"] == "demo"


def test_blocked_requests_do_not_consume_cap():
    from streamlit_app.live_ai import run_ai_request

    state = {}

    def blocked():
        raise GovernanceBlocked("not yet validated")

    r = run_ai_request("MRM-001", blocked, lambda reason: {"text": "demo", "sources": []}, state=state,
                       credentials_present=True)
    assert r["blocked"] and state["live_requests_used"] == 0
    assert r["output"]["trace"]["mode"] == "blocked"


# -- demo outputs are data-derived --------------------------------------------------

@pytest.mark.parametrize("question", [
    "What are my highest risk positions?", "What is the portfolio's technology weight?",
    "Should I be worried about TSLA?", "What does my CVaR tell me?", "Run the stress tests",
    "What is the Fed doing?", "Any earnings misses?", "hello",
])
def test_demo_chat_numbers_all_come_from_sources(question):
    from src.demo.templates import demo_chat_answer

    d = demo_chat_answer(question)
    g = deterministic_grade(d["text"], d["sources"])
    assert g["numeric_check"]["mismatches"] == [], d["text"]


def test_demo_agent_reports_numbers_come_from_sources():
    from src.demo.templates import demo_market_report, demo_portfolio_recommendations, demo_risk_report

    for fn in (demo_market_report, demo_risk_report, demo_portfolio_recommendations):
        d = fn()
        assert deterministic_grade(d["text"], d["sources"])["numeric_check"]["mismatches"] == [], fn.__name__
    txt = demo_chat_answer_default()
    assert "15 positions" in txt and "$32,355,808.19" in txt
    assert "5.42M" not in txt and "43.5%" not in txt


def demo_chat_answer_default():
    from src.demo.templates import demo_chat_answer

    return demo_chat_answer("hello")["text"]


# -- formatting / chunking -----------------------------------------------------------

def test_md_money_and_escape():
    from streamlit_app.formatting import md_escape, md_money, money

    assert md_money("VaR $829,099 vs CVaR $1.19M") == "VaR \\$829,099 vs CVaR \\$1.19M"
    assert md_money(md_money("$5")) == "\\$5"
    assert md_money(None) == ""
    assert md_escape("*bold* $x [link](y)") == "\\*bold\\* \\$x \\[link\\]\\(y\\)"
    assert money(-1234.5) == "-\\$1,234"


def test_chunking_counts_and_overlap():
    from src.rag.chunking import count_tokens, split_text

    # Beta : 1 . 77 , P & L - 31 . 65 % .
    assert count_tokens("Beta: 1.77, P&L -31.65%.") == 15
    assert split_text("short record.") == ["short record."]
    long = " ".join(f"w{i}" for i in range(2500))
    chunks = split_text(long, 1000, 200)
    assert [count_tokens(c) for c in chunks] == [1000, 1000, 900]
    assert chunks[1].split()[0] == "w800"


# -- eval harness (fake) ------------------------------------------------------------

def test_eval_dataset_is_computed_and_split():
    from collections import Counter

    from src.eval.dataset import build_dataset, load_dataset

    items = build_dataset()
    rag = [i for i in items if i["component"] == "MRM-001"]
    assert len(rag) >= 40
    c = Counter(i["split"] for i in rag)
    assert abs(c["heldout"] - c["dev"]) <= 1
    assert build_dataset() == items  # deterministic
    assert load_dataset() == items  # committed file matches the generator


def test_fake_eval_run_writes_results_but_not_approvals(tmp_path):
    from src.eval.dataset import build_dataset
    from src.eval.harness import run_eval
    from src.mrm.approvals import approvals_path
    from tests.fakes import fake_deps_factory

    ds = [i for i in build_dataset() if i["split"] == "heldout"]
    ds = [i for i in ds if i["component"] == "MRM-001"][:3] + [i for i in ds if i["component"] == "MRM-003"][:1]
    run = run_eval("heldout", ["MRM-001", "MRM-003", "MRM-006"], dataset=ds,
                   deps_factory=lambda comps: fake_deps_factory(comps, str(tmp_path / "idx")),
                   fake=True, log=lambda m: None)
    assert run["fake"] and os.path.exists(run["_path"])
    assert run["components"]["MRM-001"]["n_items"] == 3
    assert run["components"]["MRM-006"]["status"] in ("APPROVED", "CONDITIONAL", "FAILED")
    assert not os.path.exists(approvals_path())


def test_eval_status_thresholds_and_429_stop(tmp_path):
    from src.eval.dataset import build_dataset
    from src.eval.harness import decide_status, run_eval
    from tests.fakes import fake_deps_factory

    assert decide_status(0.90, 0.85) == "APPROVED"
    assert decide_status(0.90, 0.70) == "CONDITIONAL"
    assert decide_status(0.75, 0.95) == "CONDITIONAL"
    assert decide_status(0.60, 0.95) == "FAILED"

    ds = [i for i in build_dataset() if i["split"] == "heldout" and i["component"] == "MRM-002"][:2]

    def factory(comps):
        d = fake_deps_factory(comps)
        d["llm"] = FakeChatModel(errors=[RateLimitError(), RateLimitError()])
        return d

    run = run_eval("heldout", ["MRM-002"], dataset=ds, deps_factory=factory, fake=False,
                   log=lambda m: None, sleep=lambda s: None)
    assert run["complete"] is False and "429" in run["stopped_reason"]
    from src.mrm.approvals import load_approvals
    assert "MRM-002" not in load_approvals()
