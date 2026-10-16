"""Provider chain (Bedrock Haiku 4.5 -> Gemini -> demo), grader independence, labels, per-provider approvals."""

import pytest

from src.llm import config
from src.llm.guard import GovernanceBlocked, run_guarded
from src.llm.router import EmbeddingUnavailable, ProviderChainFailed, grader_candidates, run_component
from src.mrm.model_inventory import current_version
from src.utils.data import load_portfolio
from tests.fakes import FakeChatModel


class ThrottlingException(Exception):
    """Mimics botocore ClientError ThrottlingException."""


def _approve(comps, providers=("bedrock", "gemini"), status="APPROVED"):
    out = {}
    for c in comps:
        out[c] = []
        for p in providers:
            cur = current_version(c, p)
            out[c].append({"status": status, "provider": p, "model_id": cur["model_id"],
                           "prompt_versions": cur["prompt_versions"], "evidence": f"test-{p}"})
    return out


@pytest.fixture
def both_providers(monkeypatch):
    """Pretend Bedrock and Gemini are configured; route model construction to fakes."""
    from src.llm import provider as prov_mod

    monkeypatch.setattr(config, "available_providers", lambda: ["bedrock", "gemini"])
    monkeypatch.setattr(config, "bedrock_available", lambda: True)
    monkeypatch.setattr(config, "gemini_available", lambda: True)
    calls = []
    behaviour = {}

    def fake_get_chat_model(role="chat", provider=None):
        calls.append((role, provider))
        errs = behaviour.get((role, provider))  # shared queue: each queued error fails one model instance
        return FakeChatModel(errors=[errs.pop(0)] if errs else [])

    monkeypatch.setattr(prov_mod, "get_chat_model", fake_get_chat_model)
    for mod in ("src.agents.base", "src.rag.rag_chain"):
        monkeypatch.setattr(f"{mod}.get_chat_model", fake_get_chat_model)
    return {"calls": calls, "behaviour": behaviour}


# -- fallback chain ---------------------------------------------------------------

def test_bedrock_primary_when_approved(both_providers):
    from src.agents.market_agent import run_market_analysis

    res = run_market_analysis(load_portfolio(), approvals=_approve(["MRM-002"]))
    assert res["provider"] == "bedrock" and res["fallback_reason"] is None
    assert res["trace"]["provider"] == "bedrock" and res["trace"]["answer_model"] == config.HAIKU_45
    # Haiku answered -> Gemini grader (a different model)
    assert res["grader"]["grader_provider"] == "gemini"
    assert res["grader"]["grader_model"] == config.model_id("grader", "gemini")


@pytest.mark.parametrize("error", [ThrottlingException("ThrottlingException: Rate exceeded"),
                                   RuntimeError("AccessDeniedException"), TimeoutError("read timeout")])
def test_bedrock_error_falls_back_to_gemini(both_providers, error):
    from src.agents.risk_agent import run_risk_assessment

    both_providers["behaviour"][("chat", "bedrock")] = [error, error]
    res = run_risk_assessment(load_portfolio(), approvals=_approve(["MRM-003"]))
    assert res["provider"] == "gemini"
    assert "Bedrock" in res["fallback_reason"] and type(error).__name__ in res["fallback_reason"]
    assert res["trace"]["fallback_reason"] == res["fallback_reason"]
    # Gemini answered -> Haiku grader first (different model)
    assert res["grader"]["grader_provider"] == "bedrock"


def test_missing_bedrock_credentials_uses_gemini(monkeypatch):
    from src.agents.market_agent import run_market_analysis
    from src.llm import provider as prov_mod

    monkeypatch.setattr(config, "gemini_available", lambda: True)
    monkeypatch.setattr(config, "available_providers", lambda: ["gemini"])
    monkeypatch.setattr("src.agents.base.get_chat_model", lambda role="chat", provider=None: FakeChatModel())
    monkeypatch.setattr(prov_mod, "get_chat_model", lambda role="chat", provider=None: FakeChatModel())
    res = run_market_analysis(load_portfolio(), approvals=_approve(["MRM-002"], providers=("gemini",)))
    assert res["provider"] == "gemini"
    assert res["grader"]["grader_model"] == config.model_id("grader", "gemini")  # Bedrock unavailable


def test_bedrock_not_approved_falls_back_with_reason(both_providers):
    from src.agents.portfolio_agent import run_portfolio_management

    res = run_portfolio_management(load_portfolio(), approvals=_approve(["MRM-004"], providers=("gemini",)))
    assert res["provider"] == "gemini"
    assert "not yet validated for Bedrock" in res["fallback_reason"]
    assert ("chat", "bedrock") not in both_providers["calls"]  # gate runs BEFORE any model call


def test_all_providers_fail_goes_to_demo(both_providers):
    from src.agents.market_agent import run_market_analysis

    both_providers["behaviour"][("chat", "bedrock")] = [RuntimeError("bedrock down")] * 2
    both_providers["behaviour"][("chat", "gemini")] = [RuntimeError("gemini down")] * 2
    with pytest.raises(ProviderChainFailed):
        run_market_analysis(load_portfolio(), approvals=_approve(["MRM-002"]))
    res = run_guarded(lambda: run_market_analysis(load_portfolio(), approvals=_approve(["MRM-002"])),
                      lambda reason: {"text": "demo", "sources": []}, credentials_present=True)
    assert res["mode"] == "demo" and res["reason"].startswith("All live providers failed")


def test_no_provider_approved_is_governance_block(both_providers):
    with pytest.raises(GovernanceBlocked, match="not yet validated for Bedrock"):
        run_component("MRM-002", lambda p: {"x": 1}, approvals={})


def test_orchestrator_sections_follow_the_chain(both_providers):
    from src.agents.orchestrator import run_full_analysis

    both_providers["behaviour"][("chat", "bedrock")] = [ThrottlingException("throttled")]
    approvals = _approve(["MRM-002", "MRM-003", "MRM-004", "MRM-005"])
    res = run_full_analysis(load_portfolio(), approvals=approvals)
    assert res["market"]["provider"] == "gemini" and "Throttl" in res["market"]["fallback_reason"]
    assert res["risk"]["provider"] == "bedrock"  # the next call on Bedrock succeeded
    assert res["trace"]["extra"]["section_providers"] == {"market": "gemini", "risk": "bedrock",
                                                          "portfolio": "bedrock"}


def test_orchestrator_skips_provider_without_mrm005_approval(both_providers):
    from src.agents.orchestrator import run_full_analysis

    approvals = _approve(["MRM-002", "MRM-003", "MRM-004"])
    approvals.update(_approve(["MRM-005"], providers=("gemini",)))
    res = run_full_analysis(load_portfolio(), approvals=approvals)
    assert {s["provider"] for s in (res["market"], res["risk"], res["portfolio"])} == {"gemini"}
    assert "MRM-005 is not yet validated for Bedrock" in res["trace"]["fallback_reason"]


def test_embedding_failure_goes_straight_to_demo(both_providers, tmp_path):
    from src.rag.rag_chain import answer_question
    from src.rag.vector_store import build_index
    from tests.fakes import FakeEmbedder

    vi = build_index(FakeEmbedder(), path=str(tmp_path / "idx"))

    class BrokenEmbedder(FakeEmbedder):
        def embed_passage(self, text):
            raise ConnectionError("embedding endpoint unreachable")

    def live():
        return answer_question("largest position?", embedder=BrokenEmbedder(), index=vi,
                               approvals=_approve(["MRM-001"]))

    with pytest.raises(EmbeddingUnavailable):
        live()
    res = run_guarded(live, lambda r: {"text": "demo", "sources": []}, credentials_present=True)
    assert res["mode"] == "demo" and "query embedding" in res["reason"]
    assert ("chat", "gemini") not in both_providers["calls"]  # no provider fallback for embedding failures


# -- grader independence ---------------------------------------------------------------

@pytest.mark.parametrize("bedrock,gemini", [(True, True), (True, False), (False, True), (False, False)])
def test_grader_candidates_never_include_answer_model(monkeypatch, bedrock, gemini):
    monkeypatch.setattr(config, "bedrock_available", lambda: bedrock)
    monkeypatch.setattr(config, "gemini_available", lambda: gemini)
    for prov in ("bedrock", "gemini"):
        answer = config.model_id("chat", prov)
        cands = grader_candidates(prov, answer)
        assert all(m != answer for _, m in cands)
        if prov == "bedrock":
            assert cands == ([("gemini", config.model_id("grader", "gemini"))] if gemini else [])


def test_self_grading_is_refused_and_marked_ungraded(monkeypatch):
    from src.mrm.grader import grade_output

    src = [{"source_id": "fund:summary", "text": "Total AUM: $32,355,808.19."}]
    g = grade_output("AUM $32.36M", src, llm=FakeChatModel(), answer_provider="gemini",
                     answer_model="gemini-3.5-flash-lite", grader_provider="gemini",
                     grader_model="gemini-3.5-flash-lite")
    assert g["verdict"] == "UNGRADED"
    # Gemini answered, Bedrock unavailable, grader model overridden to the answering model -> UNGRADED
    monkeypatch.setattr(config, "bedrock_available", lambda: False)
    monkeypatch.setattr(config, "gemini_available", lambda: True)
    monkeypatch.setenv("LLM_GRADER_MODEL", config.model_id("chat", "gemini"))
    g2 = grade_output("AUM $32.36M", src, answer_provider="gemini", answer_model=config.model_id("chat", "gemini"))
    assert g2["verdict"] == "UNGRADED" and "different" in g2["grader_error"]


# -- labels ----------------------------------------------------------------------------

def test_output_labels_name_the_answering_provider_and_model():
    from streamlit_app.live_ai import grader_caption, output_label

    assert output_label("live", trace={"provider": "bedrock", "answer_model": config.HAIKU_45}) == \
        "Live AI · Bedrock · Claude Haiku 4.5"
    lab = output_label("live", trace={"provider": "gemini", "answer_model": "gemini-3.5-flash-lite",
                                      "fallback_reason": "Bedrock failed: ThrottlingException"})
    assert lab == "Live AI · Gemini · gemini-3.5-flash-lite (fallback: Bedrock failed: ThrottlingException)"
    assert output_label("demo", "Session cap reached") == "Demo mode (Session cap reached)"
    cap = grader_caption({"verdict": "PASS", "grader_provider": "gemini", "grader_model": "gemini-3.1-flash-lite",
                          "groundedness": 1.0, "numeric_consistency": 1.0,
                          "numeric_check": {"checked": 1, "matched": 1, "mismatches": []}})
    assert cap.startswith("Grader (Gemini · gemini-3.1-flash-lite)")


def test_sidebar_provider_status_lines(monkeypatch):
    from streamlit_app.live_ai import provider_status_lines

    monkeypatch.setattr(config, "bedrock_available", lambda: True)
    lines = provider_status_lines()
    assert lines[0] == "Bedrock (Claude Haiku 4.5): configured"
    assert lines[1].endswith("not configured") and lines[2] == "Demo mode: always available"


# -- approvals per provider ---------------------------------------------------------------

def test_approvals_are_kept_per_provider_and_legacy_files_load(tmp_path):
    import json

    from src.mrm.approvals import load_approvals, write_approval
    from src.mrm.model_inventory import check_approval

    path = str(tmp_path / "a.json")
    json.dump({"approvals": {"MRM-002": {"status": "APPROVED", "model_id": "m", "prompt_versions": ["p"]}}},
              open(path, "w"))
    legacy = load_approvals(path)
    assert legacy["MRM-002"][0]["provider"] == "gemini"
    for prov in ("gemini", "bedrock"):
        cur = current_version("MRM-002", prov)
        write_approval("MRM-002", {"status": "APPROVED", "provider": prov, "model_id": cur["model_id"],
                                   "prompt_versions": cur["prompt_versions"], "evidence": prov}, path=path)
    recs = load_approvals(path)["MRM-002"]
    assert {r["provider"] for r in recs} == {"gemini", "bedrock"} and len(recs) == 3
    appr = load_approvals(path)
    for prov in ("gemini", "bedrock"):
        cur = current_version("MRM-002", prov)
        assert check_approval("MRM-002", cur["model_id"], cur["prompt_versions"], appr, provider=prov)["allowed"]
    # a Gemini approval does not approve the Bedrock version
    only_gemini = {"MRM-002": [r for r in recs if r["provider"] == "gemini"]}
    cur_b = current_version("MRM-002", "bedrock")
    assert not check_approval("MRM-002", cur_b["model_id"], cur_b["prompt_versions"], only_gemini,
                              provider="bedrock")["allowed"]


# -- Bedrock client wrapper / credentials / cost ------------------------------------------------

def test_bedrock_converse_request_has_token_cap_and_records_usage():
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.llm.provider import BedrockConverseChat, bedrock_cost_usd, usage_mark, usage_since

    class StubClient:
        def __init__(self):
            self.req = None

        def converse(self, **req):
            self.req = req
            return {"output": {"message": {"content": [{"text": "hello"}]}}, "stopReason": "end_turn",
                    "usage": {"inputTokens": 1000, "outputTokens": 200}}

    stub = StubClient()
    chat = BedrockConverseChat(config.HAIKU_45, 1024, 0.1, {}, client=stub)
    mark = usage_mark()
    msg = chat.invoke([SystemMessage(content="sys"), HumanMessage(content="hi")])
    assert msg.content == "hello" and msg.usage_metadata["input_tokens"] == 1000
    assert stub.req["inferenceConfig"]["maxTokens"] == 1024 and stub.req["system"] == [{"text": "sys"}]
    assert stub.req["modelId"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert usage_since(mark) == [{"provider": "bedrock", "model_id": config.HAIKU_45,
                                  "input_tokens": 1000, "output_tokens": 200}]
    assert bedrock_cost_usd(1000, 200) == pytest.approx(0.0011 + 0.0011)


def test_streamlit_uses_only_aws_secrets(monkeypatch):
    # conftest: inside the Streamlit runtime and no [aws] secrets -> Bedrock skipped
    assert config.bedrock_client_kwargs() is None and not config.bedrock_available()
    secrets = {("aws", "access_key_id"): "AKIA_TEST", ("aws", "secret_access_key"): "s", ("aws", "region"): "us-east-1"}
    monkeypatch.setattr(config, "_read_secret", lambda sec, key: secrets.get((sec, key)))
    kw = config.bedrock_client_kwargs()
    assert kw == {"aws_access_key_id": "AKIA_TEST", "aws_secret_access_key": "s", "region_name": "us-east-1"}


def test_every_role_has_an_output_cap():
    for prov in ("bedrock", "gemini"):
        for role in ("chat", "hyde", "grader"):
            assert 0 < config.max_output_tokens(role, prov) <= 2048


def test_dry_run_cost_estimate_and_split_rule():
    from src.eval.harness import estimate_cost

    est = estimate_cost(provider="bedrock")
    assert set(est["components"]) == {"MRM-001", "MRM-002", "MRM-003", "MRM-004", "MRM-005"}
    assert est["components"]["MRM-005"]["calls"] == 30 and est["components"]["MRM-001"]["calls"] == 46
    assert est["total_cost_usd_est"] == pytest.approx(sum(v["cost_usd_est"] for v in est["components"].values()),
                                                      abs=1e-5)
    assert est["max_single_component_usd_est"] <= 1.0


def test_fake_bedrock_eval_writes_provider_keyed_approval(tmp_path):
    from src.eval.dataset import build_dataset
    from src.eval.harness import run_eval
    from src.mrm.approvals import load_approvals
    from tests.fakes import fake_deps_factory

    ds = [i for i in build_dataset() if i["split"] == "heldout" and i["component"] == "MRM-003"][:2]
    run = run_eval("heldout", ["MRM-003"], dataset=ds, deps_factory=fake_deps_factory, fake=False,
                   provider="bedrock", log=lambda m: None, sleep=lambda s: None, max_cost_usd=1.0)
    assert run["provider"] == "bedrock" and run["run_id"].endswith("-bedrock")
    assert run["components"]["MRM-003"]["usage"]["cost_usd"] == 0.0  # fakes record no Bedrock usage
    recs = load_approvals()["MRM-003"]
    assert recs[0]["provider"] == "bedrock" and recs[0]["model_id"] == config.HAIKU_45


def test_bedrock_calls_are_non_streaming_on_the_us_profile(monkeypatch):
    """The app's IAM policy allows only bedrock:InvokeModel on the us.* Haiku 4.5 profile (no streaming)."""
    from src.llm import provider as prov_mod

    class StrictClient:
        def __init__(self):
            self.calls = []

        def converse(self, **req):
            self.calls.append(req)
            return {"output": {"message": {"content": [{"text": "ok"}]}}, "usage": {"inputTokens": 1, "outputTokens": 1}}

        def converse_stream(self, **req):  # pragma: no cover - must never be called
            raise AssertionError("ConverseStream is denied by the IAM policy")

        def invoke_model_with_response_stream(self, **req):  # pragma: no cover
            raise AssertionError("InvokeModelWithResponseStream is denied by the IAM policy")

    client = StrictClient()
    monkeypatch.setattr(config, "bedrock_client_kwargs", lambda: {"region_name": "us-east-1"})
    monkeypatch.setattr(prov_mod, "_bedrock_client", lambda kwargs: client)
    for role in ("chat", "hyde", "grader"):
        m = prov_mod.get_chat_model(role, "bedrock")
        assert not hasattr(m, "stream")
        m.invoke("hi")
    assert [c["modelId"] for c in client.calls] == ["us.anthropic.claude-haiku-4-5-20251001-v1:0"] * 3
    assert all(c["modelId"].startswith("us.") and not c["modelId"].startswith("global.") for c in client.calls)
    assert [c["inferenceConfig"]["maxTokens"] for c in client.calls] == [1024, 256, 512]
