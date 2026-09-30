"""
FinTech AI - held-out eval harness.

Runs each LLM component on a split of data/eval/fintech_eval.jsonl, grades
every output with the independent grader + deterministic numeric check +
exact-fact check, writes data/eval/results/<run_id>.json and (for real runs)
updates data/mrm/approvals.json for the exact model_id + prompt versions that
were evaluated:

    APPROVED     pass_rate >= 0.85 and mean groundedness >= 0.80
    CONDITIONAL  pass_rate >= 0.70
    FAILED       otherwise

MRM-006 is approved from the out-of-sample VaR backtest
(GREEN -> APPROVED, YELLOW -> CONDITIONAL, RED -> FAILED).

Fake runs (tests / --fake) are marked ``"fake": true`` and never write approvals.
Runs interrupted by repeated HTTP 429 are marked incomplete and never write approvals;
a component with any errored item (e.g. persistent 503) does not get an approval written.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from src.eval.dataset import load_dataset, variant_portfolio
from src.eval.results import results_dir
from src.llm import config
from src.llm.guard import is_rate_limit_error, sanitize_error
from src.mrm.approvals import write_approval
from src.mrm.grader import GROUNDEDNESS_PASS, contains_value
from src.mrm.model_inventory import current_version

APPROVE_PASS_RATE = 0.85
APPROVE_GROUNDEDNESS = 0.80
CONDITIONAL_PASS_RATE = 0.70
LLM_COMPONENTS = ("MRM-001", "MRM-002", "MRM-003", "MRM-004", "MRM-005")


class RateLimitStop(RuntimeError):
    """Raised when the provider keeps returning 429 - the run stops."""


def decide_status(pass_rate: float, mean_groundedness: float) -> str:
    if pass_rate >= APPROVE_PASS_RATE and mean_groundedness >= APPROVE_GROUNDEDNESS:
        return "APPROVED"
    if pass_rate >= CONDITIONAL_PASS_RATE:
        return "CONDITIONAL"
    return "FAILED"


def backtest_status(zone: str) -> str:
    return {"GREEN": "APPROVED", "YELLOW": "CONDITIONAL"}.get(zone, "FAILED")


def exact_fact_check(output: str, item: dict) -> dict:
    missing_values = [v for v in item.get("required_values", []) if not contains_value(output, v["value"])]
    missing_terms = [t for t in item.get("required_terms", [])
                     if not re.search(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", output, flags=re.IGNORECASE)]
    return {"ok": not missing_values and not missing_terms,
            "missing_values": missing_values, "missing_terms": missing_terms}


def _grader_ok(g: Optional[dict]) -> bool:
    return bool(g) and g.get("verdict") == "PASS" and (g.get("groundedness") or 0) >= GROUNDEDNESS_PASS


def run_item(item: dict, deps: dict) -> dict:
    """Run one eval item. ``deps`` carries llm / grader_llm / hyde_llm / embedder / index / reranker."""
    from src.agents.market_agent import run_market_analysis
    from src.agents.orchestrator import run_full_analysis
    from src.agents.portfolio_agent import run_portfolio_management
    from src.agents.risk_agent import run_risk_assessment
    from src.rag.rag_chain import answer_question

    comp = item["component"]
    started = time.perf_counter()
    kw = dict(grader_llm=deps.get("grader_llm"), enforce_approval=False)
    if comp == "MRM-001":
        res = answer_question(item["question"], chat_llm=deps.get("llm"), hyde_llm=deps.get("hyde_llm"),
                              embedder=deps["embedder"], index=deps["index"], reranker=deps.get("reranker"), **kw)
        outputs = [res["answer"]]
        graders = [res["grader"]]
        extra = {"sources": [{"source_id": s["source_id"], "rerank_score": s["rerank_score"]} for s in res["sources"]],
                 "hypothetical_passage": res["hypothetical_passage"], "trace_id": res["trace"]["trace_id"]}
    else:
        pf = variant_portfolio(item["variant"])
        if comp == "MRM-002":
            res = run_market_analysis(pf, llm=deps.get("llm"), **kw)
            outputs, graders = [res["market_analysis"]], [res["grader"]]
        elif comp == "MRM-003":
            res = run_risk_assessment(pf, "", llm=deps.get("llm"), **kw)
            outputs, graders = [res["risk_report"]], [res["grader"]]
        elif comp == "MRM-004":
            res = run_portfolio_management(pf, "", "", llm=deps.get("llm"), **kw)
            outputs, graders = [res["portfolio_recommendations"]], [res["grader"]]
        elif comp == "MRM-005":
            from src.agents.orchestrator import SECTION_KEYS

            res = run_full_analysis(pf, llm=deps.get("llm"), **kw)
            secs = [res[k] for k in SECTION_KEYS]
            for s in secs:
                if s["mode"] != "live":
                    raise RuntimeError(f"orchestrator section fell back: {s['reason']}")
            outputs, graders = [s["text"] for s in secs], [s["grader"] for s in secs]
        else:
            raise ValueError(comp)
        extra = {"trace_id": res["trace"]["trace_id"]}

    for g in graders:
        if g and g.get("grader_error"):
            if g.get("grader_rate_limited"):
                raise RateLimitStop(f"grader 429: {g['grader_error']}")
            raise RuntimeError(f"grader unavailable: {g['grader_error']}")
    facts = exact_fact_check(outputs[0], item) if item["grading"] == "exact_fact" else None
    passed = all(_grader_ok(g) for g in graders) and (facts is None or facts["ok"])
    return {
        "id": item["id"], "component": comp, "grading": item["grading"],
        "question": item.get("question"), "variant": (item.get("variant") or {}).get("name"),
        "outputs": outputs, "graders": graders, "exact_fact": facts, "passed": passed,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1), "error": None, **extra,
    }


def summarize(items: List[dict], component: str) -> dict:
    rows = [i for i in items if i["component"] == component]
    n = len(rows)
    g_scores = [
        float(g["groundedness"]) if isinstance(g.get("groundedness"), (int, float)) else 0.0
        for r in rows for g in (r.get("graders") or [{}])
    ]
    passed = sum(1 for r in rows if r["passed"])
    exact = [r for r in rows if r["grading"] == "exact_fact"]
    return {
        "n_items": n,
        "passed": passed,
        "pass_rate": round(passed / n, 4) if n else 0.0,
        "mean_groundedness": round(statistics.mean(g_scores), 4) if g_scores else 0.0,
        "exact_fact_accuracy": round(sum(1 for r in exact if r["exact_fact"] and r["exact_fact"]["ok"]) / len(exact), 4)
        if exact else None,
        "grader_pass_rate": round(sum(1 for r in rows if all(_grader_ok(g) for g in (r.get("graders") or [None])))
                                  / n, 4) if n else 0.0,
        "errors": sum(1 for r in rows if r.get("error")),
    }


def run_eval(
    split: str = "heldout",
    components: Optional[List[str]] = None,
    *,
    deps_factory: Optional[Callable[[List[str]], dict]] = None,
    throttle_s: float = 0.0,
    fake: bool = False,
    write_approvals: bool = True,
    dataset: Optional[List[dict]] = None,
    max_items: Optional[int] = None,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    retry_errors: bool = True,
    retry_pause_s: float = 60.0,
) -> dict:
    components = components or list(LLM_COMPONENTS) + ["MRM-006"]
    llm_comps = [c for c in components if c in LLM_COMPONENTS]
    dataset = dataset if dataset is not None else load_dataset()
    todo = [i for i in dataset if i["split"] == split and i["component"] in llm_comps]
    if max_items:
        todo = todo[:max_items]

    started_at = datetime.now(timezone.utc)
    run_id = f"{split}-{started_at.strftime('%Y%m%dT%H%M%SZ')}" + ("-fake" if fake else "")
    versions = {c: current_version(c) for c in llm_comps}
    run = {
        "run_id": run_id, "split": split, "fake": fake, "started_at": started_at.isoformat(timespec="seconds"),
        "provider": config.get_provider(), "models": config.all_model_ids(),
        "versions": versions, "throttle_s": throttle_s, "items": [], "components": {},
        "complete": True, "stopped_reason": None,
    }

    deps = deps_factory(llm_comps) if (deps_factory and todo) else {}
    for n, item in enumerate(todo, 1):
        try:
            result = run_item(item, deps)
        except Exception as exc:
            if is_rate_limit_error(exc):
                run["complete"] = False
                run["stopped_reason"] = f"HTTP 429 / quota after retry at item {item['id']} ({n}/{len(todo)})"
                log(f"STOP: {run['stopped_reason']}")
                break
            result = {"id": item["id"], "component": item["component"], "grading": item["grading"],
                      "question": item.get("question"), "outputs": [], "graders": [], "exact_fact": None,
                      "passed": False, "error": sanitize_error(exc)}
        run["items"].append(result)
        log(f"[{n}/{len(todo)}] {item['id']}: {'PASS' if result['passed'] else 'FAIL'}"
            + (f" ({result['error']})" if result.get("error") else ""))
        if throttle_s and n < len(todo):
            sleep(throttle_s)

    # Items that errored for infrastructure reasons (e.g. persistent HTTP 503) get exactly one
    # more attempt after a pause; the retry is recorded on the item.
    if run["complete"] and retry_errors:
        for idx, res in enumerate(run["items"]):
            if not res.get("error"):
                continue
            item = next(i for i in todo if i["id"] == res["id"])
            sleep(retry_pause_s)
            try:
                new = run_item(item, deps)
            except Exception as exc:
                if is_rate_limit_error(exc):
                    run["complete"] = False
                    run["stopped_reason"] = f"HTTP 429 / quota while retrying {item['id']}"
                    log(f"STOP: {run['stopped_reason']}")
                    break
                new = {**res, "error": sanitize_error(exc)}
            new["retried_after_error"] = res["error"]
            run["items"][idx] = new
            log(f"[retry] {item['id']}: {'PASS' if new['passed'] else 'FAIL'}"
                + (f" ({new['error']})" if new.get("error") else ""))

    for comp in llm_comps:
        s = summarize(run["items"], comp)
        s["status"] = decide_status(s["pass_rate"], s["mean_groundedness"]) if s["n_items"] else None
        run["components"][comp] = s

    if "MRM-006" in components:
        from src.mrm.validation.model_validator import portfolio_backtest

        bt = portfolio_backtest()
        run["components"]["MRM-006"] = {
            "status": backtest_status(bt["overall_zone"]), "basel_zone": bt["overall_zone"],
            "methods": {k: {x: v[x] for x in v if x != "exception_dates"} for k, v in bt["methods"].items()},
            "period": bt["period"], "window": bt["window"], "confidence": bt["confidence"],
        }

    run["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_dir = results_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_id}.json")
    with open(path, "w") as f:
        json.dump(run, f, indent=2, default=str)
    run["_path"] = path

    if write_approvals and not fake and split == "heldout":
        for comp, s in run["components"].items():
            if comp == "MRM-006":
                write_approval(comp, {
                    "status": s["status"], "model_id": current_version("MRM-006")["model_id"], "prompt_versions": [],
                    "approved_at": run["finished_at"], "approved_by": "eval-harness (scripts/run_eval.py)",
                    "evidence": run_id, "metrics": {"basel_zone": s["basel_zone"], "period": s["period"]},
                })
            elif run["complete"] and s["n_items"] and not s["errors"]:
                write_approval(comp, {
                    "status": s["status"], "model_id": versions[comp]["model_id"],
                    "prompt_versions": versions[comp]["prompt_versions"],
                    "approved_at": run["finished_at"], "approved_by": "eval-harness (scripts/run_eval.py)",
                    "evidence": run_id,
                    "metrics": {k: s[k] for k in ("n_items", "pass_rate", "mean_groundedness", "exact_fact_accuracy",
                                                  "grader_pass_rate")},
                    "grader_model": config.model_id("grader"),
                })
    return run


def live_deps_factory(components: List[str]) -> dict:
    """Real Gemini models + real index + local cross-encoder."""
    from src.llm.provider import Embedder, get_chat_model
    from src.rag.reranker import get_reranker
    from src.rag.vector_store import get_index

    deps: Dict[str, object] = {
        "llm": get_chat_model("chat"),
        "grader_llm": get_chat_model("grader"),
        "hyde_llm": get_chat_model("hyde"),
    }
    if "MRM-001" in components:
        emb = Embedder()
        deps["embedder"] = emb
        deps["index"] = get_index(emb)
        deps["reranker"] = get_reranker()
    return deps
