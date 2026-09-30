"""
FinTech AI - independent output grader.

Every AI output is scored twice:
  1. an LLM grader - a SECOND, independent call to a different model
     (config role "grader", default gemini-2.5-flash-lite) with its own prompt,
     given the exact sources/inputs the output was generated from; and
  2. a deterministic numeric check - every metric-looking number in the output
     must appear in (or round from) a number in the sources.
"""

from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional

from src.llm import config
from src.llm.guard import call_llm, is_rate_limit_error, sanitize_error
from src.llm.provider import message_text
from src.prompts import registry

GROUNDEDNESS_PASS = 0.8
NUMERIC_PASS = 0.9

# A number with optional $, thousands separators, decimals, and a unit suffix.
_NUM_RE = re.compile(
    r"(?P<neg>[-−])?(?P<cur>\$)?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<suffix>\s?(?:%|x\b|[KkMB]\b|bn\b|bps\b|million\b|billion\b|thousand\b))?"
)
_SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}


def _parse_numbers(text: str) -> List[dict]:
    out = []
    for m in _NUM_RE.finditer(text):
        raw = m.group("num")
        # skip digits that are part of an identifier like "MRM-001", "Q1", "2Y", "SMA-50"
        start = m.start("num")
        prev = text[start - 1] if start > 0 else " "
        nxt = text[m.end("num")] if m.end("num") < len(text) else " "
        if prev.isalpha() or (prev == "-" and start >= 2 and text[start - 2].isalpha()) or nxt.isalpha() and not m.group("suffix"):
            continue
        value = float(raw.replace(",", ""))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        suffix = (m.group("suffix") or "").strip().lower()
        out.append({
            "text": m.group(0).strip(),
            "value": value,
            "decimals": decimals,
            "currency": bool(m.group("cur")),
            "suffix": suffix,
            "negative": bool(m.group("neg")),
            "has_separator": "," in raw,
        })
    return out


def _is_metric_like(n: dict) -> bool:
    """Numbers we hold the output accountable for."""
    if n["currency"] or n["suffix"] or n["decimals"] > 0 or n["has_separator"]:
        return True
    return False


def source_values(texts: Iterable[str]) -> List[float]:
    vals = []
    for t in texts:
        for n in _parse_numbers(t):
            v = n["value"] * _SCALE.get(n["suffix"], 1.0)
            vals.append(-v if n["negative"] else v)
    return vals


def _candidates(n: dict) -> List[float]:
    """Absolute values the displayed number could stand for."""
    v = n["value"]
    s = n["suffix"]
    if s in _SCALE:
        return [v * _SCALE[s]]
    if s == "%":
        return [v, v / 100.0]
    return [v]


def _matches(n: dict, src: float) -> bool:
    src = abs(src)
    for cand in _candidates(n):
        scale = _SCALE.get(n["suffix"], 1.0)
        if n["suffix"] == "%" and cand < 1 and abs(n["value"]) >= 1:
            scale = 0.01
        tol = 0.5 * (10 ** -n["decimals"]) * scale * 1.0001 + 1e-9
        if abs(cand - src) <= tol:
            return True
        # truncation instead of rounding
        if 0 <= src - cand < 2 * tol:
            return True
    return False


def numeric_check(output: str, sources: Iterable[str]) -> dict:
    """
    Every metric-like number in ``output`` ($, %, x, K/M/B, decimals or
    thousands separators) must equal a number from ``sources`` after rounding
    to the precision shown. Returns the mismatches.
    """
    src_vals = source_values(sources)
    checked, mismatches = 0, []
    for n in _parse_numbers(output):
        if not _is_metric_like(n):
            continue
        checked += 1
        if not any(_matches(n, s) for s in src_vals):
            mismatches.append(n["text"])
    matched = checked - len(mismatches)
    return {
        "checked": checked,
        "matched": matched,
        "mismatches": mismatches,
        "score": round(matched / checked, 4) if checked else 1.0,
    }


def contains_value(output: str, value: float, decimals_hint: Optional[int] = None) -> bool:
    """True if ``output`` shows a number that rounds from ``value``."""
    for n in _parse_numbers(output):
        if _matches(n, value):
            return True
    return False


def parse_grader_json(text: str) -> dict:
    """Parse the grader's JSON with a safe fallback (verdict FLAG on failure)."""
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.MULTILINE).strip()
    data = None
    try:
        data = json.loads(raw)
    except ValueError:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except ValueError:
                data = None
    if not isinstance(data, dict):
        return {"groundedness": None, "numeric_consistency": None, "unsupported_claims": [],
                "verdict": "FLAG", "parse_error": True}

    def unit(x):
        try:
            return min(1.0, max(0.0, float(x)))
        except (TypeError, ValueError):
            return None

    g = unit(data.get("groundedness"))
    nc = unit(data.get("numeric_consistency"))
    claims = data.get("unsupported_claims") or []
    if not isinstance(claims, list):
        claims = [str(claims)]
    verdict = str(data.get("verdict", "")).strip().upper()
    if verdict not in ("PASS", "FLAG"):
        verdict = "PASS" if (g is not None and nc is not None and g >= GROUNDEDNESS_PASS and nc >= NUMERIC_PASS) else "FLAG"
    return {"groundedness": g, "numeric_consistency": nc,
            "unsupported_claims": [str(c)[:200] for c in claims[:5]],
            "verdict": verdict, "parse_error": g is None or nc is None}


def format_sources(sources: List[dict]) -> str:
    return "\n\n".join(f"[{s['source_id']}] {s['text']}" for s in sources)


def grade_output(output: str, sources: List[dict], llm=None) -> dict:
    """
    Grade ``output`` against ``sources`` (list of {"source_id", "text"}) with the
    independent grader model plus the deterministic numeric check. ``llm`` defaults
    to the configured grader model.

    If the grader call fails (e.g. HTTP 503 "high demand" after retries, or 429) the
    output is NOT dropped: the result carries verdict "UNGRADED", ``grader_error`` and
    the numeric check only, so callers can label it (and the eval harness treats it as
    an errored item).
    """
    prompt = registry.get("grader")
    num = numeric_check(output, [s["text"] for s in sources])
    base = {"numeric_check": num, "grader_model": config.model_id("grader"), "grader_prompt": prompt.ref}
    try:
        msg = call_llm(llm or _grader_llm(), prompt.format(sources=format_sources(sources), output=output))
        raw = message_text(msg)
    except Exception as exc:
        return {"groundedness": None, "numeric_consistency": None, "unsupported_claims": [],
                "llm_verdict": None, "verdict": "UNGRADED", "parse_error": False,
                "grader_error": sanitize_error(exc), "grader_rate_limited": is_rate_limit_error(exc), **base}
    llm_result = parse_grader_json(raw)
    verdict = "PASS" if llm_result["verdict"] == "PASS" and not num["mismatches"] else "FLAG"
    return {**llm_result, "llm_verdict": llm_result["verdict"], "verdict": verdict, **base}


def deterministic_grade(output: str, sources: List[dict]) -> dict:
    """Numeric check only (used for demo outputs, where no LLM is called)."""
    num = numeric_check(output, [s["text"] for s in sources])
    return {"groundedness": None, "numeric_consistency": num["score"], "unsupported_claims": [],
            "llm_verdict": None, "verdict": "PASS" if not num["mismatches"] else "FLAG",
            "numeric_check": num, "grader_model": None, "grader_prompt": None,
            "note": "LLM grader not run (demo output); deterministic numeric check only"}


def _grader_llm():
    from src.llm.provider import get_chat_model

    return get_chat_model("grader")
