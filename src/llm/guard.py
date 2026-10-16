"""
FinTech AI - live-call guard (framework independent).

``run_guarded(live_fn, demo_fn)`` runs a live AI request and falls back to a
deterministic demo output when:
  * no provider is configured (no Bedrock credentials and no Gemini key),
  * a per-session request cap is reached (the cap is tracked by the caller),
  * model governance blocks the call (``GovernanceBlocked``),
  * the provider returns HTTP 429 / RESOURCE_EXHAUSTED (every model call goes
    through ``call_llm`` which retries once after a short backoff first),
  * any other exception is raised.

It never raises. The Streamlit wrapper (streamlit_app/live_ai.py) adds the
10-requests-per-session cap on top of this.
"""

from __future__ import annotations

import re
import time
from typing import Callable, Optional

from src.llm import config

RATE_LIMIT_BACKOFF_S = 3.0
# Transient server errors (e.g. 503 "model is experiencing high demand"): retry with these waits.
TRANSIENT_BACKOFFS_S = (10.0, 30.0)

_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{10,}")


class GovernanceBlocked(RuntimeError):
    """Raised by the approval gate before any model call is made."""


def is_rate_limit_error(exc: BaseException) -> bool:
    """Detect Google ResourceExhausted / HTTP 429 / RESOURCE_EXHAUSTED errors."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        name = type(exc).__name__
        if name in ("ResourceExhausted", "GoogleRateLimitError", "RateLimitError", "ModelRateLimitError"):
            return True
        for attr in ("code", "status_code", "status"):
            val = getattr(exc, attr, None)
            if val == 429 or val == "RESOURCE_EXHAUSTED":
                return True
        text = str(exc)
        if "429" in text or "RESOURCE_EXHAUSTED" in text or "ResourceExhausted" in text:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def is_transient_server_error(exc: BaseException) -> bool:
    """HTTP 500/502/503/504 / UNAVAILABLE from the provider (not quota)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        for attr in ("code", "status_code"):
            if getattr(exc, attr, None) in (500, 502, 503, 504):
                return True
        text = str(exc)
        if any(t in text for t in ("503 UNAVAILABLE", "500 INTERNAL", "502 Bad Gateway", "504 DEADLINE", "high demand")):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


_RETRY_IN = re.compile(r"retry in ([0-9.]+)\s*(ms|s)", re.IGNORECASE)
_RETRY_DELAY = re.compile(r"retryDelay'?\"?\s*[:=]\s*'?\"?([0-9.]+)s")
MAX_SERVER_RETRY_DELAY_S = 30.0


def suggested_retry_delay(exc: BaseException) -> Optional[float]:
    """Seconds the provider asked us to wait (Gemini puts it in the 429 message), capped."""
    text = str(exc)
    m = _RETRY_IN.search(text)
    if m:
        v = float(m.group(1)) / (1000.0 if m.group(2).lower() == "ms" else 1.0)
        return min(v + 1.0, MAX_SERVER_RETRY_DELAY_S)
    m = _RETRY_DELAY.search(text)
    if m:
        return min(float(m.group(1)) + 1.0, MAX_SERVER_RETRY_DELAY_S)
    return None


def sanitize_error(exc: BaseException) -> str:
    """Short, key-free description of an exception for display."""
    msg = _KEY_PATTERN.sub("[REDACTED]", str(exc)).replace("\n", " ")
    if len(msg) > 160:
        msg = msg[:157] + "..."
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


def _short(text: str, n: int = 240) -> str:
    text = _KEY_PATTERN.sub("[REDACTED]", text).replace("\n", " ")
    return text if len(text) <= n else text[: n - 3] + "..."


def run_guarded(
    live_fn: Callable[[], dict],
    demo_fn: Callable[[str], dict],
    *,
    credentials_present: Optional[bool] = None,
) -> dict:
    """
    ``demo_fn`` receives the fallback reason.
    Returns a dict: {"mode": "live"|"demo", "reason": str|None,
    "blocked": bool, "output": <live or demo payload>, "latency_ms": float}.
    """
    started = time.perf_counter()

    def demo(reason: str, blocked: bool = False) -> dict:
        try:
            payload = demo_fn(reason)
        except Exception as exc:  # demo generation must never crash the UI
            payload = {"text": f"Demo output unavailable ({sanitize_error(exc)})."}
        return {
            "mode": "demo",
            "reason": reason,
            "blocked": blocked,
            "output": payload,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    if credentials_present is None:
        credentials_present = config.has_credentials()
    if not credentials_present:
        return demo("No LLM provider configured (no AWS credentials for Bedrock, no Gemini API key)")

    try:
        payload = live_fn()
        return {
            "mode": "live",
            "reason": None,
            "blocked": False,
            "output": payload,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    except GovernanceBlocked as exc:
        return demo(f"Blocked by model governance: {exc}", blocked=True)
    except Exception as exc:
        name = type(exc).__name__
        if name == "EmbeddingUnavailable":
            return demo(_short(str(exc)))
        if name == "ProviderChainFailed":
            return demo("All live providers failed - " + _short(str(exc), 300))
        if is_rate_limit_error(exc):
            return demo("Rate limit / quota reached (HTTP 429, retried once)")
        return demo(f"Live AI call failed ({sanitize_error(exc)})")


def call_with_retry(
    fn: Callable[[], object],
    *,
    backoff_s: float = RATE_LIMIT_BACKOFF_S,
    sleep: Callable[[float], None] = time.sleep,
    transient_backoffs: Optional[tuple] = None,
):
    """
    Call ``fn``. On HTTP 429 / RESOURCE_EXHAUSTED wait the provider's suggested
    delay (capped at 30 s; ``backoff_s`` if none) and retry exactly once. On transient 5xx (e.g. 503 high demand) retry after each wait
    in ``transient_backoffs``. Anything else (or a repeat failure) is re-raised.
    """
    rate_limited_once = False
    transient_left = list(TRANSIENT_BACKOFFS_S if transient_backoffs is None else transient_backoffs)
    while True:
        try:
            return fn()
        except Exception as exc:
            if is_rate_limit_error(exc) and not rate_limited_once:
                rate_limited_once = True
                sleep(max(backoff_s, suggested_retry_delay(exc) or 0.0))
                continue
            if not is_rate_limit_error(exc) and is_transient_server_error(exc) and transient_left:
                sleep(transient_left.pop(0))
                continue
            raise


def call_llm(llm, messages, *, backoff_s: float = RATE_LIMIT_BACKOFF_S,
             sleep: Callable[[float], None] = time.sleep, transient_backoffs: Optional[tuple] = None):
    """Invoke a LangChain chat model with ``call_with_retry`` semantics."""
    return call_with_retry(lambda: llm.invoke(messages), backoff_s=backoff_s, sleep=sleep,
                           transient_backoffs=transient_backoffs)
