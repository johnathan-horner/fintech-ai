"""
FinTech AI - Model Monitoring
SR 11-7 requires ongoing monitoring of models in production.

Everything here is computed from real inputs - there are no random or
simulated values:
  1. Performance - derived from the actual trace log (latency, errors,
     demo fallbacks, grader verdicts). No traces -> NO_DATA.
  2. Drift (MRM-006) - PSI of portfolio daily returns: reference = first
     250-day window of data/price_history.json, current = last 60 days.
  3. Drift (LLM components) - PSI of grader groundedness scores: reference =
     the latest held-out eval run, current = scores logged in this
     session / process. n < 30 -> "insufficient data".
  4. Validation expiry - from approval dates in data/mrm/approvals.json.
"""

import math
import statistics
from collections import Counter
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

PSI_MIN_CURRENT_N = 30
PSI_BINS = 10
PSI_EPS = 1e-4
REFERENCE_WINDOW = 250
CURRENT_WINDOW = 60


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


MONITORING_THRESHOLDS = {
    "error_rate_warning": 0.05,        # >5% errors -> WARNING
    "error_rate_critical": 0.15,       # >15% errors -> CRITICAL
    "avg_latency_warning_ms": 10000,   # >10s avg latency -> WARNING
    "avg_latency_critical_ms": 30000,  # >30s avg latency -> CRITICAL
    "flag_rate_warning": 0.20,         # >20% of graded outputs FLAGged -> WARNING
    "psi_moderate": 0.10,
    "psi_significant": 0.25,
    "validation_expiry_warning_days": 60,
}


#  PSI

def decile_edges(reference: List[float], bins: int = PSI_BINS) -> List[float]:
    """Interior bin edges at the reference quantiles (bins-1 edges)."""
    xs = sorted(reference)
    n = len(xs)
    edges = []
    for k in range(1, bins):
        q = k / bins * (n - 1)
        lo = int(math.floor(q))
        hi = min(lo + 1, n - 1)
        edges.append(xs[lo] + (xs[hi] - xs[lo]) * (q - lo))
    return edges


def _bin_shares(values: List[float], edges: List[float]) -> List[float]:
    counts = [0] * (len(edges) + 1)
    for v in values:
        i = 0
        while i < len(edges) and v > edges[i]:
            i += 1
        counts[i] += 1
    n = len(values)
    return [c / n for c in counts]


def psi(reference: List[float], current: List[float], bins: int = PSI_BINS, eps: float = PSI_EPS) -> float:
    """
    Population Stability Index with bins at the reference deciles:
        PSI = sum_i (a_i - e_i) * ln(a_i / e_i)
    e_i / a_i are the reference / current shares per bin (floored at eps).
    """
    if not reference or not current:
        raise ValueError("reference and current must be non-empty")
    edges = decile_edges(reference, bins)
    e = _bin_shares(reference, edges)
    a = _bin_shares(current, edges)
    total = 0.0
    for ei, ai in zip(e, a):
        ei, ai = max(ei, eps), max(ai, eps)
        total += (ai - ei) * math.log(ai / ei)
    return total


def _psi_alert(value: float) -> dict:
    if value > MONITORING_THRESHOLDS["psi_significant"]:
        return {"severity": AlertSeverity.CRITICAL.value,
                "message": f"PSI={value:.3f} > 0.25 - significant drift. Re-validation required.",
                "action": "Trigger re-validation workflow."}
    if value > MONITORING_THRESHOLDS["psi_moderate"]:
        return {"severity": AlertSeverity.WARNING.value,
                "message": f"PSI={value:.3f} > 0.10 - moderate drift. Monitor closely.",
                "action": "Increase monitoring frequency."}
    return {"severity": AlertSeverity.INFO.value,
            "message": f"PSI={value:.3f} <= 0.10 - distribution stable.",
            "action": "No action required."}


def returns_drift(portfolio_data: Optional[dict] = None) -> dict:
    """MRM-006: PSI of portfolio daily returns, first 250 days vs last 60 days."""
    from src.utils.data import load_portfolio, load_price_history, price_history_available
    from src.utils.metrics import portfolio_pnl_series

    base = {
        "model_id": "MRM-006",
        "drift_metric": "PSI of portfolio daily returns (decile bins from reference)",
        "interpretation": "PSI <= 0.10: stable | 0.10-0.25: moderate drift | > 0.25: significant drift",
    }
    if not price_history_available():
        return {**base, "status": "NO_DATA", "psi_value": None, "alerts": [],
                "message": "data/price_history.json missing - run scripts/generate_price_history.py"}
    portfolio_data = portfolio_data or load_portfolio()
    series = portfolio_pnl_series(load_price_history(), portfolio_data["positions"])
    rets, dates = series["returns"], series["dates"]
    ref, cur = rets[:REFERENCE_WINDOW], rets[-CURRENT_WINDOW:]
    value = psi(ref, cur)
    return {
        **base,
        "status": "OK",
        "psi_value": round(value, 4),
        "reference_period": f"{dates[0]} to {dates[REFERENCE_WINDOW - 1]} (n={len(ref)})",
        "current_period": f"{dates[-CURRENT_WINDOW]} to {dates[-1]} (n={len(cur)})",
        "reference_n": len(ref),
        "current_n": len(cur),
        "data": "data/price_history.json (SYNTHETIC)",
        "alerts": [_psi_alert(value)],
    }


def grader_score_drift(model_id: str, current_scores: List[float], reference_scores: Optional[List[float]]) -> dict:
    """LLM components: PSI of grader groundedness scores (eval reference vs live)."""
    base = {
        "model_id": model_id,
        "drift_metric": "PSI of grader groundedness scores (held-out eval vs this session/process)",
        "current_n": len(current_scores),
        "reference_n": len(reference_scores or []),
    }
    if not reference_scores:
        return {**base, "status": "NO_REFERENCE", "psi_value": None, "alerts": [],
                "message": "No held-out eval run with grader scores yet - run scripts/run_eval.py --split heldout"}
    if len(current_scores) < PSI_MIN_CURRENT_N:
        return {**base, "status": "INSUFFICIENT_DATA", "psi_value": None, "alerts": [],
                "message": f"insufficient data (n={len(current_scores)}; need >= {PSI_MIN_CURRENT_N})"}
    value = psi(reference_scores, current_scores)
    return {**base, "status": "OK", "psi_value": round(value, 4), "alerts": [_psi_alert(value)]}


def detect_input_drift(model_id: str = "MRM-006", traces: Optional[List[dict]] = None) -> dict:
    if model_id == "MRM-006":
        return returns_drift()
    from src.eval.results import reference_grader_scores

    current = [
        t["grader"]["groundedness"] for t in (traces or [])
        if t.get("inventory_id") == model_id and t.get("mode") == "live"
        and isinstance(t.get("grader"), dict) and isinstance(t["grader"].get("groundedness"), (int, float))
    ]
    return grader_score_drift(model_id, current, reference_grader_scores(model_id))


#  Performance (from the trace log)

def monitor_performance(model_id: str, traces: Optional[List[dict]] = None) -> dict:
    model_traces = [t for t in (traces or []) if t.get("inventory_id") == model_id]
    if not model_traces:
        return {"model_id": model_id, "status": "NO_DATA", "alerts": [], "metrics": {"requests": 0},
                "message": "No traced requests for this model in the current session/process."}
    alerts = []
    n = len(model_traces)
    live = [t for t in model_traces if t.get("mode") == "live"]
    errors = [t for t in model_traces if t.get("error")]
    latencies = sorted(t["latency_ms"] for t in live if isinstance(t.get("latency_ms"), (int, float)))
    graded = [t for t in live if isinstance(t.get("grader"), dict) and t["grader"].get("verdict")]
    flagged = [t for t in graded if t["grader"]["verdict"] != "PASS"]
    error_rate = len(errors) / n
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else 0.0
    flag_rate = len(flagged) / len(graded) if graded else 0.0

    th = MONITORING_THRESHOLDS
    if error_rate > th["error_rate_critical"]:
        alerts.append({"severity": "CRITICAL", "message": f"Error rate {error_rate:.1%} > {th['error_rate_critical']:.0%}."})
    elif error_rate > th["error_rate_warning"]:
        alerts.append({"severity": "WARNING", "message": f"Error rate {error_rate:.1%} > {th['error_rate_warning']:.0%}."})
    if avg_latency > th["avg_latency_critical_ms"]:
        alerts.append({"severity": "CRITICAL", "message": f"Avg latency {avg_latency:.0f}ms > {th['avg_latency_critical_ms']}ms."})
    elif avg_latency > th["avg_latency_warning_ms"]:
        alerts.append({"severity": "WARNING", "message": f"Avg latency {avg_latency:.0f}ms > {th['avg_latency_warning_ms']}ms."})
    if graded and flag_rate > th["flag_rate_warning"]:
        alerts.append({"severity": "WARNING", "message": f"{flag_rate:.0%} of graded outputs FLAGged by the grader."})

    return {
        "model_id": model_id,
        "status": "CRITICAL" if any(a["severity"] == "CRITICAL" for a in alerts) else "WARNING" if alerts else "HEALTHY",
        "metrics": {
            "requests": n,
            "live": len(live),
            "demo_or_blocked": n - len(live),
            "errors": len(errors),
            "error_rate": round(error_rate, 4),
            "avg_latency_ms": round(avg_latency, 0),
            "p95_latency_ms": round(p95_latency, 0),
            "graded": len(graded),
            "grader_flag_rate": round(flag_rate, 4),
            "modes": dict(Counter(t.get("mode") for t in model_traces)),
        },
        "alerts": alerts,
        "assessed_at": datetime.today().isoformat(timespec="seconds"),
    }


#  Validation Expiry Monitor

def check_validation_expiry() -> List[dict]:
    """Check all approved models for upcoming or expired validation deadlines."""
    from src.mrm.model_inventory import get_inventory

    alerts = []
    today = datetime.today().date()
    warn = timedelta(days=MONITORING_THRESHOLDS["validation_expiry_warning_days"])
    for model in get_inventory().get_all():
        due_str = model.get("next_validation_due")
        if not due_str:
            continue
        due = datetime.strptime(due_str, "%Y-%m-%d").date()
        days_until = (due - today).days
        if days_until < 0:
            alerts.append({"model_id": model["model_id"], "model_name": model["name"], "severity": "CRITICAL",
                           "message": f"Validation EXPIRED {abs(days_until)} days ago (due {due_str}).",
                           "action": "Re-run the eval harness or suspend model use."})
        elif days_until <= warn.days:
            alerts.append({"model_id": model["model_id"], "model_name": model["name"], "severity": "WARNING",
                           "message": f"Validation expires in {days_until} days (due {due_str}).",
                           "action": "Schedule re-validation."})
    return alerts


#  Full Monitoring Report

def run_monitoring_report(model_ids: Optional[List[str]] = None, traces: Optional[List[dict]] = None) -> dict:
    """Monitoring report across all models from real data and the given trace log."""
    from src.mrm.model_inventory import get_inventory
    from src.mrm.trace import recent_traces

    inv = get_inventory()
    model_ids = model_ids or list(inv.models.keys())
    traces = recent_traces(500) if traces is None else traces

    performance = [monitor_performance(mid, traces) for mid in model_ids if mid != "MRM-006"]
    drift = {mid: detect_input_drift(mid, traces) for mid in model_ids}
    expiry = check_validation_expiry()

    sev = [a["severity"] for p in performance for a in p["alerts"]]
    sev += [a["severity"] for d in drift.values() for a in d.get("alerts", [])]
    sev += [a["severity"] for a in expiry]
    overall = "CRITICAL" if "CRITICAL" in sev else "WARNING" if "WARNING" in sev else "HEALTHY"
    return {
        "report_date": datetime.today().isoformat(timespec="seconds"),
        "overall_status": overall,
        "summary": {
            "models_monitored": len(model_ids),
            "traced_requests": len(traces),
            "critical_alerts": sev.count("CRITICAL"),
            "warning_alerts": sev.count("WARNING"),
        },
        "performance": performance,
        "drift": drift,
        "validation_expiry": expiry,
    }
