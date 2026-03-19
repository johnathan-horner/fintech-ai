"""
FinTech AI - Model Monitoring
SR 11-7 requires ongoing monitoring of models in production.
Tracks: output quality, drift, error rates, usage, and anomalies.

Monitoring dimensions:
  1. Performance Monitoring - is the model still producing quality outputs?
  2. Data Drift - has the input distribution shifted?
  3. Usage Monitoring - who is using it, how often, with what queries?
  4. Anomaly Detection - sudden changes in output patterns
  5. Alert Generation - triggers for re-validation or escalation
"""

import json
import os
import random
import statistics
import math
from datetime import datetime, timedelta
from typing import List, Optional
from enum import Enum

LOG_PATH = os.path.join(os.path.dirname(__file__), "../../../logs/mrm_monitoring.json")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class MonitoringAlert:
    def __init__(self, model_id: str, severity: AlertSeverity, message: str, metric: str, value: float):
        self.model_id = model_id
        self.severity = severity
        self.message = message
        self.metric = metric
        self.value = value
        self.timestamp = datetime.today().isoformat()

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "severity": self.severity,
            "message": self.message,
            "metric": self.metric,
            "value": self.value,
            "timestamp": self.timestamp,
        }


#  Thresholds 

MONITORING_THRESHOLDS = {
    "error_rate_warning": 0.05,        # >5% errors -> WARNING
    "error_rate_critical": 0.15,       # >15% errors -> CRITICAL
    "avg_latency_warning_ms": 10000,   # >10s avg latency -> WARNING
    "avg_latency_critical_ms": 30000,  # >30s avg latency -> CRITICAL
    "output_length_min_chars": 50,     # outputs < 50 chars likely degenerate
    "daily_usage_min": 1,              # < 1 call/day -> model may be abandoned
    "var_drift_threshold": 0.25,       # VaR shifts >25% without market explanation -> investigate
    "validation_expiry_warning_days": 60,  # warn 60 days before validation expires
}


#  Simulated Usage Log Generator 

def generate_usage_logs(model_id: str, days: int = 30) -> List[dict]:
    """
    Generate synthetic usage logs for a model over the past N days.
    In production this would pull from CloudWatch logs.
    """
    random.seed(42)
    logs = []
    base_date = datetime.today() - timedelta(days=days)

    for i in range(days):
        date = base_date + timedelta(days=i)
        # Simulate realistic usage patterns (more during trading hours)
        n_calls = random.randint(2, 25) if date.weekday() < 5 else 0

        for _ in range(n_calls):
            latency_ms = random.gauss(4500, 1200) if model_id in ["MRM-001", "MRM-002"] else random.gauss(8000, 2000)
            error = random.random() < 0.03  # 3% base error rate
            output_length = 0 if error else random.randint(200, 3000)

            logs.append({
                "model_id": model_id,
                "timestamp": date.isoformat(),
                "latency_ms": max(100, latency_ms),
                "error": error,
                "output_length_chars": output_length,
                "user": random.choice(["analyst_1", "analyst_2", "portfolio_mgr"]),
                "query_type": random.choice(["chat", "market", "risk", "portfolio"]),
            })

    return logs


#  Performance Monitor 

def monitor_performance(model_id: str, logs: Optional[List[dict]] = None) -> dict:
    """
    Analyze usage logs for performance KPIs:
    - Error rate
    - Average/P95 latency
    - Output quality (length proxy)
    - Daily usage volume
    """
    if logs is None:
        logs = generate_usage_logs(model_id, days=30)

    model_logs = [l for l in logs if l["model_id"] == model_id]
    alerts = []

    if not model_logs:
        return {"model_id": model_id, "status": "NO_DATA", "alerts": [], "metrics": {}}

    # Error rate
    error_rate = sum(1 for l in model_logs if l.get("error")) / len(model_logs)

    # Latency
    latencies = [l["latency_ms"] for l in model_logs if not l.get("error")]
    avg_latency = statistics.mean(latencies) if latencies else 0
    latencies_sorted = sorted(latencies)
    p95_latency = latencies_sorted[int(0.95 * len(latencies_sorted))] if latencies else 0

    # Output quality
    lengths = [l["output_length_chars"] for l in model_logs if not l.get("error")]
    avg_output_len = statistics.mean(lengths) if lengths else 0
    degenerate_outputs = sum(1 for l in lengths if l < MONITORING_THRESHOLDS["output_length_min_chars"])

    # Daily usage
    from collections import Counter
    daily_counts = Counter(l["timestamp"][:10] for l in model_logs)
    avg_daily_calls = statistics.mean(daily_counts.values()) if daily_counts else 0

    # Generate alerts
    if error_rate > MONITORING_THRESHOLDS["error_rate_critical"]:
        alerts.append(MonitoringAlert(
            model_id, AlertSeverity.CRITICAL,
            f"Error rate {error_rate:.1%} exceeds critical threshold of {MONITORING_THRESHOLDS['error_rate_critical']:.0%}.",
            "error_rate", error_rate
        ))
    elif error_rate > MONITORING_THRESHOLDS["error_rate_warning"]:
        alerts.append(MonitoringAlert(
            model_id, AlertSeverity.WARNING,
            f"Error rate {error_rate:.1%} exceeds warning threshold.",
            "error_rate", error_rate
        ))

    if avg_latency > MONITORING_THRESHOLDS["avg_latency_critical_ms"]:
        alerts.append(MonitoringAlert(
            model_id, AlertSeverity.CRITICAL,
            f"Avg latency {avg_latency:.0f}ms exceeds critical threshold of {MONITORING_THRESHOLDS['avg_latency_critical_ms']}ms.",
            "avg_latency_ms", avg_latency
        ))
    elif avg_latency > MONITORING_THRESHOLDS["avg_latency_warning_ms"]:
        alerts.append(MonitoringAlert(
            model_id, AlertSeverity.WARNING,
            f"Avg latency {avg_latency:.0f}ms is elevated.",
            "avg_latency_ms", avg_latency
        ))

    if degenerate_outputs > 0:
        alerts.append(MonitoringAlert(
            model_id, AlertSeverity.WARNING,
            f"{degenerate_outputs} outputs below minimum length threshold - potential quality degradation.",
            "degenerate_outputs", degenerate_outputs
        ))

    metrics = {
        "total_calls_30d": len(model_logs),
        "avg_daily_calls": round(avg_daily_calls, 1),
        "error_rate": round(error_rate, 4),
        "avg_latency_ms": round(avg_latency, 0),
        "p95_latency_ms": round(p95_latency, 0),
        "avg_output_length_chars": round(avg_output_len, 0),
        "degenerate_outputs": degenerate_outputs,
    }

    return {
        "model_id": model_id,
        "period": "last_30_days",
        "status": "CRITICAL" if any(a.severity == AlertSeverity.CRITICAL for a in alerts)
                  else "WARNING" if alerts else "HEALTHY",
        "metrics": metrics,
        "alerts": [a.to_dict() for a in alerts],
        "assessed_at": datetime.today().isoformat(),
    }


#  Data Drift Detection 

def detect_input_drift(model_id: str = "MRM-006") -> dict:
    """
    Detect input distribution drift using Population Stability Index (PSI).
    PSI > 0.25 indicates significant drift requiring re-validation.
    Compares baseline (training) beta distribution vs. current portfolio.
    """
    # Simulate baseline distribution (at model training / last validation)
    random.seed(42)
    baseline_betas = [random.gauss(1.1, 0.35) for _ in range(200)]

    # Simulate current distribution (slight drift introduced)
    random.seed(99)
    current_betas = [random.gauss(1.35, 0.55) for _ in range(50)]  # shifted higher

    def psi(expected: List[float], actual: List[float], buckets: int = 10) -> float:
        min_val = min(min(expected), min(actual))
        max_val = max(max(expected), max(actual))
        bucket_size = (max_val - min_val) / buckets

        psi_value = 0.0
        for i in range(buckets):
            lower = min_val + i * bucket_size
            upper = lower + bucket_size
            exp_count = sum(1 for x in expected if lower <= x < upper) / len(expected)
            act_count = sum(1 for x in actual if lower <= x < upper) / len(actual)
            exp_count = max(exp_count, 1e-6)
            act_count = max(act_count, 1e-6)
            psi_value += (act_count - exp_count) * math.log(act_count / exp_count)
        return psi_value

    psi_value = psi(baseline_betas, current_betas)

    alerts = []
    if psi_value > 0.25:
        alerts.append({
            "severity": AlertSeverity.CRITICAL,
            "message": f"PSI={psi_value:.3f} > 0.25 - significant input drift detected. Model re-validation required.",
            "action": "Trigger re-validation workflow. Suspend Tier 1 decisions until validated.",
        })
    elif psi_value > 0.10:
        alerts.append({
            "severity": AlertSeverity.WARNING,
            "message": f"PSI={psi_value:.3f} > 0.10 - moderate input drift detected. Monitor closely.",
            "action": "Increase monitoring frequency. Schedule re-validation within 30 days.",
        })
    else:
        alerts.append({
            "severity": AlertSeverity.INFO,
            "message": f"PSI={psi_value:.3f} < 0.10 - input distribution stable.",
            "action": "No action required.",
        })

    return {
        "model_id": model_id,
        "drift_metric": "Population Stability Index (PSI) - Beta distribution",
        "baseline_period": "Last validation date",
        "current_period": "Last 30 days",
        "psi_value": round(psi_value, 4),
        "interpretation": "PSI < 0.10: No change | 0.10-0.25: Moderate drift | > 0.25: Significant drift",
        "alerts": alerts,
        "assessed_at": datetime.today().isoformat(),
    }


#  Validation Expiry Monitor 

def check_validation_expiry() -> List[dict]:
    """Check all models for upcoming or expired validation deadlines."""
    from src.mrm.model_inventory import get_inventory

    inv = get_inventory()
    alerts = []
    today = datetime.today().date()
    warn_threshold = timedelta(days=MONITORING_THRESHOLDS["validation_expiry_warning_days"])

    for model in inv.get_all():
        due_str = model.get("next_validation_due")
        if not due_str:
            continue
        due_date = datetime.strptime(due_str, "%Y-%m-%d").date()
        days_until = (due_date - today).days

        if days_until < 0:
            alerts.append({
                "model_id": model["model_id"],
                "model_name": model["name"],
                "severity": AlertSeverity.CRITICAL,
                "message": f"Validation EXPIRED {abs(days_until)} days ago (due {due_str}).",
                "days_overdue": abs(days_until),
                "action": "Immediately schedule re-validation or suspend model use.",
            })
        elif days_until <= MONITORING_THRESHOLDS["validation_expiry_warning_days"]:
            alerts.append({
                "model_id": model["model_id"],
                "model_name": model["name"],
                "severity": AlertSeverity.WARNING,
                "message": f"Validation expires in {days_until} days (due {due_str}).",
                "days_until_expiry": days_until,
                "action": "Schedule re-validation immediately.",
            })

    return alerts


#  Full Monitoring Report 

def run_monitoring_report(model_ids: Optional[List[str]] = None) -> dict:
    """Generate a full monitoring report across all models."""
    from src.mrm.model_inventory import get_inventory

    inv = get_inventory()
    if model_ids is None:
        model_ids = list(inv.models.keys())

    performance_reports = [monitor_performance(mid) for mid in model_ids]
    drift_report = detect_input_drift("MRM-006")
    expiry_alerts = check_validation_expiry()

    critical_count = sum(1 for r in performance_reports if r["status"] == "CRITICAL")
    warning_count = sum(1 for r in performance_reports if r["status"] == "WARNING")
    expiry_criticals = sum(1 for a in expiry_alerts if a["severity"] == AlertSeverity.CRITICAL)

    overall_status = (
        "CRITICAL" if critical_count > 0 or expiry_criticals > 0
        else "WARNING" if warning_count > 0 or len(expiry_alerts) > 0
        else "HEALTHY"
    )

    return {
        "report_date": datetime.today().isoformat(),
        "overall_status": overall_status,
        "summary": {
            "models_monitored": len(model_ids),
            "critical_alerts": critical_count + expiry_criticals,
            "warning_alerts": warning_count + len([a for a in expiry_alerts if a["severity"] == AlertSeverity.WARNING]),
        },
        "performance": performance_reports,
        "drift": drift_report,
        "validation_expiry": expiry_alerts,
    }


if __name__ == "__main__":
    report = run_monitoring_report()
    print(f"\n=== MRM Monitoring Report ===")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Models Monitored: {report['summary']['models_monitored']}")
    print(f"Critical Alerts: {report['summary']['critical_alerts']}")
    print(f"Warning Alerts: {report['summary']['warning_alerts']}")
    print(f"\nDrift PSI (MRM-006): {report['drift']['psi_value']}")
    print(f"\nValidation Expiry Alerts: {len(report['validation_expiry'])}")
    for a in report['validation_expiry']:
        print(f"  [{a['severity']}] {a['model_id']}: {a['message']}")
