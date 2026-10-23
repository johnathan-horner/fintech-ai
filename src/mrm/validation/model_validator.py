"""
FinTech AI - Model Validation Engine
SR 11-7 requires independent validation of all models.
This module runs conceptual soundness, data quality, outcome analysis and
sensitivity tests, and reads held-out eval evidence for LLM components.

Validation covers:
  1. Conceptual Soundness - is the model design appropriate for its intended use?
  2. Data Quality - are inputs clean, complete, and representative?
  3. Outcome Analysis - out-of-sample 99% VaR backtest (MRM-006)
  4. Sensitivity Analysis - how do outputs change with input perturbations? (MRM-006)
  5. Held-out Eval Evidence - measured by scripts/run_eval.py (LLM components)
"""

import json
import statistics
import math
from datetime import datetime
from typing import List, Optional
from src.mrm.model_inventory import get_inventory, ValidationStatus, ModelStatus


#  Validation Result 

class ValidationResult:
    def __init__(self, model_id: str, test_name: str):
        self.model_id = model_id
        self.test_name = test_name
        self.passed = False
        self.score = 0.0          # 0.0 - 1.0
        self.findings: List[str] = []
        self.recommendations: List[str] = []
        self.timestamp = datetime.today().isoformat()
        self.details: dict = {}

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "test_name": self.test_name,
            "passed": self.passed,
            "score": round(self.score, 4),
            "findings": self.findings,
            "recommendations": self.recommendations,
            "details": self.details,
            "timestamp": self.timestamp,
        }


#  Test 1: Conceptual Soundness 

def test_conceptual_soundness(model_id: str) -> ValidationResult:
    """
    SR 11-7 ?IV: Validate that the model design is appropriate for its stated purpose.
    Checks: documented limitations, stated use case vs. actual use, dependency mapping.
    """
    result = ValidationResult(model_id, "Conceptual Soundness")
    inv = get_inventory()
    model = inv.get_by_id(model_id)

    if not model:
        result.findings.append(f"Model {model_id} not found in inventory.")
        return result

    score = 1.0
    findings = []
    recommendations = []

    # Check: documented limitations exist
    limitations = model.get("known_limitations", [])
    if len(limitations) == 0:
        score -= 0.3
        findings.append("FAIL: No known limitations documented - SR 11-7 requires explicit limitation disclosure.")
        recommendations.append("Document all known model limitations in the model inventory.")
    elif len(limitations) < 3:
        score -= 0.1
        findings.append("WARN: Fewer than 3 limitations documented - likely incomplete.")
        recommendations.append("Conduct a thorough limitation analysis and expand documentation.")
    else:
        findings.append(f"PASS: {len(limitations)} limitations documented.")

    # Check: mitigations documented for each limitation
    mitigations = model.get("mitigations", [])
    if len(mitigations) < len(limitations) * 0.5:
        score -= 0.2
        findings.append("FAIL: Insufficient mitigations relative to known limitations.")
        recommendations.append("Document explicit mitigations for each known limitation.")
    else:
        findings.append(f"PASS: {len(mitigations)} mitigations documented.")

    # Check: business use clearly defined
    if not model.get("business_use"):
        score -= 0.2
        findings.append("FAIL: Business use not defined.")
    else:
        findings.append(f"PASS: Business use defined - '{model['business_use']}'.")

    # Check: Tier 1 models in staging/production must have validation
    from src.mrm.model_inventory import ModelTier
    if model["tier"] == ModelTier.TIER_1 and model["status"] == ModelStatus.PRODUCTION:
        if model["validation_status"] == ValidationStatus.PENDING:
            score -= 0.4
            findings.append("CRITICAL FAIL: Tier 1 model in PRODUCTION with PENDING validation - SR 11-7 violation.")
            recommendations.append("Immediately suspend production use or complete independent validation.")

    # Check: data sources documented
    if not model.get("data_sources"):
        score -= 0.1
        findings.append("WARN: Data sources not documented.")
    else:
        findings.append(f"PASS: Data sources documented - {model['data_sources']}.")

    result.score = max(0.0, score)
    result.passed = result.score >= 0.7
    result.findings = findings
    result.recommendations = recommendations
    return result


#  Test 2: Data Quality 

def test_data_quality(model_id: str, portfolio_data: Optional[dict] = None) -> ValidationResult:
    """
    Validate input data quality for the given model.
    Checks: completeness, range validity, outlier detection.
    """
    result = ValidationResult(model_id, "Data Quality")

    # Load portfolio if not provided
    if portfolio_data is None:
        try:
            import os
            data_path = os.path.join(os.path.dirname(__file__), "../../../data/portfolio.json")
            with open(data_path) as f:
                portfolio_data = json.load(f)
        except Exception as e:
            result.findings.append(f"FAIL: Could not load portfolio data - {e}")
            return result

    positions = portfolio_data.get("positions", [])
    if not positions:
        result.findings.append("FAIL: No portfolio positions found.")
        return result

    score = 1.0
    findings = []
    recommendations = []

    # Completeness check
    required_fields = ["ticker", "sector", "shares", "avg_cost", "current_price",
                       "market_value", "beta", "weight_pct"]
    missing_fields = []
    for p in positions:
        for field in required_fields:
            if field not in p or p[field] is None:
                missing_fields.append(f"{p.get('ticker', '?')}.{field}")

    if missing_fields:
        score -= min(0.3, len(missing_fields) * 0.02)
        findings.append(f"FAIL: {len(missing_fields)} missing required fields: {missing_fields[:5]}...")
        recommendations.append("Ensure all required fields are populated in portfolio data.")
    else:
        findings.append(f"PASS: All {len(required_fields)} required fields present across {len(positions)} positions.")

    # Range validation
    range_violations = []
    for p in positions:
        beta = p.get("beta", 1.0)
        if beta < 0 or beta > 5:
            range_violations.append(f"{p['ticker']}: beta={beta} out of range [0, 5]")
        weight = p.get("weight_pct", 0)
        if weight < 0 or weight > 100:
            range_violations.append(f"{p['ticker']}: weight_pct={weight} out of range [0, 100]")
        shares = p.get("shares", 0)
        if shares <= 0:
            range_violations.append(f"{p['ticker']}: shares={shares} must be > 0")

    if range_violations:
        score -= min(0.2, len(range_violations) * 0.05)
        findings.append(f"FAIL: {len(range_violations)} range violations detected.")
        for v in range_violations[:3]:
            findings.append(f"  -> {v}")
    else:
        findings.append("PASS: All field ranges within expected bounds.")

    # Weight sum check (should sum to ~100%)
    total_weight = sum(p.get("weight_pct", 0) for p in positions)
    if abs(total_weight - 100) > 2.0:
        score -= 0.15
        findings.append(f"WARN: Portfolio weights sum to {total_weight:.1f}% - expected ~100%.")
        recommendations.append("Recalculate weight_pct to ensure weights sum to 100%.")
    else:
        findings.append(f"PASS: Portfolio weights sum to {total_weight:.1f}%.")

    # Outlier detection on beta
    betas = [p.get("beta", 1.0) for p in positions]
    mean_beta = statistics.mean(betas)
    std_beta = statistics.stdev(betas) if len(betas) > 1 else 0
    outliers = [p["ticker"] for p in positions if abs(p.get("beta", 1.0) - mean_beta) > 2 * std_beta]
    if outliers:
        findings.append(f"WARN: Beta outliers detected (>2? from mean): {outliers}")
        recommendations.append("Review beta outliers for data accuracy.")
    else:
        findings.append(f"PASS: No beta outliers detected (mean={mean_beta:.2f}, ?={std_beta:.2f}).")

    result.score = max(0.0, score)
    result.passed = result.score >= 0.8
    result.findings = findings
    result.recommendations = recommendations
    return result


#  Test 3: Outcome Analysis - out-of-sample VaR Backtesting

BACKTEST_WINDOW = 250
BACKTEST_CONFIDENCE = 0.99
BASEL_ZONE_RANK = {"GREEN": 0, "YELLOW": 1, "RED": 2}


def kupiec_pof(T: int, x: int, p: float) -> dict:
    """
    Kupiec proportion-of-failures likelihood-ratio test.

        LR = -2 ln[(1-p)^(T-x) p^x] + 2 ln[(1-x/T)^(T-x) (x/T)^x]

    Edge cases: x = 0 and x = T use 0*ln(0) = 0. LR >= 0.
    p-value from chi-square(df=1): P(chi2 > LR) = erfc(sqrt(LR/2)).
    """
    if T <= 0 or not 0 < p < 1 or x < 0 or x > T:
        raise ValueError("require T > 0, 0 < p < 1 and 0 <= x <= T")

    def xlogy(a: float, b: float) -> float:
        return 0.0 if a == 0 else a * math.log(b)

    phat = x / T
    ll_null = xlogy(T - x, 1 - p) + xlogy(x, p)
    ll_alt = xlogy(T - x, 1 - phat) + xlogy(x, phat)
    lr = max(0.0, -2.0 * ll_null + 2.0 * ll_alt)
    return {"T": T, "x": x, "p": p, "lr": lr, "p_value": math.erfc(math.sqrt(lr / 2.0))}


def basel_zone(exceptions_250: int) -> str:
    """Basel traffic light for 250 observations at 99%: green 0-4, yellow 5-9, red >= 10."""
    if exceptions_250 <= 4:
        return "GREEN"
    if exceptions_250 <= 9:
        return "YELLOW"
    return "RED"


def run_var_backtest(
    pnl: List[float],
    dates: Optional[List[str]] = None,
    window: int = BACKTEST_WINDOW,
    confidence: float = BACKTEST_CONFIDENCE,
    beta_proxy_var: Optional[float] = None,
) -> dict:
    """
    Out-of-sample backtest: for each day t >= window, estimate VaR from the
    previous ``window`` P&L observations only and compare with the realised
    P&L on day t. An exception is a loss larger than the VaR.
    Kupiec uses every test day; the Basel light uses the most recent 250.
    """
    from src.utils.metrics import historical_var, parametric_var

    n = len(pnl)
    if n <= window:
        raise ValueError(f"need more than {window} P&L observations, got {n}")
    p = 1 - confidence
    methods = {
        "historical_rolling": lambda hist: historical_var(hist, confidence),
        "parametric_rolling": lambda hist: parametric_var(hist, confidence),
    }
    if beta_proxy_var is not None:
        methods["beta_proxy_engine"] = lambda hist: beta_proxy_var

    results = {}
    for name, fn in methods.items():
        flags, vars_ = [], []
        for t in range(window, n):
            var_t = fn(pnl[t - window:t])
            vars_.append(var_t)
            flags.append(pnl[t] < -var_t)
        T = len(flags)
        x = sum(flags)
        last250 = flags[-250:]
        kup = kupiec_pof(T, x, p)
        zone = basel_zone(sum(last250)) if len(last250) == 250 else "N/A"
        exception_dates = [dates[window + i] for i, f in enumerate(flags) if f] if dates else []
        results[name] = {
            "test_observations": T,
            "exceptions": x,
            "exception_rate": round(x / T, 5),
            "expected_rate": round(p, 6),
            "exceptions_last_250": sum(last250),
            "basel_zone": zone,
            "kupiec_lr": round(kup["lr"], 4),
            "kupiec_p_value": round(kup["p_value"], 4),
            "kupiec_reject_95": kup["lr"] > 3.841,
            "mean_var": round(statistics.mean(vars_), 2),
            "last_var": round(vars_[-1], 2),
            "exception_dates": exception_dates,
        }
    worst = max((r["basel_zone"] for r in results.values() if r["basel_zone"] in BASEL_ZONE_RANK),
                key=lambda z: BASEL_ZONE_RANK[z], default="N/A")
    return {
        "window": window,
        "confidence": confidence,
        "n_pnl_days": n,
        "methods": results,
        "overall_zone": worst,
        "zone_rule": "overall zone = worst Basel zone across all tested VaR methods",
    }


def portfolio_backtest(portfolio_data: Optional[dict] = None) -> dict:
    """Backtest the current positions on data/price_history.json (synthetic)."""
    from src.utils.data import load_portfolio, load_price_history
    from src.utils.metrics import calculate_portfolio_var, portfolio_pnl_series

    portfolio_data = portfolio_data or load_portfolio()
    positions = portfolio_data["positions"]
    ph = load_price_history()
    series = portfolio_pnl_series(ph, positions)
    engine = calculate_portfolio_var(positions, confidence=0.99)
    result = run_var_backtest(series["pnl"], series["dates"], beta_proxy_var=abs(engine["portfolio_var_1d"]))
    result["data"] = "data/price_history.json (SYNTHETIC, seeded)"
    result["positions"] = len(positions)
    result["period"] = f"{series['dates'][0]} to {series['dates'][-1]}"
    return result


def test_var_backtesting(model_id: str = "MRM-006", portfolio_data: Optional[dict] = None) -> ValidationResult:
    """
    SR 11-7 outcome analysis: out-of-sample 99% VaR backtest (rolling
    250-day window) with Kupiec POF and the Basel traffic light.
    """
    result = ValidationResult(model_id, "VaR Backtesting (out-of-sample, 99%)")
    findings, recommendations = [], []
    try:
        bt = portfolio_backtest(portfolio_data)
    except Exception as e:
        result.findings = [f"FAIL: backtest could not run - {e}"]
        return result

    score = 1.0
    findings.append(f"Data: {bt['data']}; period {bt['period']}; window {bt['window']} days; confidence 99%.")
    for name, r in bt["methods"].items():
        verdict = {"GREEN": "PASS", "YELLOW": "WARN", "RED": "FAIL"}.get(r["basel_zone"], "WARN")
        findings.append(
            f"{verdict}: {name} - {r['exceptions']} exceptions in {r['test_observations']} test days "
            f"({r['exceptions_last_250']} in last 250 -> {r['basel_zone']}); Kupiec LR={r['kupiec_lr']:.3f}, "
            f"p={r['kupiec_p_value']:.3f}{' (reject at 95%)' if r['kupiec_reject_95'] else ''}."
        )
    zone = bt["overall_zone"]
    if zone == "YELLOW":
        score -= 0.3
        recommendations.append("Investigate VaR calibration (fat tails): at least one method is in the Basel yellow zone.")
    elif zone == "RED":
        score -= 0.6
        recommendations.append("At least one VaR method is in the Basel red zone - recalibrate before use.")
    if any(r["kupiec_reject_95"] for r in bt["methods"].values()):
        score -= 0.2
        recommendations.append("Kupiec POF rejects at least one method at the 95% level.")

    # CVaR/ES sanity check on the real portfolio (seeded Monte Carlo)
    try:
        from src.utils.data import load_portfolio
        from src.utils.metrics import calculate_portfolio_cvar
        cvar_result = calculate_portfolio_cvar((portfolio_data or load_portfolio())["positions"], confidence=0.95)
        cvar, var = cvar_result.get("cvar_1d", 0), cvar_result.get("var_1d", 0)
        if cvar != 0 and abs(cvar) >= abs(var):
            findings.append(
                f"PASS: Monte Carlo CVaR (${abs(cvar):,.0f}) >= VaR (${abs(var):,.0f}) at 95%; "
                f"tail ratio {cvar_result.get('tail_ratio')}x."
            )
        else:
            score -= 0.15
            findings.append("FAIL: CVaR < VaR - implementation error in tail calculation.")
    except Exception as e:
        score -= 0.1
        findings.append(f"WARN: CVaR check failed - {e}.")

    result.score = max(0.0, score)
    result.passed = zone == "GREEN" and result.score >= 0.7
    result.findings = findings
    result.recommendations = recommendations
    result.details = bt
    return result


#  Test 4: Sensitivity Analysis 

def test_sensitivity_analysis(model_id: str = "MRM-006") -> ValidationResult:
    """
    Test how VaR and risk scores respond to input perturbations.
    SR 11-7: Models should behave predictably when inputs change.
    """
    from src.utils.metrics import calculate_portfolio_var, score_position_risk

    result = ValidationResult(model_id, "Sensitivity Analysis")
    findings = []
    recommendations = []

    # Base portfolio
    base_positions = [
        {"ticker": "AAPL", "market_value": 500000, "beta": 1.2, "pnl_pct": 15, "weight_pct": 25, "analyst_rating": "Buy"},
        {"ticker": "NVDA", "market_value": 300000, "beta": 1.9, "pnl_pct": -8, "weight_pct": 15, "analyst_rating": "Hold"},
        {"ticker": "JPM",  "market_value": 400000, "beta": 0.8, "pnl_pct": 5,  "weight_pct": 20, "analyst_rating": "Buy"},
        {"ticker": "XOM",  "market_value": 200000, "beta": 1.1, "pnl_pct": 10, "weight_pct": 10, "analyst_rating": "Hold"},
    ]

    base_var = calculate_portfolio_var(base_positions)
    base_var_1d = base_var["portfolio_var_1d"]
    score = 1.0

    # Sensitivity 1: Double all betas - VaR should roughly double
    high_beta_positions = [{**p, "beta": p["beta"] * 2} for p in base_positions]
    high_beta_var = calculate_portfolio_var(high_beta_positions)["portfolio_var_1d"]
    beta_sensitivity = abs(high_beta_var) / abs(base_var_1d) if base_var_1d != 0 else 0

    if 1.5 <= beta_sensitivity <= 2.5:
        findings.append(f"PASS: Beta sensitivity - 2x beta -> {beta_sensitivity:.2f}x VaR (expected ~2x).")
    else:
        score -= 0.2
        findings.append(f"WARN: Beta sensitivity - 2x beta -> {beta_sensitivity:.2f}x VaR (unexpected ratio).")
        recommendations.append("Investigate VaR formula's linearity assumptions.")

    # Sensitivity 2: Remove highest-risk position - VaR should decrease
    reduced_positions = [p for p in base_positions if p["ticker"] != "NVDA"]
    reduced_var = calculate_portfolio_var(reduced_positions)["portfolio_var_1d"]
    if abs(reduced_var) < abs(base_var_1d):
        findings.append(f"PASS: Removal of high-beta position reduces VaR (${abs(reduced_var):,.0f} < ${abs(base_var_1d):,.0f}).")
    else:
        score -= 0.2
        findings.append("FAIL: Removing high-beta position did not reduce VaR - unexpected behavior.")

    # Sensitivity 3: Risk score monotonicity - higher beta should always -> higher risk score
    low_beta_score = score_position_risk({**base_positions[0], "beta": 0.5, "pnl_pct": 5, "weight_pct": 5})
    high_beta_score = score_position_risk({**base_positions[0], "beta": 2.5, "pnl_pct": 5, "weight_pct": 5})
    if high_beta_score["risk_score"] > low_beta_score["risk_score"]:
        findings.append(
            f"PASS: Risk score monotonicity - ?=2.5 scores {high_beta_score['risk_score']} > ?=0.5 scores {low_beta_score['risk_score']}."
        )
    else:
        score -= 0.3
        findings.append("FAIL: Risk score not monotonic in beta - model behavior is inconsistent.")
        recommendations.append("Review risk score formula to ensure beta is a strictly increasing risk factor.")

    # Sensitivity 4: PnL impact
    loss_score = score_position_risk({**base_positions[0], "beta": 1.0, "pnl_pct": -30, "weight_pct": 5})
    gain_score = score_position_risk({**base_positions[0], "beta": 1.0, "pnl_pct": 20, "weight_pct": 5})
    if loss_score["risk_score"] > gain_score["risk_score"]:
        findings.append(f"PASS: PnL sensitivity - loss position scores higher risk ({loss_score['risk_score']} > {gain_score['risk_score']}).")
    else:
        score -= 0.2
        findings.append("FAIL: PnL not properly reflected in risk score.")

    result.score = max(0.0, score)
    result.passed = result.score >= 0.75
    result.findings = findings
    result.recommendations = recommendations
    return result


#  Test 5: Held-out eval evidence (LLM components)

def test_eval_evidence(model_id: str) -> ValidationResult:
    """
    LLM components are validated by the held-out eval harness (scripts/run_eval.py),
    per provider (Bedrock Claude Haiku 4.5 / Gemini). Passes when at least one provider's
    running version holds an APPROVED record; each provider's gate result is reported.
    """
    from src.llm import config
    from src.mrm.approvals import load_approvals
    from src.mrm.model_inventory import check_approval, current_version

    result = ValidationResult(model_id, "Held-out Eval Evidence")
    approvals = load_approvals()
    records = approvals.get(model_id) or []
    gates = {}
    for prov in config.provider_order():
        cur = current_version(model_id, prov)
        gates[prov] = (cur, check_approval(model_id, cur["model_id"], cur["prompt_versions"], approvals, provider=prov))
    if not records:
        result.findings = [
            "FAIL: No held-out eval run recorded for this component (data/mrm/approvals.json has no entry).",
        ] + [f"Running version on {p}: {c['model_id']} with prompts {c['prompt_versions']}." for p, (c, _) in gates.items()]
        result.recommendations = ["Run: python scripts/run_eval.py --split heldout --provider <bedrock|gemini>"]
        return result
    best = None
    for prov, (cur, gate) in gates.items():
        rec = gate.get("record") or {}
        metrics = rec.get("metrics") or {}
        detail = (f" (eval run {rec.get('evidence')}: pass rate {metrics.get('pass_rate')}, mean groundedness "
                  f"{metrics.get('mean_groundedness')}, n={metrics.get('n_items')})" if rec else "")
        result.findings.append(("PASS: " if gate["allowed"] else "FAIL: ") + gate["reason"] + detail)
        if gate["allowed"] and (best is None or rec.get("status") == "APPROVED"):
            best = rec
    result.score = float(((best or {}).get("metrics") or {}).get("pass_rate") or 0.0)
    result.passed = bool(best) and best.get("status") == "APPROVED"
    result.details = {"approval": best, "providers": {p: {"current_version": c, "gate": {k: v for k, v in g.items()
                                                                                         if k != "record"}}
                                                      for p, (c, g) in gates.items()}}
    if not best:
        result.recommendations = ["Run: python scripts/run_eval.py --split heldout --provider <bedrock|gemini>"]
    return result


#  Full Validation Suite

def run_validation_suite(model_id: str, portfolio_data: Optional[dict] = None) -> dict:
    """
    Run the full SR 11-7 validation suite for a given model.
    Returns a validation report with overall pass/fail and all test results.
    """
    tests = [
        test_conceptual_soundness(model_id),
        test_data_quality(model_id, portfolio_data),
    ]

    if model_id == "MRM-006":
        tests.append(test_var_backtesting(model_id, portfolio_data))
        tests.append(test_sensitivity_analysis(model_id))
    else:
        tests.append(test_eval_evidence(model_id))

    overall_score = statistics.mean(t.score for t in tests)
    all_passed = all(t.passed for t in tests)

    # Determine validation outcome
    if overall_score >= 0.85 and all_passed:
        recommended_status = ValidationStatus.APPROVED
    elif overall_score >= 0.65:
        recommended_status = ValidationStatus.CONDITIONAL
    else:
        recommended_status = ValidationStatus.FAILED

    # LLM components can never be recommended above what the held-out eval earned.
    evidence = next((t for t in tests if t.test_name == "Held-out Eval Evidence"), None)
    if evidence is not None and not evidence.passed:
        rec = (evidence.details or {}).get("approval") or {}
        recommended_status = (ValidationStatus.CONDITIONAL if rec.get("status") == "CONDITIONAL"
                              else ValidationStatus.PENDING)

    return {
        "model_id": model_id,
        "validation_date": datetime.today().isoformat(),
        "overall_score": round(overall_score, 4),
        "all_tests_passed": all_passed,
        "recommended_status": recommended_status.value,
        "tests": [t.to_dict() for t in tests],
        "summary": (
            f"Model {model_id} validation: overall score {overall_score:.1%}. "
            f"Recommended status: {recommended_status.value}. "
            f"{'All tests passed.' if all_passed else 'Some tests failed - see findings.'}"
        ),
    }


def run_all_validations(portfolio_data: Optional[dict] = None) -> dict:
    """Run the suite for every inventory model. Returns {model_id: report}."""
    inv = get_inventory()
    return {mid: run_validation_suite(mid, portfolio_data) for mid in inv.models.keys()}


if __name__ == "__main__":
    for mid, report in run_all_validations().items():
        print(f"\n{'='*50}")
        print(f"Model: {mid} | Score: {report['overall_score']:.1%} | Status: {report['recommended_status']}")
        for test in report["tests"]:
            status = "[PASS]" if test["passed"] else "[FAIL]"
            print(f"  {status} {test['test_name']}: {test['score']:.1%}")
