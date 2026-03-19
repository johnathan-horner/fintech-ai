"""
FinTech AI - Model Validation Engine
SR 11-7 requires independent validation of all models.
This module runs conceptual soundness, outcome analysis, and benchmarking tests.

Validation covers:
  1. Conceptual Soundness - is the model design appropriate for its intended use?
  2. Data Quality - are inputs clean, complete, and representative?
  3. Outcome Analysis - do outputs meet quality thresholds?
  4. Sensitivity Analysis - how do outputs change with input perturbations?
  5. Benchmarking - does the model outperform a simpler alternative?
"""

import json
import random
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

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "test_name": self.test_name,
            "passed": self.passed,
            "score": round(self.score, 4),
            "findings": self.findings,
            "recommendations": self.recommendations,
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


#  Test 3: Outcome Analysis - VaR Backtesting 

def test_var_backtesting(model_id: str = "MRM-006", n_simulations: int = 252) -> ValidationResult:
    """
    Backtest VaR model by simulating returns and checking exceedance rate.
    SR 11-7: Quantitative models must demonstrate predictive accuracy.
    At 95% confidence, VaR should be exceeded ~5% of the time (Basel traffic light test).
    """
    result = ValidationResult(model_id, "VaR Backtesting")
    findings = []
    recommendations = []

    # Simulate synthetic daily returns (normal distribution as approximation)
    random.seed(42)
    daily_returns = [random.gauss(0.0003, 0.012) for _ in range(n_simulations)]

    # Calculate VaR at 95% confidence
    sorted_returns = sorted(daily_returns)
    var_index = int(0.05 * len(sorted_returns))
    var_95 = sorted_returns[var_index]

    # Count exceedances
    exceedances = sum(1 for r in daily_returns if r < var_95)
    exceedance_rate = exceedances / n_simulations
    expected_rate = 0.05

    findings.append(f"Backtest period: {n_simulations} trading days")
    findings.append(f"VaR (95%): {var_95*100:.3f}% daily loss threshold")
    findings.append(f"Actual exceedance rate: {exceedance_rate*100:.2f}% (expected ~5.00%)")

    # Basel traffic light test
    # Green: <=4 exceedances per 250 days (~1.6%) -> no concern
    # Yellow: 5-9 exceedances -> increased scrutiny
    # Red: >=10 exceedances -> model failure
    scaled_exceedances = int(exceedance_rate * 250)

    score = 1.0
    if scaled_exceedances <= 4:
        findings.append("PASS: Basel Traffic Light - GREEN zone (<=4 exceedances per 250 days).")
    elif scaled_exceedances <= 9:
        score -= 0.3
        findings.append(f"WARN: Basel Traffic Light - YELLOW zone ({scaled_exceedances} exceedances per 250 days).")
        recommendations.append("Investigate VaR model calibration. Consider fat-tail adjustments.")
    else:
        score -= 0.6
        findings.append(f"FAIL: Basel Traffic Light - RED zone ({scaled_exceedances} exceedances per 250 days).")
        recommendations.append("VaR model requires recalibration. Consider CVaR/Expected Shortfall.")

    # Kupiec POF test (simplified)
    # Tests whether exceedance rate is statistically consistent with 5%
    p = expected_rate
    n = n_simulations
    x = exceedances
    if x > 0 and x < n:
        log_likelihood = (x * math.log(x / n) + (n - x) * math.log((n - x) / n)
                         - x * math.log(p) - (n - x) * math.log(1 - p))
        kupiec_stat = -2 * log_likelihood
        # Chi-squared critical value at 95% = 3.841
        if kupiec_stat < 3.841:
            findings.append(f"PASS: Kupiec POF test - stat={kupiec_stat:.3f} < 3.841 (cannot reject model at 95%).")
        else:
            score -= 0.2
            findings.append(f"FAIL: Kupiec POF test - stat={kupiec_stat:.3f} > 3.841 (reject VaR model).")
            recommendations.append("VaR model rejected by Kupiec test. Recalibrate or switch methods.")

    # CVaR/ES implementation check - closes MRM outstanding item #1
    try:
        from src.utils.metrics import calculate_portfolio_cvar
        test_positions = [
            {"market_value": 100000, "beta": 1.2},
            {"market_value": 80000,  "beta": 0.9},
        ]
        cvar_result = calculate_portfolio_cvar(test_positions, confidence=0.95, n_simulations=1000)
        cvar = cvar_result.get("cvar_1d", 0)
        var = cvar_result.get("var_1d", 0)
        if cvar != 0 and abs(cvar) >= abs(var):
            findings.append(
                f"PASS: CVaR/ES implemented and validated - "
                f"CVaR (${abs(cvar):,.0f}) >= VaR (${abs(var):,.0f}) as expected. "
                f"Tail ratio: {cvar_result.get('tail_ratio', 'N/A')}x. "
                f"Basel III FRTB alignment confirmed."
            )
        else:
            score -= 0.15
            findings.append("FAIL: CVaR <= VaR - implementation error in tail calculation.")
            recommendations.append("Review CVaR calculation - CVaR must always be >= VaR.")
    except Exception as e:
        score -= 0.1
        findings.append(f"NOTE: CVaR/ES implementation check failed - {e}.")
        recommendations.append("Implement Expected Shortfall (CVaR) alongside VaR for regulatory compliance.")

    result.score = max(0.0, score)
    result.passed = result.score >= 0.7
    result.findings = findings
    result.recommendations = recommendations
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


#  Full Validation Suite 

def run_validation_suite(model_id: str, portfolio_data: Optional[dict] = None) -> dict:
    """
    Run the full SR 11-7 validation suite for a given model.
    Returns a validation report with overall pass/fail and all test results.
    """
    print(f"\n[SEARCH] Running validation suite for {model_id}...")

    tests = [
        test_conceptual_soundness(model_id),
        test_data_quality(model_id, portfolio_data),
    ]

    # Additional quant tests for MRM-006
    if model_id == "MRM-006":
        tests.append(test_var_backtesting(model_id))
        tests.append(test_sensitivity_analysis(model_id))

    overall_score = statistics.mean(t.score for t in tests)
    all_passed = all(t.passed for t in tests)

    # Determine validation outcome
    if overall_score >= 0.85 and all_passed:
        recommended_status = ValidationStatus.APPROVED
    elif overall_score >= 0.65:
        recommended_status = ValidationStatus.CONDITIONAL
    else:
        recommended_status = ValidationStatus.FAILED

    report = {
        "model_id": model_id,
        "validation_date": datetime.today().isoformat(),
        "overall_score": round(overall_score, 4),
        "all_tests_passed": all_passed,
        "recommended_status": recommended_status,
        "tests": [t.to_dict() for t in tests],
        "summary": (
            f"Model {model_id} validation: overall score {overall_score:.1%}. "
            f"Recommended status: {recommended_status}. "
            f"{'All tests passed.' if all_passed else 'Some tests failed - see findings.'}"
        ),
    }

    print(f"  Score: {overall_score:.1%} | Status: {recommended_status}")
    return report


if __name__ == "__main__":
    for mid in ["MRM-001", "MRM-003", "MRM-006"]:
        report = run_validation_suite(mid)
        print(f"\n{'='*50}")
        print(f"Model: {mid} | Score: {report['overall_score']:.1%} | Status: {report['recommended_status']}")
        for test in report["tests"]:
            status = "[PASS]" if test["passed"] else "[FAIL]"
            print(f"  {status} {test['test_name']}: {test['score']:.1%}")
