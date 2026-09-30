"""
FinTech AI - Financial Metrics Utilities
Calculates VaR, CVaR/Expected Shortfall, Sharpe Ratio, Beta,
drawdown, Sortino, Calmar, and stress test scenarios.

CVaR (Conditional Value at Risk), also known as Expected Shortfall (ES),
is the Basel III / FRTB-preferred risk measure. It answers:
"Given that we breach VaR, what is the expected loss?"
CVaR is always >= VaR and better captures tail risk in fat-tailed distributions.
"""

import math
import statistics
from typing import List, Dict, Optional, Tuple


def calculate_var(returns: List[float], confidence: float = 0.95) -> float:
    """
    Value at Risk (Historical Simulation).
    Returns the loss threshold at the given confidence level.
    e.g., VaR(0.95) = -0.032 means 95% chance of not losing more than 3.2% in a day.
    """
    if not returns:
        return 0.0
    sorted_returns = sorted(returns)
    index = int((1 - confidence) * len(sorted_returns))
    return sorted_returns[index]


def calculate_sharpe(returns: List[float], risk_free_rate: float = 0.05) -> float:
    """
    Annualized Sharpe Ratio.
    risk_free_rate: annual rate (e.g., 0.05 for 5%)
    returns: daily returns list
    """
    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess = [r - daily_rf for r in returns]
    mean_excess = statistics.mean(excess)
    std = statistics.stdev(excess)
    if std == 0:
        return 0.0
    return round((mean_excess / std) * math.sqrt(252), 3)


def calculate_beta(asset_returns: List[float], market_returns: List[float]) -> float:
    """
    Beta relative to market (S&P 500 proxy).
    """
    if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
        return 1.0
    n = len(asset_returns)
    mean_a = statistics.mean(asset_returns)
    mean_m = statistics.mean(market_returns)
    covariance = sum((a - mean_a) * (m - mean_m) for a, m in zip(asset_returns, market_returns)) / (n - 1)
    variance_m = statistics.variance(market_returns)
    if variance_m == 0:
        return 1.0
    return round(covariance / variance_m, 3)


def calculate_max_drawdown(prices: List[float]) -> float:
    """
    Maximum drawdown from peak to trough.
    Returns as a negative percentage (e.g., -0.23 = -23%).
    """
    if not prices:
        return 0.0
    max_dd = 0.0
    peak = prices[0]
    for price in prices:
        if price > peak:
            peak = price
        dd = (price - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return round(max_dd, 4)


def calculate_portfolio_var(positions: List[Dict], confidence: float = 0.95) -> Dict:
    """
    Simplified portfolio VaR using position betas and weights.
    Returns a risk summary dict.
    """
    total_value = sum(p.get("market_value", 0) for p in positions)
    if total_value == 0:
        return {"portfolio_var_1d": 0, "portfolio_var_10d": 0, "risk_level": "Unknown"}

    # Simplified: weighted average beta * assumed market daily vol of 1.2%
    market_daily_vol = 0.012
    weighted_beta = sum(
        (p.get("market_value", 0) / total_value) * p.get("beta", 1.0)
        for p in positions
    )
    portfolio_daily_vol = weighted_beta * market_daily_vol

    # Normal distribution VaR approximation
    z_score = 1.645 if confidence == 0.95 else 2.326  # 99%
    var_1d = round(-z_score * portfolio_daily_vol * total_value, 2)
    var_10d = round(var_1d * math.sqrt(10), 2)

    risk_level = (
        "HIGH" if abs(var_1d) / total_value > 0.02
        else "MEDIUM" if abs(var_1d) / total_value > 0.01
        else "LOW"
    )

    return {
        "portfolio_var_1d": var_1d,
        "portfolio_var_10d": var_10d,
        "portfolio_var_1d_pct": round(abs(var_1d) / total_value * 100, 3),
        "weighted_beta": round(weighted_beta, 3),
        "risk_level": risk_level,
        "confidence": confidence,
    }


def calculate_cvar(returns: List[float], confidence: float = 0.95) -> float:
    """
    Conditional Value at Risk (CVaR) / Expected Shortfall (ES).
    Basel III / FRTB preferred risk measure over VaR.

    Answers: "Given that losses exceed VaR, what is the average loss?"
    CVaR >= VaR always. Better captures fat-tail and jump risk.

    Args:
        returns: list of daily return floats (e.g., -0.03 = -3% day)
        confidence: confidence level (0.95 or 0.99)

    Returns:
        CVaR as a negative float (e.g., -0.048 = expected 4.8% loss in tail)
    """
    if len(returns) < 10:
        return 0.0
    sorted_returns = sorted(returns)
    cutoff_index = int((1 - confidence) * len(sorted_returns))
    if cutoff_index == 0:
        cutoff_index = 1
    tail_losses = sorted_returns[:cutoff_index]
    return round(statistics.mean(tail_losses), 6)


def calculate_portfolio_cvar(
    positions: List[Dict],
    confidence: float = 0.95,
    n_simulations: int = 10000,
    random_seed: int = 42,
) -> Dict:
    """
    Portfolio CVaR using Monte Carlo simulation.
    Simulates correlated daily P&L across all positions and computes ES.

    Improvement over parametric VaR:
    - Does not assume normal distribution
    - Captures non-linear position interactions
    - Reports both VaR and CVaR side-by-side for Basel III alignment

    Args:
        positions: list of position dicts with market_value and beta
        confidence: confidence level (0.95 default, use 0.975 for FRTB)
        n_simulations: number of Monte Carlo paths
        random_seed: for reproducibility

    Returns:
        dict with var_1d, cvar_1d, var_10d, cvar_10d, tail_ratio, risk_level
    """
    import random
    random.seed(random_seed)

    total_value = sum(p.get("market_value", 0) for p in positions)
    if total_value == 0 or not positions:
        return {
            "var_1d": 0, "cvar_1d": 0,
            "var_10d": 0, "cvar_10d": 0,
            "tail_ratio": 0, "risk_level": "Unknown",
            "method": "Monte Carlo",
        }

    # Market daily vol with fat-tail adjustment (t-distribution approximation)
    # Using degrees of freedom = 4 to model equity fat tails
    market_daily_vol = 0.012

    # Simulate portfolio daily P&L
    pnl_simulations = []
    for _ in range(n_simulations):
        # Draw a correlated market shock (shared across positions)
        # Use Box-Muller with a small uniform mixture for fat tails
        u = random.random()
        if u < 0.05:
            # Fat tail event: draw from a wider distribution
            market_shock = random.gauss(0, market_daily_vol * 2.5)
        else:
            market_shock = random.gauss(0, market_daily_vol)

        # Portfolio P&L = sum of position-level P&L
        portfolio_pnl = 0.0
        for p in positions:
            beta = p.get("beta", 1.0)
            value = p.get("market_value", 0)
            # Position-specific idiosyncratic shock (smaller)
            idio_shock = random.gauss(0, market_daily_vol * 0.5)
            position_return = beta * market_shock + idio_shock
            portfolio_pnl += position_return * value

        pnl_simulations.append(portfolio_pnl)

    # Sort for percentile calculations
    pnl_simulations.sort()
    cutoff = int((1 - confidence) * n_simulations)
    cutoff = max(cutoff, 1)

    # VaR: loss at confidence percentile
    var_1d = pnl_simulations[cutoff]

    # CVaR: mean of all losses beyond VaR (the tail)
    tail = pnl_simulations[:cutoff]
    cvar_1d = statistics.mean(tail) if tail else var_1d

    # Scale to 10-day using square-root-of-time rule
    var_10d = round(var_1d * math.sqrt(10), 2)
    cvar_10d = round(cvar_1d * math.sqrt(10), 2)

    # Tail ratio: CVaR / VaR - measures how fat the tail is
    # Ratio > 1.5 indicates significant tail risk beyond VaR
    tail_ratio = round(abs(cvar_1d) / abs(var_1d), 3) if var_1d != 0 else 1.0

    # Risk level based on CVaR (more conservative than VaR-based)
    cvar_pct = abs(cvar_1d) / total_value
    risk_level = (
        "CRITICAL" if cvar_pct > 0.03
        else "HIGH" if cvar_pct > 0.02
        else "MEDIUM" if cvar_pct > 0.01
        else "LOW"
    )

    return {
        "var_1d": round(var_1d, 2),
        "cvar_1d": round(cvar_1d, 2),
        "var_1d_pct": round(abs(var_1d) / total_value * 100, 3),
        "cvar_1d_pct": round(cvar_pct * 100, 3),
        "var_10d": var_10d,
        "cvar_10d": cvar_10d,
        "tail_ratio": tail_ratio,
        "tail_ratio_interpretation": (
            "Severe tail risk - CVaR significantly exceeds VaR" if tail_ratio > 1.8
            else "Elevated tail risk" if tail_ratio > 1.4
            else "Moderate tail risk" if tail_ratio > 1.2
            else "Normal tail behavior"
        ),
        "risk_level": risk_level,
        "confidence": confidence,
        "n_simulations": n_simulations,
        "method": "Monte Carlo with fat-tail mixture",
        "regulatory_note": (
            "CVaR/ES is the Basel III FRTB-preferred risk measure. "
            f"At {confidence:.0%} confidence, expected loss in the worst {(1-confidence)*100:.0f}% "
            f"of days is ${abs(cvar_1d):,.0f} (CVaR) vs ${abs(var_1d):,.0f} (VaR). "
            f"Tail ratio of {tail_ratio:.2f}x indicates "
            f"{'material' if tail_ratio > 1.4 else 'moderate'} tail risk beyond VaR."
        ),
    }


def calculate_sortino(returns: List[float], risk_free_rate: float = 0.05) -> float:
    """
    Sortino Ratio - like Sharpe but only penalizes downside volatility.
    Better for asymmetric return distributions (hedge funds).
    Sortino > 1.0 is considered good; > 2.0 is excellent.
    """
    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess = [r - daily_rf for r in returns]
    mean_excess = statistics.mean(excess)
    downside = [r for r in excess if r < 0]
    if not downside:
        return float("inf")
    downside_std = math.sqrt(sum(r ** 2 for r in downside) / len(downside))
    if downside_std == 0:
        return 0.0
    return round((mean_excess / downside_std) * math.sqrt(252), 3)


def calculate_calmar(annualized_return: float, max_drawdown: float) -> float:
    """
    Calmar Ratio = Annualized Return / |Max Drawdown|.
    Measures return per unit of drawdown risk. Popular in hedge funds.
    Calmar > 1.0 is considered acceptable; > 3.0 is excellent.
    """
    if max_drawdown == 0:
        return 0.0
    return round(annualized_return / abs(max_drawdown), 3)


#  Stress Testing 

STRESS_SCENARIOS = {
    "2008_gfc": {
        "name": "2008 Global Financial Crisis",
        "description": "Lehman Brothers collapse - peak equity drawdown ~-50% over 17 months",
        "market_shock": -0.40,        # -40% market return (acute phase)
        "vol_multiplier": 3.5,        # VIX spiked to ~80
        "beta_amplifier": 1.3,        # Betas rose during crisis (correlation->1)
        "credit_spread_bps": 500,
        "duration_days": 30,
    },
    "covid_crash": {
        "name": "COVID-19 Market Crash (Feb-Mar 2020)",
        "description": "Fastest 30% decline in S&P 500 history - recovered in ~5 months",
        "market_shock": -0.34,
        "vol_multiplier": 2.8,
        "beta_amplifier": 1.2,
        "credit_spread_bps": 300,
        "duration_days": 23,
    },
    "rate_shock_200bps": {
        "name": "Sudden Rate Shock +200bps",
        "description": "Fed emergency hike - equity repricing, duration assets hit hardest",
        "market_shock": -0.15,
        "vol_multiplier": 1.8,
        "beta_amplifier": 1.1,
        "credit_spread_bps": 150,
        "duration_days": 5,
    },
    "tech_selloff": {
        "name": "Tech Sector Selloff (-30%)",
        "description": "Valuation compression in high-multiple tech names",
        "market_shock": -0.30,
        "vol_multiplier": 2.0,
        "beta_amplifier": 1.4,        # Tech betas amplify more
        "credit_spread_bps": 100,
        "duration_days": 60,
        "sector_filter": "Technology",
    },
    "flash_crash": {
        "name": "Flash Crash (Single Day -10%)",
        "description": "Algorithmic cascade - rapid intraday recovery but full day impact",
        "market_shock": -0.10,
        "vol_multiplier": 4.0,
        "beta_amplifier": 1.5,
        "credit_spread_bps": 50,
        "duration_days": 1,
    },
}


def run_stress_tests(positions: List[Dict], scenarios: Optional[List[str]] = None) -> Dict:
    """
    Run historical stress scenarios against the portfolio.
    Returns P&L impact for each scenario with risk decomposition.

    Args:
        positions: portfolio position list
        scenarios: list of scenario keys (default: all scenarios)

    Returns:
        dict of scenario results with P&L impact, worst positions, and risk flags
    """
    total_value = sum(p.get("market_value", 0) for p in positions)
    if total_value == 0:
        return {"error": "Empty portfolio"}

    if scenarios is None:
        scenarios = list(STRESS_SCENARIOS.keys())

    results = {}

    for scenario_key in scenarios:
        scenario = STRESS_SCENARIOS.get(scenario_key)
        if not scenario:
            continue

        market_shock = scenario["market_shock"]
        beta_amp = scenario["beta_amplifier"]
        sector_filter = scenario.get("sector_filter")

        position_impacts = []
        total_pnl = 0.0

        for p in positions:
            ticker = p.get("ticker", "?")
            sector = p.get("sector", "")
            value = p.get("market_value", 0)
            beta = p.get("beta", 1.0)

            # Apply sector-specific amplification if applicable
            effective_shock = market_shock
            if sector_filter and sector == sector_filter:
                effective_shock = market_shock * 1.5   # sector-specific stress
            elif sector_filter and sector != sector_filter:
                effective_shock = market_shock * 0.4   # other sectors less affected

            # Position P&L = value * beta (amplified) * market_shock
            position_return = beta * beta_amp * effective_shock
            position_pnl = value * position_return
            total_pnl += position_pnl

            position_impacts.append({
                "ticker": ticker,
                "sector": sector,
                "market_value": value,
                "effective_shock_pct": round(position_return * 100, 2),
                "pnl_impact": round(position_pnl, 2),
                "pnl_pct": round(position_return * 100, 2),
            })

        # Sort by worst impact
        position_impacts.sort(key=lambda x: x["pnl_impact"])
        pnl_pct = total_pnl / total_value

        results[scenario_key] = {
            "scenario_name": scenario["name"],
            "description": scenario["description"],
            "portfolio_pnl": round(total_pnl, 2),
            "portfolio_pnl_pct": round(pnl_pct * 100, 2),
            "portfolio_value_after": round(total_value + total_pnl, 2),
            "duration_days": scenario["duration_days"],
            "market_shock_pct": round(market_shock * 100, 1),
            "worst_5_positions": position_impacts[:5],
            "severity": (
                "CATASTROPHIC" if pnl_pct < -0.30
                else "SEVERE" if pnl_pct < -0.20
                else "SIGNIFICANT" if pnl_pct < -0.10
                else "MODERATE" if pnl_pct < -0.05
                else "MILD"
            ),
            "survival_flag": total_value + total_pnl > 0,
        }

    # Cross-scenario summary
    worst_scenario = min(results.items(), key=lambda x: x[1]["portfolio_pnl"])
    best_scenario = max(results.items(), key=lambda x: x[1]["portfolio_pnl"])

    return {
        "total_portfolio_value": total_value,
        "scenarios": results,
        "summary": {
            "worst_case_scenario": worst_scenario[1]["scenario_name"],
            "worst_case_pnl": worst_scenario[1]["portfolio_pnl"],
            "worst_case_pnl_pct": worst_scenario[1]["portfolio_pnl_pct"],
            "least_severe_scenario": best_scenario[1]["scenario_name"],
            "scenarios_tested": len(results),
            "catastrophic_scenarios": sum(
                1 for r in results.values() if r["severity"] in ["CATASTROPHIC", "SEVERE"]
            ),
        },
        "regulatory_note": (
            "Stress testing complements VaR and CVaR by modeling named historical scenarios "
            "rather than statistical distributions. Required under Basel III Pillar 2 ICAAP "
            "and FRTB internal models approach."
        ),
    }


def full_risk_report(positions: List[Dict], confidence: float = 0.95) -> Dict:
    """
    Master risk function - runs VaR, CVaR, and all stress scenarios together.
    Single call that produces the complete Basel III-aligned risk picture.
    """
    var_result = calculate_portfolio_var(positions, confidence)
    cvar_result = calculate_portfolio_cvar(positions, confidence)
    stress_result = run_stress_tests(positions)

    return {
        "var": var_result,
        "cvar": cvar_result,
        "stress_tests": stress_result,
        "risk_comparison": {
            "var_1d": var_result.get("portfolio_var_1d"),
            "cvar_1d": cvar_result.get("cvar_1d"),
            "worst_stress_1d": min(
                (r["portfolio_pnl"] / r["duration_days"])
                for r in stress_result["scenarios"].values()
            ) if stress_result.get("scenarios") else 0,
            "note": (
                "CVaR > VaR always. Stress tests may exceed both for named tail events. "
                "Use all three together for a complete risk picture."
            ),
        },
    }


def score_position_risk(position: Dict) -> Dict:
    """
    Scores a single position's risk level based on beta, PnL%, and weight.
    Returns risk_score (0-100) and risk_level (LOW/MEDIUM/HIGH/CRITICAL).
    """
    beta = position.get("beta", 1.0)
    pnl_pct = position.get("pnl_pct", 0)
    weight = position.get("weight_pct", 0)

    score = 0
    flags = []

    # Beta risk
    if beta > 1.8:
        score += 35
        flags.append(f"Very high beta ({beta}x) - amplified market exposure")
    elif beta > 1.3:
        score += 20
        flags.append(f"Elevated beta ({beta}x)")

    # Drawdown / PnL risk
    if pnl_pct < -25:
        score += 30
        flags.append(f"Severe drawdown ({pnl_pct}% unrealized loss)")
    elif pnl_pct < -10:
        score += 15
        flags.append(f"Notable drawdown ({pnl_pct}% unrealized loss)")

    # Concentration risk
    if weight > 15:
        score += 25
        flags.append(f"High concentration ({weight}% of portfolio)")
    elif weight > 8:
        score += 10
        flags.append(f"Moderate concentration ({weight}% of portfolio)")

    # Analyst rating risk
    rating = position.get("analyst_rating", "Hold")
    if rating in ["Sell", "Strong Sell"]:
        score += 10
        flags.append(f"Analyst consensus: {rating}")

    risk_level = (
        "CRITICAL" if score >= 60
        else "HIGH" if score >= 35
        else "MEDIUM" if score >= 15
        else "LOW"
    )

    return {
        "ticker": position.get("ticker"),
        "risk_score": min(score, 100),
        "risk_level": risk_level,
        "flags": flags,
        "recommended_action": (
            "Consider immediate exit or hedge" if risk_level == "CRITICAL"
            else "Review position size and add stop-loss" if risk_level == "HIGH"
            else "Monitor closely" if risk_level == "MEDIUM"
            else "Hold - within risk tolerance"
        ),
    }


# -- Portfolio KPIs (everything shown in the UI is computed here) ------------

# Illustrative internal policy limits used for concentration checks.
# These are stated policy parameters, not measured values.
SECTOR_LIMIT_PCT = 30.0
SINGLE_NAME_LIMIT_PCT = 15.0


def portfolio_kpis(portfolio: Dict) -> Dict:
    """Headline KPIs computed from the bundled portfolio positions."""
    positions = portfolio.get("positions", [])
    total_value = sum(p.get("market_value", 0) for p in positions)
    sectors: Dict[str, Dict] = {}
    for p in positions:
        s = sectors.setdefault(p.get("sector", "Other"), {"value": 0.0, "pnl_pcts": [], "tickers": [], "unrealized_pnl": 0.0})
        s["value"] += p.get("market_value", 0)
        s["pnl_pcts"].append(p.get("pnl_pct", 0))
        s["tickers"].append(p["ticker"])
        s["unrealized_pnl"] += p.get("unrealized_pnl", 0)
    sector_weights = {
        name: {
            "value": round(s["value"], 2),
            "weight_pct": round(s["value"] / total_value * 100, 2) if total_value else 0.0,
            "avg_pnl_pct": round(sum(s["pnl_pcts"]) / len(s["pnl_pcts"]), 2),
            "unrealized_pnl": round(s["unrealized_pnl"], 2),
            "tickers": s["tickers"],
        }
        for name, s in sorted(sectors.items(), key=lambda kv: -kv[1]["value"])
    }
    weighted_beta = (
        sum(p.get("market_value", 0) * p.get("beta", 1.0) for p in positions) / total_value if total_value else 0.0
    )
    largest = max(positions, key=lambda p: p.get("market_value", 0)) if positions else {}
    return {
        "fund_name": portfolio.get("fund_name"),
        "as_of_date": portfolio.get("as_of_date"),
        "aum": portfolio.get("aum", total_value),
        "total_market_value": round(total_value, 2),
        "n_positions": len(positions),
        "total_unrealized_pnl": round(sum(p.get("unrealized_pnl", 0) for p in positions), 2),
        "avg_beta": round(sum(p.get("beta", 1.0) for p in positions) / len(positions), 3) if positions else 0.0,
        "weighted_beta": round(weighted_beta, 3),
        "sector_weights": sector_weights,
        "tech_weight_pct": sector_weights.get("Technology", {}).get("weight_pct", 0.0),
        "largest_position": {
            "ticker": largest.get("ticker"),
            "market_value": largest.get("market_value"),
            "weight_pct": largest.get("weight_pct"),
        },
        "high_beta_positions": sorted(
            [p["ticker"] for p in positions if p.get("beta", 0) > 1.5]
        ),
        "loss_positions": sorted(
            [
                {"ticker": p["ticker"], "unrealized_pnl": p["unrealized_pnl"], "pnl_pct": p["pnl_pct"]}
                for p in positions if p.get("unrealized_pnl", 0) < 0
            ],
            key=lambda x: x["unrealized_pnl"],
        ),
    }


def concentration_check(portfolio: Dict) -> Dict:
    """Compare sector and single-name weights with the policy limits above."""
    kpis = portfolio_kpis(portfolio)
    sector_breaches = [
        {"sector": s, "weight_pct": v["weight_pct"], "limit_pct": SECTOR_LIMIT_PCT}
        for s, v in kpis["sector_weights"].items() if v["weight_pct"] > SECTOR_LIMIT_PCT
    ]
    name_breaches = [
        {"ticker": p["ticker"], "weight_pct": p["weight_pct"], "limit_pct": SINGLE_NAME_LIMIT_PCT}
        for p in portfolio.get("positions", []) if p.get("weight_pct", 0) > SINGLE_NAME_LIMIT_PCT
    ]
    return {
        "sector_limit_pct": SECTOR_LIMIT_PCT,
        "single_name_limit_pct": SINGLE_NAME_LIMIT_PCT,
        "sector_breaches": sector_breaches,
        "single_name_breaches": name_breaches,
        "max_sector_weight_pct": max((v["weight_pct"] for v in kpis["sector_weights"].values()), default=0.0),
        "max_single_name_weight_pct": max((p.get("weight_pct", 0) for p in portfolio.get("positions", [])), default=0.0),
    }


def technical_signals(market_data: List[Dict], tickers: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Latest technical indicators per ticker from market_data.json rows."""
    by_ticker: Dict[str, List[Dict]] = {}
    for row in market_data:
        if tickers is None or row["ticker"] in tickers:
            by_ticker.setdefault(row["ticker"], []).append(row)
    out = {}
    for t, rows in by_ticker.items():
        rows = sorted(rows, key=lambda r: r["date"])
        last = rows[-1]
        first = rows[0]
        signals = []
        if last["rsi_14"] >= 70:
            signals.append(f"RSI {last['rsi_14']} overbought (>=70)")
        elif last["rsi_14"] <= 30:
            signals.append(f"RSI {last['rsi_14']} oversold (<=30)")
        if last["sma_50"] < last["sma_200"]:
            signals.append(f"SMA-50 {last['sma_50']} below SMA-200 {last['sma_200']}")
        else:
            signals.append(f"SMA-50 {last['sma_50']} above SMA-200 {last['sma_200']}")
        if last["macd_signal"] < 0:
            signals.append(f"MACD signal {last['macd_signal']} negative")
        out[t] = {
            "date": last["date"],
            "close": last["close"],
            "rsi_14": last["rsi_14"],
            "sma_50": last["sma_50"],
            "sma_200": last["sma_200"],
            "macd_signal": last["macd_signal"],
            "period_start": first["date"],
            "period_return_pct": round((last["close"] / first["close"] - 1) * 100, 2),
            "signals": signals,
            "status": (
                "ALERT" if (last["rsi_14"] <= 30 or last["rsi_14"] >= 70) and last["sma_50"] < last["sma_200"]
                else "WATCH" if (last["rsi_14"] <= 30 or last["rsi_14"] >= 70 or last["sma_50"] < last["sma_200"])
                else "OK"
            ),
        }
    return out


# -- VaR estimators used by the out-of-sample backtest -----------------------

def historical_var(values: List[float], confidence: float = 0.99) -> float:
    """
    Historical-simulation VaR of a P&L series, returned as a positive loss.
    Uses the linear-interpolated (1 - confidence) empirical quantile.
    """
    if not values:
        return 0.0
    xs = sorted(values)
    q = (1 - confidence) * (len(xs) - 1)
    lo = int(math.floor(q))
    hi = min(lo + 1, len(xs) - 1)
    quantile = xs[lo] + (xs[hi] - xs[lo]) * (q - lo)
    return -quantile


def parametric_var(values: List[float], confidence: float = 0.99) -> float:
    """Gaussian (variance-covariance) VaR of a P&L series, positive loss."""
    if len(values) < 2:
        return 0.0
    mu = statistics.mean(values)
    sd = statistics.stdev(values)
    z = statistics.NormalDist().inv_cdf(1 - confidence)  # negative
    return -(mu + z * sd)


def portfolio_pnl_series(price_history: Dict, positions: List[Dict]) -> Dict:
    """
    Daily P&L of the CURRENT positions (static shares) over the price history:
        pnl_t = sum_i shares_i * (P_i,t - P_i,t-1)
    Also returns the daily portfolio return (pnl_t / value_{t-1}).
    """
    tickers = price_history["tickers"]
    n = len(price_history["dates"]) - 1
    pnl = [0.0] * n
    value_prev = [0.0] * n
    for p in positions:
        series = tickers.get(p["ticker"])
        if not series:
            continue
        prices = series["prices"]
        for t in range(n):
            pnl[t] += p["shares"] * (prices[t + 1] - prices[t])
            value_prev[t] += p["shares"] * prices[t]
    returns = [pnl[t] / value_prev[t] if value_prev[t] else 0.0 for t in range(n)]
    return {"dates": price_history["dates"][1:], "pnl": pnl, "returns": returns}


def history_metrics(price_history: Dict, positions: List[Dict], risk_free_rate: float) -> Dict:
    """
    Sharpe, Sortino, annualized volatility and max drawdown of the CURRENT
    positions held over the (synthetic) price history.
    """
    series = portfolio_pnl_series(price_history, positions)
    rets = series["returns"]
    values = []
    tickers = price_history["tickers"]
    for t in range(len(price_history["dates"])):
        values.append(sum(p["shares"] * tickers[p["ticker"]]["prices"][t] for p in positions if p["ticker"] in tickers))
    return {
        "period": f"{series['dates'][0]} to {series['dates'][-1]}",
        "n_days": len(rets),
        "risk_free_rate": risk_free_rate,
        "sharpe": calculate_sharpe(rets, risk_free_rate),
        "sortino": calculate_sortino(rets, risk_free_rate),
        "annualized_vol_pct": round(statistics.stdev(rets) * math.sqrt(252) * 100, 2),
        "max_drawdown_pct": round(calculate_max_drawdown(values) * 100, 2),
    }


# -- Volume ---------------------------------------------------------------------

def volume_stats(market_data: List[Dict], tickers: Optional[List[str]] = None, window: int = 20) -> Dict[str, Dict]:
    """Latest volume, trailing average volume and their ratio per ticker."""
    by_t: Dict[str, List[Dict]] = {}
    for r in market_data:
        if tickers is None or r["ticker"] in tickers:
            by_t.setdefault(r["ticker"], []).append(r)
    out = {}
    for t, rows in by_t.items():
        rows = sorted(rows, key=lambda r: r["date"])
        recent = rows[-window:]
        avg = statistics.mean(r["volume"] for r in recent)
        out[t] = {
            "date": rows[-1]["date"],
            "latest_volume": rows[-1]["volume"],
            "avg_volume_20d": round(avg),
            "volume_ratio_vs_20d": round(rows[-1]["volume"] / avg, 2) if avg else None,
        }
    return out


# -- Concentration excess (precomputed so agents never subtract) ----------------

def concentration_excess(portfolio: Dict) -> Dict:
    kpis = portfolio_kpis(portfolio)
    total = kpis["total_market_value"]
    sectors = [
        {"sector": s, "weight_pct": v["weight_pct"], "limit_pct": SECTOR_LIMIT_PCT,
         "excess_pct_points": round(v["weight_pct"] - SECTOR_LIMIT_PCT, 2),
         "excess_value": round((v["weight_pct"] - SECTOR_LIMIT_PCT) / 100 * total, 2)}
        for s, v in kpis["sector_weights"].items() if v["weight_pct"] > SECTOR_LIMIT_PCT
    ]
    names = [
        {"ticker": p["ticker"], "weight_pct": p["weight_pct"], "limit_pct": SINGLE_NAME_LIMIT_PCT,
         "excess_pct_points": round(p["weight_pct"] - SINGLE_NAME_LIMIT_PCT, 2),
         "excess_value": round((p["weight_pct"] - SINGLE_NAME_LIMIT_PCT) / 100 * total, 2)}
        for p in portfolio.get("positions", []) if p.get("weight_pct", 0) > SINGLE_NAME_LIMIT_PCT
    ]
    losses = [p for p in portfolio.get("positions", []) if p.get("unrealized_pnl", 0) < 0]
    return {
        "sector_excess": sectors,
        "single_name_excess": names,
        "total_unrealized_loss": round(sum(p["unrealized_pnl"] for p in losses), 2),
        "n_loss_positions": len(losses),
    }
