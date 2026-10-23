"""
FinTech AI - MCP tool implementations (pure functions, no fastmcp import).

Every value returned is computed from the bundled data/*.json files,
src/utils/metrics.py and src/mrm. There is no randomness here (the Monte
Carlo CVaR in metrics.py uses a fixed seed), and outputs carry the data's
as-of date rather than wall-clock time, so the same input always gives the
same output.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils import metrics
from src.utils.data import REPO_ROOT, load_earnings, load_macro, load_market_data, load_portfolio, load_price_history

HORIZON_ROWS = {"1_day": 1, "1_week": 5, "1_month": 21, "3_month": 63}
RISK_LEVEL_CAPS = {"conservative": 0.10, "moderate": 0.15, "aggressive": 0.25}


def _positions(symbols: Optional[List[str]] = None):
    pf = load_portfolio()
    pos = pf["positions"]
    if symbols:
        wanted = {s.upper() for s in symbols}
        unknown = sorted(wanted - {p["ticker"] for p in pos})
        pos = [p for p in pos if p["ticker"] in wanted]
    else:
        unknown = []
    return pf, pos, unknown


def analyze_portfolio_risk(portfolio_symbols: Optional[List[str]] = None, confidence_level: float = 0.95) -> Dict[str, Any]:
    pf, pos, unknown = _positions(portfolio_symbols)
    if not pos:
        return {"error": "no matching positions", "unknown_symbols": unknown,
                "held_symbols": sorted(p["ticker"] for p in pf["positions"])}
    var = metrics.calculate_portfolio_var(pos, confidence_level)
    cvar = metrics.calculate_portfolio_cvar(pos, confidence_level)
    stress = metrics.run_stress_tests(pos)
    scores = sorted((metrics.score_position_risk(p) for p in pos), key=lambda r: -r["risk_score"])
    return {
        "as_of_date": pf["as_of_date"],
        "positions_analyzed": [p["ticker"] for p in pos],
        "unknown_symbols": unknown,
        "portfolio_value": round(sum(p["market_value"] for p in pos), 2),
        "parametric_var": var,
        "monte_carlo_cvar": {k: v for k, v in cvar.items() if k != "regulatory_note"},
        "stress_test_summary": stress["summary"],
        "stress_scenarios": {k: {x: s[x] for x in ("scenario_name", "portfolio_pnl", "portfolio_pnl_pct", "severity")}
                             for k, s in stress["scenarios"].items()},
        "position_risk_scores": scores,
        "data": "bundled synthetic portfolio (data/portfolio.json)",
    }


def generate_market_insights(sector: str, time_horizon: str = "1_week") -> Dict[str, Any]:
    pf = load_portfolio()
    k = metrics.portfolio_kpis(pf)
    match = next((s for s in k["sector_weights"] if s.lower() == sector.strip().lower()), None)
    rows_back = HORIZON_ROWS.get(time_horizon)
    if rows_back is None:
        return {"error": f"time_horizon must be one of {sorted(HORIZON_ROWS)}"}
    if not match:
        return {"error": f"sector not held", "held_sectors": sorted(k["sector_weights"])}
    info = k["sector_weights"][match]
    rows = load_market_data()
    by_t: Dict[str, list] = {}
    for r in rows:
        if r["ticker"] in info["tickers"]:
            by_t.setdefault(r["ticker"], []).append(r)
    perf = {}
    for t, rs in sorted(by_t.items()):
        rs.sort(key=lambda r: r["date"])
        end, start = rs[-1], rs[max(0, len(rs) - 1 - rows_back)]
        perf[t] = {"start_date": start["date"], "end_date": end["date"],
                   "return_pct": round((end["close"] / start["close"] - 1) * 100, 2)}
    sig = metrics.technical_signals(rows, info["tickers"])
    avg_ret = round(sum(v["return_pct"] for v in perf.values()) / len(perf), 2) if perf else None
    result = {
        "sector": match,
        "time_horizon": time_horizon,
        "portfolio_weight_pct": info["weight_pct"],
        "market_value": info["value"],
        "avg_unrealized_pnl_pct": info["avg_pnl_pct"],
        "tickers": info["tickers"],
        "price_performance": perf,
        "avg_price_return_pct": avg_ret,
        "technical_status": {t: {"status": s["status"], "signals": s["signals"]} for t, s in sig.items()},
        "trend": None if avg_ret is None else "up" if avg_ret > 0 else "down" if avg_ret < 0 else "flat",
        "method": "computed from data/market_data.json closes and indicators (synthetic)",
    }
    return result


def optimize_portfolio_allocation(current_positions: Dict[str, float], target_risk_level: str = "moderate") -> Dict[str, Any]:
    """
    Long-only minimum-variance allocation with a per-name cap, using the
    covariance of daily returns in data/price_history.json (synthetic).
    Solved by projected gradient descent (deterministic).
    """
    cap = RISK_LEVEL_CAPS.get(target_risk_level)
    if cap is None:
        return {"error": f"target_risk_level must be one of {sorted(RISK_LEVEL_CAPS)}"}
    ph = load_price_history()
    weights_in = {str(k).upper(): float(v) for k, v in current_positions.items()}
    symbols = list(weights_in)
    unknown = sorted(s for s in symbols if s not in ph["tickers"])
    syms = [s for s in symbols if s in ph["tickers"]]
    if len(syms) < 2:
        return {"error": "need at least two symbols with price history", "unknown_symbols": unknown}
    if cap * len(syms) < 1:
        cap = 1.0 / len(syms)
    R = np.array([ph["tickers"][s]["returns"] for s in syms])
    cov = np.cov(R) * 252
    w = np.full(len(syms), 1.0 / len(syms))
    step = 0.5 / np.linalg.norm(cov, 2)
    for _ in range(5000):
        w = _project_capped_simplex(w - step * 2 * cov @ w, cap)
    cur = np.array([weights_in[s] for s in syms])
    cur = cur / cur.sum() if cur.sum() > 0 else np.full(len(syms), 1.0 / len(syms))
    vol_cur = float(np.sqrt(cur @ cov @ cur))
    vol_opt = float(np.sqrt(w @ cov @ w))
    actions = [
        {"symbol": s, "action": "INCREASE" if w[i] > cur[i] else "DECREASE",
         "from_weight": round(float(cur[i]), 4), "to_weight": round(float(w[i]), 4)}
        for i, s in enumerate(syms) if abs(w[i] - cur[i]) > 0.02
    ]
    return {
        "method": "long-only minimum-variance (max weight per name) on synthetic price_history covariance",
        "target_risk_level": target_risk_level,
        "max_single_position": round(cap, 4),
        "unknown_symbols": unknown,
        "current_allocation": {s: round(float(cur[i]), 4) for i, s in enumerate(syms)},
        "optimized_allocation": {s: round(float(w[i]), 4) for i, s in enumerate(syms)},
        "annualized_volatility_current": round(vol_cur, 4),
        "annualized_volatility_optimized": round(vol_opt, 4),
        "rebalancing_actions": actions,
        "diversification_score": round(float(1 - np.sum(w ** 2)), 4),
    }


def _project_capped_simplex(v: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto {w : sum w = 1, 0 <= w <= cap} (bisection on the shift)."""
    lo, hi = float(v.min() - cap), float(v.max())
    for _ in range(100):
        tau = (lo + hi) / 2
        s = np.clip(v - tau, 0, cap).sum()
        if s > 1:
            lo = tau
        else:
            hi = tau
    return np.clip(v - (lo + hi) / 2, 0, cap)


def analyze_stock_fundamentals(symbol: str) -> Dict[str, Any]:
    sym = symbol.upper()
    pf = load_portfolio()
    pos = next((p for p in pf["positions"] if p["ticker"] == sym), None)
    earnings = sorted((e for e in load_earnings() if e["ticker"] == sym), key=lambda e: e["date"])
    sig = metrics.technical_signals(load_market_data(), [sym]).get(sym)
    if not pos and not earnings and not sig:
        known = sorted({r["ticker"] for r in load_market_data()})
        return {"error": f"{sym} not in bundled data", "known_symbols": known}
    out: Dict[str, Any] = {"symbol": sym, "held": pos is not None}
    if pos:
        out["position"] = {k: pos[k] for k in ("sector", "shares", "avg_cost", "current_price", "market_value",
                                               "unrealized_pnl", "pnl_pct", "weight_pct", "beta", "52w_high",
                                               "52w_low", "pe_ratio", "analyst_rating")}
        out["pct_below_52w_high"] = round((1 - pos["current_price"] / pos["52w_high"]) * 100, 2)
        out["risk_score"] = metrics.score_position_risk(pos)
    if earnings:
        out["earnings"] = [{k: e[k] for k in ("quarter", "date", "revenue_actual", "revenue_estimate", "eps_actual",
                                             "eps_estimate", "beat_miss")} for e in earnings]
        last = earnings[-1]
        out["latest_revenue_surprise_pct"] = round((last["revenue_actual"] / last["revenue_estimate"] - 1) * 100, 2)
        out["latest_eps_surprise_pct"] = (round((last["eps_actual"] / last["eps_estimate"] - 1) * 100, 2)
                                          if last["eps_estimate"] else None)
    if sig:
        out["technicals"] = sig
    out["note"] = ("Only fields present in the bundled synthetic data are reported "
                   "(no ROE/margins/leverage data exists in this repo).")
    return out


def get_market_regime_analysis() -> Dict[str, Any]:
    from src.utils.macro import market_regime

    macro = load_macro()
    r = market_regime(macro)
    return {
        "as_of": r["date"],
        "current_regime": r["regime"],
        "rule": r["rule"],
        "signals": {
            "volatility_regime": r["vix_regime"],
            "yield_curve_slope_bps_10y_2y": r["spread_10y_2y_bps"],
            "curve_inverted": r["curve_inverted"],
            "pmi_expansion": r["pmi_expansion"],
            "cpi_trend": r["cpi_trend"],
            "month_over_month_changes": r["mom_changes"],
        },
        "latest_indicators": r["latest"],
        "previous_month": r["previous_date"],
        "data": "data/macro_indicators.json (synthetic)",
    }


def generate_compliance_report(portfolio_data: Optional[Dict[str, Any]] = None, report_type: str = "daily") -> Dict[str, Any]:
    from src.mrm.governance.model_governance import get_governance_summary
    from src.mrm.model_inventory import get_inventory
    from src.mrm.monitoring.model_monitor import returns_drift
    from src.mrm.validation.model_validator import portfolio_backtest

    pf = load_portfolio()
    inv = get_inventory()
    conc = metrics.concentration_check(pf)
    bt = portfolio_backtest()
    drift = returns_drift()
    gov = get_governance_summary()
    return {
        "report_type": report_type,
        "as_of_date": pf["as_of_date"],
        "framework": "SR 11-7 Model Risk Management",
        "model_inventory": [
            {"model_id": m["model_id"], "name": m["name"], "tier": m["tier"].value,
             "validation_status": m["validation_status"].value,
             "approved_evidence": (m.get("approved_version") or {}).get("evidence")}
            for m in inv.get_all()
        ],
        "inventory_summary": inv.summary_stats(),
        "var_backtest": {"overall_zone": bt["overall_zone"], "period": bt["period"],
                         "methods": {k: {x: v[x] for x in ("exceptions", "test_observations", "basel_zone", "kupiec_lr",
                                                           "kupiec_p_value")} for k, v in bt["methods"].items()}},
        "returns_drift_psi": {"psi_value": drift.get("psi_value"), "status": drift.get("status")},
        "concentration": conc,
        "governance": {k: gov[k] for k in ("total_models", "tier1_violations", "pending_change_requests",
                                           "audit_log_entries")},
        "data": "computed from bundled synthetic data and src/mrm",
    }


def rag_pipeline_info() -> Dict[str, Any]:
    from src.llm import config
    from src.rag.chunking import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, TOKENIZER
    from src.rag.query_transform import HYDE_TOP_K
    from src.rag.reranker import RERANK_TOP_N, RERANKER_MODEL

    return {
        "pipeline_architecture": {
            "embedding_model": config.model_id("embedding"),
            "vector_store": "FAISS (faiss-cpu IndexFlatIP, cosine)",
            "chunk_size_tokens": CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
            "tokenizer": TOKENIZER,
            "retrieval_strategy": f"HyDE passage embedding top-{HYDE_TOP_K} + raw-query top-{HYDE_TOP_K} merge",
            "reranker": f"{RERANKER_MODEL} cross-encoder, top {RERANK_TOP_N}",
        },
        "llm_integration": {
            "provider_chain": config.provider_order() + ["demo"],
            "models": {p: config.all_model_ids(p) for p in config.provider_order()},
            "free_tier_label": config.free_tier_label(),
        },
        "data_sources": [
            {"name": "Portfolio positions", "file": "data/portfolio.json", "type": "synthetic"},
            {"name": "Market OHLCV + indicators", "file": "data/market_data.json", "type": "synthetic"},
            {"name": "Earnings call summaries", "file": "data/earnings_transcripts.json", "type": "synthetic"},
            {"name": "Macro indicators", "file": "data/macro_indicators.json", "type": "synthetic"},
            {"name": "Computed risk metrics", "file": "src/utils/metrics.py", "type": "computed"},
        ],
    }


def market_data_sources() -> Dict[str, Any]:
    retention_path = os.path.join(REPO_ROOT, "infra", "retention_config.json")
    with open(retention_path) as f:
        retention = json.load(f)
    md = load_market_data()
    dates = sorted({r["date"] for r in md})
    return {
        "sources": {
            "market_data": {"file": "data/market_data.json", "type": "synthetic",
                            "tickers": len({r["ticker"] for r in md}), "first_date": dates[0], "last_date": dates[-1]},
            "price_history": {"file": "data/price_history.json", "type": "synthetic (seeded generator)"},
            "macro": {"file": "data/macro_indicators.json", "type": "synthetic"},
        },
        "live_feeds": "none - no external market data client exists in this repo",
        "compliance": {
            "audit_retention_configured_in_infra": retention,
            "source": "infra/retention_config.json (read by the CDK stack)",
        },
    }
