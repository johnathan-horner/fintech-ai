"""
FinTech AI - agent input builders.

Each agent receives a DATA block made of source records built from the
bundled data and from metrics computed by src/utils/metrics.py. The same
records are handed to the grader, so every number an agent writes can be
checked against exactly what it was given.
"""

from __future__ import annotations

from typing import List

from src.rag.document_loader import (
    load_earnings_documents,
    load_macro_documents,
    load_market_documents,
    load_portfolio_documents,
    load_risk_documents,
)
from src.utils import metrics
from src.utils.data import load_earnings, load_macro, load_market_data, load_price_history, price_history_available
from src.utils.macro import market_regime


# -- Derived records (computed by code so agents never do arithmetic) -----------

def derived_macro_source() -> dict:
    r = market_regime(load_macro())
    ch = "; ".join(f"{k} {v:+}" for k, v in r["mom_changes"].items())
    return {"source_id": "derived:macro", "text": (
        f"Derived macro for {r['date']} (vs {r['previous_date']}): 10Y-2Y spread {r['spread_10y_2y_pct']} "
        f"percentage points ({r['spread_10y_2y_bps']} bps), curve inverted: {r['curve_inverted']}. "
        f"Month-over-month changes: {ch}. VIX regime: {r['vix_regime']}. PMI above 50: {r['pmi_expansion']}. "
        f"CPI trend: {r['cpi_trend']}. Rule-based market regime: {r['regime']}."
    )}


def derived_volume_source(portfolio: dict) -> dict:
    vs = metrics.volume_stats(load_market_data(), sorted(_held(portfolio)))
    parts = [f"{t} latest volume {v['latest_volume']:,} vs 20-day average {v['avg_volume_20d']:,} "
             f"(ratio {v['volume_ratio_vs_20d']})" for t, v in sorted(vs.items())]
    return {"source_id": "derived:volume", "text": "Volume as of latest market-data row: " + "; ".join(parts) + "."}


def derived_history_source(portfolio: dict) -> dict:
    if not price_history_available():
        return {"source_id": "derived:history", "text": "Historical risk metrics unavailable (no price history)."}
    rf = sorted(load_macro(), key=lambda m: m["date"])[-1]["Federal Funds Rate"]
    h = metrics.history_metrics(load_price_history(), portfolio["positions"], rf / 100)
    return {"source_id": "derived:history", "text": (
        f"Historical risk of the current positions over synthetic price history {h['period']}: annualized Sharpe "
        f"{h['sharpe']}, Sortino {h['sortino']} (risk-free = latest Fed funds {rf}%), annualized volatility "
        f"{h['annualized_vol_pct']}%, max drawdown {h['max_drawdown_pct']}%."
    )}


def derived_concentration_source(portfolio: dict) -> dict:
    c = metrics.concentration_excess(portfolio)
    sec = "; ".join(f"{x['sector']} {x['weight_pct']}% vs {x['limit_pct']}% limit, excess {x['excess_pct_points']} "
                    f"points (${x['excess_value']:,.2f})" for x in c["sector_excess"]) or "none"
    nm = "; ".join(f"{x['ticker']} {x['weight_pct']}% vs {x['limit_pct']}% limit, excess {x['excess_pct_points']} "
                   f"points (${x['excess_value']:,.2f})" for x in c["single_name_excess"]) or "none"
    return {"source_id": "derived:concentration", "text": (
        f"Concentration excess over policy limits - sectors: {sec}. Single names: {nm}. "
        f"Positions with unrealized losses: {c['n_loss_positions']}, total unrealized loss "
        f"-${abs(c['total_unrealized_loss']):,.2f}."
    )}


def derived_earnings_flags_source(portfolio: dict) -> dict:
    """Latest quarter per held ticker: revenue / EPS above or below estimate, computed by code."""
    latest = {}
    for e in load_earnings():
        if e["ticker"] in _held(portfolio) and (e["ticker"] not in latest or e["date"] > latest[e["ticker"]]["date"]):
            latest[e["ticker"]] = e

    def flag(actual, est):
        return "above" if actual > est else "below" if actual < est else "equal to"

    parts = [f"{t} {e['quarter']}: revenue {flag(e['revenue_actual'], e['revenue_estimate'])} estimate, EPS "
             f"{flag(e['eps_actual'], e['eps_estimate'])} estimate, reported result {e['beat_miss']}"
             for t, e in sorted(latest.items())]
    return {"source_id": "derived:earnings_vs_estimates",
            "text": "Earnings versus estimates (computed by code): " + "; ".join(parts) + "."}


def _src(doc) -> dict:
    return {"source_id": doc.metadata["source_id"], "text": doc.page_content}


def _held(portfolio: dict) -> set:
    return {p["ticker"] for p in portfolio["positions"]}


def latest_macro_source() -> dict:
    docs = load_macro_documents()
    return _src(sorted(docs, key=lambda d: d.metadata["date"])[-1])


def market_sources(portfolio: dict, provider: str = "gemini") -> List[dict]:
    """Market agent inputs. provider="bedrock" adds the explicit Haiku labels (src/agents/haiku_context.py)."""
    held = _held(portfolio)
    pdocs = load_portfolio_documents(portfolio)
    out = [_src(d) for d in pdocs if d.metadata["type"] in ("fund_summary", "sector_summary")]
    out += [_src(d) for d in load_market_documents() if d.metadata["ticker"] in held]
    latest = {}
    for d in load_earnings_documents():
        t = d.metadata["ticker"]
        if t in held and (t not in latest or d.metadata["date"] > latest[t].metadata["date"]):
            latest[t] = d
    out += [_src(latest[t]) for t in sorted(latest)]
    out.append(latest_macro_source())
    out.append(derived_volume_source(portfolio))
    out.append(derived_earnings_flags_source(portfolio))
    out.append(derived_macro_source())
    if provider == "bedrock":
        from src.agents.haiku_context import label_agent_sources

        return label_agent_sources(out, portfolio)
    return out


def risk_sources(portfolio: dict, top_n: int = 8) -> List[dict]:
    rdocs = load_risk_documents(portfolio)
    scores = sorted(
        (metrics.score_position_risk(p) for p in portfolio["positions"]),
        key=lambda r: -r["risk_score"],
    )[:top_n]
    keep = {f"risk:score:{r['ticker']}" for r in scores}
    out = [_src(d) for d in load_portfolio_documents(portfolio) if d.metadata["type"] == "fund_summary"]
    out += [_src(d) for d in rdocs if d.metadata["type"] in ("risk_metric", "stress_scenario")]
    out += [_src(d) for d in rdocs if d.metadata["source_id"] in keep]
    out.append(latest_macro_source())
    out.append(derived_history_source(portfolio))
    out.append(derived_concentration_source(portfolio))
    out.append(derived_macro_source())
    return out


def portfolio_sources(portfolio: dict, provider: str = "gemini") -> List[dict]:
    """Portfolio agent inputs. provider="bedrock" adds the explicit Haiku labels."""
    rdocs = load_risk_documents(portfolio)
    pdocs = load_portfolio_documents(portfolio)
    flagged = {
        r["ticker"] for r in (metrics.score_position_risk(p) for p in portfolio["positions"])
        if r["risk_level"] in ("CRITICAL", "HIGH")
    }
    losers = {p["ticker"] for p in portfolio["positions"] if p["unrealized_pnl"] < 0}
    out = [_src(d) for d in pdocs if d.metadata["type"] in ("fund_summary", "sector_summary")]
    out += [_src(d) for d in rdocs if d.metadata["source_id"] == "risk:concentration"]
    out += [_src(d) for d in rdocs if d.metadata["type"] == "risk_score" and d.metadata["ticker"] in flagged]
    out += [_src(d) for d in pdocs if d.metadata["type"] == "position" and d.metadata["ticker"] in (flagged | losers)]
    out += [_src(d) for d in load_market_documents() if d.metadata["ticker"] in (flagged | losers)]
    out.append(derived_concentration_source(portfolio))
    if provider == "bedrock":
        from src.agents.haiku_context import label_agent_sources

        return label_agent_sources(out, portfolio)
    return out


def technical_alerts(portfolio: dict) -> List[str]:
    """Deterministic alerts computed from the data (not LLM output)."""
    sig = metrics.technical_signals(load_market_data(), sorted(_held(portfolio)))
    alerts = []
    for t, s in sorted(sig.items()):
        if s["status"] != "OK":
            alerts.append(f"{t} [{s['status']}]: " + "; ".join(s["signals"]))
    for p in portfolio["positions"]:
        if p.get("beta", 1.0) > 1.8:
            alerts.append(f"{p['ticker']}: beta {p['beta']} - high market sensitivity")
        if p.get("pnl_pct", 0) < -20:
            alerts.append(f"{p['ticker']}: {p['pnl_pct']}% unrealized loss")
    return alerts


def action_items(portfolio: dict) -> List[dict]:
    """Deterministic action items from score_position_risk (CRITICAL/HIGH)."""
    items = []
    for r in sorted((metrics.score_position_risk(p) for p in portfolio["positions"]), key=lambda r: -r["risk_score"]):
        if r["risk_level"] in ("CRITICAL", "HIGH"):
            items.append({
                "ticker": r["ticker"],
                "action": r["recommended_action"],
                "risk_level": r["risk_level"],
                "risk_score": r["risk_score"],
                "flags": r["flags"],
                "priority": "URGENT" if r["risk_level"] == "CRITICAL" else "HIGH",
            })
    return items
