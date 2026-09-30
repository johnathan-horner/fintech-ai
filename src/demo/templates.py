"""
FinTech AI - deterministic demo-mode outputs.

When live AI is unavailable (no key, session cap, governance block, 429,
error) the UI and API show these instead. They are templates filled ONLY with
values computed from the bundled data by src/utils/metrics.py, and they return
the source records they used so the deterministic numeric check can verify
every number.
"""

from __future__ import annotations

import re
from typing import List, Optional

from src.agents.context import (
    action_items,
    latest_macro_source,
    market_sources,
    portfolio_sources,
    risk_sources,
)
from src.rag.document_loader import load_all_documents
from src.utils import metrics
from src.utils.data import load_earnings, load_macro, load_market_data, load_portfolio

DEMO_LABEL = "deterministic template over the bundled data (no LLM call)"


def _money(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def _docs_by_id() -> dict:
    return {d.metadata["source_id"]: {"source_id": d.metadata["source_id"], "text": d.page_content}
            for d in load_all_documents()}


def _latest_macro() -> dict:
    return sorted(load_macro(), key=lambda m: m["date"])[-1]


def _macro_sentence(m: dict) -> str:
    return (f"Latest macro reading ({m['date']}): Federal Funds Rate {m['Federal Funds Rate']}%, CPI YoY "
            f"{m['CPI YoY']}%, 10Y Treasury {m['10Y Treasury Yield']}% vs 2Y {m['2Y Treasury Yield']}%, "
            f"VIX {m['VIX']}, ISM Manufacturing PMI {m['ISM Manufacturing PMI']}.")


# -- Agents -------------------------------------------------------------------

def demo_market_report(portfolio: Optional[dict] = None) -> dict:
    portfolio = portfolio or load_portfolio()
    k = metrics.portfolio_kpis(portfolio)
    held = sorted(p["ticker"] for p in portfolio["positions"])
    sig = metrics.technical_signals(load_market_data(), held)
    sectors = "; ".join(
        f"{name} {s['weight_pct']}% (avg P&L {s['avg_pnl_pct']}%; {', '.join(s['tickers'])})"
        for name, s in k["sector_weights"].items()
    )
    flagged = [(t, s) for t, s in sorted(sig.items()) if s["status"] != "OK"]
    tech_lines = "\n".join(
        f"- **{t}** [{s['status']}]: RSI-14 {s['rsi_14']}, SMA-50 {s['sma_50']} vs SMA-200 {s['sma_200']}, "
        f"MACD signal {s['macd_signal']}"
        for t, s in flagged[:6]
    ) or "- No held position shows an RSI extreme or SMA-50 below SMA-200."
    latest = {}
    for e in load_earnings():
        if e["ticker"] in held and (e["ticker"] not in latest or e["date"] > latest[e["ticker"]]["date"]):
            latest[e["ticker"]] = e
    beats = sorted(t for t, e in latest.items() if e["beat_miss"] == "beat")
    misses = sorted(t for t, e in latest.items() if e["beat_miss"] == "miss")
    text = (
        f"**Sector exposure.** {sectors}. Technology weight is {k['tech_weight_pct']}% of "
        f"{_money(k['total_market_value'])}.\n\n"
        f"**Technical signals** ({len(flagged)} of {len(sig)} held tickers flagged):\n{tech_lines}\n\n"
        f"**Earnings (latest quarter per holding).** Beat: {', '.join(beats) or 'none'}. "
        f"Missed: {', '.join(misses) or 'none'}.\n\n"
        f"**Macro.** {_macro_sentence(_latest_macro())}"
    )
    return {"text": text, "sources": market_sources(portfolio), "label": DEMO_LABEL}


def demo_risk_report(portfolio: Optional[dict] = None) -> dict:
    portfolio = portfolio or load_portfolio()
    pos = portfolio["positions"]
    var = metrics.calculate_portfolio_var(pos, 0.95)
    cvar = metrics.calculate_portfolio_cvar(pos, 0.95)
    stress = metrics.run_stress_tests(pos)
    conc = metrics.concentration_check(portfolio)
    scored = sorted((metrics.score_position_risk(p) for p in pos), key=lambda r: -r["risk_score"])[:3]
    worst2 = sorted(stress["scenarios"].values(), key=lambda s: s["portfolio_pnl"])[:2]
    top = "\n".join(f"- **{r['ticker']}** {r['risk_score']}/100 {r['risk_level']}: {'; '.join(r['flags'])}" for r in scored)
    breaches = ", ".join(f"{b['sector']} {b['weight_pct']}%" for b in conc["sector_breaches"]) or "none"
    names = ", ".join(f"{b['ticker']} {b['weight_pct']}%" for b in conc["single_name_breaches"]) or "none"
    m = _latest_macro()
    text = (
        f"**VaR / CVaR.** 1-day 95% parametric VaR {_money(abs(var['portfolio_var_1d']))} (weighted beta "
        f"{var['weighted_beta']}); Monte Carlo 1-day CVaR {_money(abs(cvar['cvar_1d']))} vs MC VaR "
        f"{_money(abs(cvar['var_1d']))}, tail ratio {cvar['tail_ratio']} ({cvar['tail_ratio_interpretation']}). "
        f"CVaR risk level: {cvar['risk_level']}.\n\n"
        f"**Stress tests.** Worst: {worst2[0]['scenario_name']} {_money(worst2[0]['portfolio_pnl'])} "
        f"({worst2[0]['portfolio_pnl_pct']}%), then {worst2[1]['scenario_name']} "
        f"({worst2[1]['portfolio_pnl_pct']}%).\n\n"
        f"**Highest risk scores:**\n{top}\n\n"
        f"**Concentration** (limits: sector {conc['sector_limit_pct']}%, single name "
        f"{conc['single_name_limit_pct']}%): sector breaches {breaches}; single-name breaches {names}.\n\n"
        f"**Macro overlay.** VIX {m['VIX']} and Federal Funds Rate {m['Federal Funds Rate']}% as of {m['date']}."
    )
    return {"text": text, "sources": risk_sources(portfolio), "label": DEMO_LABEL}


def demo_portfolio_recommendations(portfolio: Optional[dict] = None) -> dict:
    portfolio = portfolio or load_portfolio()
    items = action_items(portfolio)
    conc = metrics.concentration_check(portfolio)
    k = metrics.portfolio_kpis(portfolio)
    lines = [f"- **{i['ticker']}** ({i['risk_level']}, score {i['risk_score']}/100): {i['action']}. "
             f"Flags: {'; '.join(i['flags'])}." for i in items] or ["- No CRITICAL/HIGH risk positions."]
    conc_lines = [f"- Sector **{b['sector']}** at {b['weight_pct']}% exceeds the {b['limit_pct']}% policy limit: "
                  f"review trimming ({', '.join(k['sector_weights'][b['sector']]['tickers'])})."
                  for b in conc["sector_breaches"]]
    conc_lines += [f"- **{b['ticker']}** at {b['weight_pct']}% exceeds the {b['limit_pct']}% single-name limit."
                   for b in conc["single_name_breaches"]]
    tlh = [f"- **{x['ticker']}** unrealized {_money(x['unrealized_pnl'])} ({x['pnl_pct']}%)" for x in k["loss_positions"]]
    text = (
        "**Immediate review (risk scores):**\n" + "\n".join(lines) + "\n\n"
        "**Concentration:**\n" + ("\n".join(conc_lines) or "- Within policy limits.") + "\n\n"
        "**Tax-loss harvesting candidates** (review with tax advisor):\n" + ("\n".join(tlh) or "- None.")
    )
    return {"text": text, "sources": portfolio_sources(portfolio), "label": DEMO_LABEL}


# -- Chat -----------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z.]+")


def demo_chat_answer(question: str) -> dict:
    portfolio = load_portfolio()
    docs = _docs_by_id()
    q = question.lower()
    words = {w.upper().strip(".") for w in _WORD.findall(question)}
    held = {p["ticker"]: p for p in portfolio["positions"]}
    market_tickers = {r["ticker"] for r in load_market_data()}
    k = metrics.portfolio_kpis(portfolio)

    mentioned = sorted(t for t in market_tickers if t in words)
    if mentioned:
        t = mentioned[0]
        used = [docs[s] for s in (f"position:{t}", f"risk:score:{t}", f"market:{t}") if s in docs]
        ern = sorted((e for e in load_earnings() if e["ticker"] == t), key=lambda e: e["date"])
        if ern:
            used.append(docs[f"earnings:{t}:{ern[-1]['quarter']}"])
        parts = []
        if t in held:
            p = held[t]
            r = metrics.score_position_risk(p)
            parts.append(f"{t} ({p['sector']}) is held: market value {_money(p['market_value'])}, weight "
                         f"{p['weight_pct']}%, beta {p['beta']}, unrealized P&L {_money(p['unrealized_pnl'])} "
                         f"({p['pnl_pct']}%), analyst rating {p['analyst_rating']} [position:{t}]. Risk score "
                         f"{r['risk_score']}/100 ({r['risk_level']}) [risk:score:{t}].")
        else:
            parts.append(f"{t} is not a current holding.")
        sig = metrics.technical_signals(load_market_data(), [t]).get(t)
        if sig:
            parts.append(f"Technicals as of {sig['date']}: RSI-14 {sig['rsi_14']}, SMA-50 {sig['sma_50']} vs "
                         f"SMA-200 {sig['sma_200']}, status {sig['status']} [market:{t}].")
        if ern:
            e = ern[-1]
            parts.append(f"Latest earnings {e['quarter']}: revenue ${e['revenue_actual']}B vs ${e['revenue_estimate']}B "
                         f"estimate, EPS ${e['eps_actual']} vs ${e['eps_estimate']} ({e['beat_miss']}) "
                         f"[earnings:{t}:{e['quarter']}].")
        return {"text": " ".join(parts), "sources": used, "label": DEMO_LABEL}

    if any(w in q for w in ("cvar", "expected shortfall", "tail", " var", "value at risk")) or q.startswith("var"):
        pos = portfolio["positions"]
        v = metrics.calculate_portfolio_var(pos, 0.95)
        c = metrics.calculate_portfolio_cvar(pos, 0.95)
        text = (f"1-day 95% parametric VaR is {_money(abs(v['portfolio_var_1d']))} [risk:var]. The Monte Carlo "
                f"estimate gives 1-day VaR {_money(abs(c['var_1d']))} and CVaR (Expected Shortfall) "
                f"{_money(abs(c['cvar_1d']))}, a tail ratio of {c['tail_ratio']} ({c['tail_ratio_interpretation']}) "
                f"[risk:cvar]. CVaR is the average loss on days beyond VaR.")
        return {"text": text, "sources": [docs["risk:var"], docs["risk:cvar"]], "label": DEMO_LABEL}

    if any(w in q for w in ("stress", "scenario", "crash", "crisis")):
        s = metrics.run_stress_tests(portfolio["positions"])
        ranked = sorted(s["scenarios"].items(), key=lambda kv: kv[1]["portfolio_pnl"])
        lines = "; ".join(f"{v['scenario_name']} {v['portfolio_pnl_pct']}%" for _, v in ranked)
        text = (f"Worst stress scenario: {s['summary']['worst_case_scenario']} at "
                f"{_money(s['summary']['worst_case_pnl'])} ({s['summary']['worst_case_pnl_pct']}%) [risk:stress]. "
                f"All scenarios: {lines}.")
        used = [docs["risk:stress"]] + [docs[f"risk:stress:{key}"] for key, _ in ranked]
        return {"text": text, "sources": used, "label": DEMO_LABEL}

    if any(w in q for w in ("fed", "macro", "rate", "inflation", "cpi", "vix", "yield")):
        src = latest_macro_source()
        m = _latest_macro()
        text = f"{_macro_sentence(m)} Commentary: {m['commentary']} [{src['source_id']}]"
        return {"text": text, "sources": [src], "label": DEMO_LABEL}

    if any(w in q for w in ("sector", "tech", "concentration", "weight", "allocation", "diversif")):
        conc = metrics.concentration_check(portfolio)
        lines = "; ".join(f"{n} {s['weight_pct']}%" for n, s in k["sector_weights"].items())
        text = (f"Sector weights: {lines} [sector:*]. Technology is {k['tech_weight_pct']}% versus the "
                f"{conc['sector_limit_pct']}% illustrative sector limit; largest single name is "
                f"{k['largest_position']['ticker']} at {k['largest_position']['weight_pct']}% [risk:concentration].")
        used = [docs[f"sector:{n}"] for n in k["sector_weights"]] + [docs["risk:concentration"]]
        return {"text": text, "sources": used, "label": DEMO_LABEL}

    if any(w in q for w in ("risk", "danger", "worst", "loss", "score", "hedge", "worried")):
        scored = sorted((metrics.score_position_risk(p) for p in portfolio["positions"]), key=lambda r: -r["risk_score"])[:3]
        lines = "; ".join(f"{r['ticker']} {r['risk_score']}/100 {r['risk_level']} ({', '.join(r['flags'])})" for r in scored)
        text = f"Highest risk scores: {lines} [risk:score:*]."
        used = [docs[f"risk:score:{r['ticker']}"] for r in scored]
        return {"text": text, "sources": used, "label": DEMO_LABEL}

    if "earning" in q:
        latest = {}
        for e in load_earnings():
            if e["ticker"] in held and (e["ticker"] not in latest or e["date"] > latest[e["ticker"]]["date"]):
                latest[e["ticker"]] = e
        beats = sorted(t for t, e in latest.items() if e["beat_miss"] == "beat")
        misses = sorted(t for t, e in latest.items() if e["beat_miss"] == "miss")
        text = (f"Latest quarter per holding: beat - {', '.join(beats) or 'none'}; missed - "
                f"{', '.join(misses) or 'none'} [earnings:*].")
        used = [docs[f"earnings:{t}:{e['quarter']}"] for t, e in sorted(latest.items())]
        return {"text": text, "sources": used, "label": DEMO_LABEL}

    text = (f"{k['fund_name']}: {k['n_positions']} positions, AUM {_money(k['aum'])}, value-weighted beta "
            f"{k['weighted_beta']}, technology weight {k['tech_weight_pct']}%, largest position "
            f"{k['largest_position']['ticker']} ({k['largest_position']['weight_pct']}%) [fund:summary]. "
            f"Ask about a ticker, VaR/CVaR, stress tests, sectors, macro or earnings.")
    return {"text": text, "sources": [docs["fund:summary"]], "label": DEMO_LABEL}
