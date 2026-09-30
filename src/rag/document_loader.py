"""
FinTech AI - Document Loader
Converts the bundled (synthetic) portfolio, market, earnings and macro JSON,
plus risk metrics computed by src/utils/metrics.py, into LangChain Documents.

Every document carries a stable ``source_id`` in its metadata (e.g.
``position:AAPL``, ``earnings:AAPL:Q1 2024``, ``risk:var``) that answers cite.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from src.rag.chunking import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, split_text
from src.utils import metrics
from src.utils.data import load_earnings, load_macro, load_market_data, load_portfolio


def _money(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def load_portfolio_documents(portfolio: dict | None = None) -> List[Document]:
    data = portfolio or load_portfolio()
    k = metrics.portfolio_kpis(data)
    docs = [Document(
        page_content=(
            f"Fund: {k['fund_name']} (portfolio {data.get('portfolio_id')}). As of: {k['as_of_date']}. "
            f"Total AUM: {_money(k['aum'])}. Number of positions: {k['n_positions']}. "
            f"Total unrealized P&L: {_money(k['total_unrealized_pnl'])}. "
            f"Value-weighted portfolio beta: {k['weighted_beta']}. Simple average beta: {k['avg_beta']}. "
            f"Largest position: {k['largest_position']['ticker']} with market value "
            f"{_money(k['largest_position']['market_value'])} ({k['largest_position']['weight_pct']}% weight). "
            f"Positions with beta above 1.5: {', '.join(k['high_beta_positions']) or 'none'}. "
            f"Technology weight: {k['tech_weight_pct']}%."
        ),
        metadata={"source_id": "fund:summary", "source": "portfolio", "type": "fund_summary"},
    )]
    for name, s in k["sector_weights"].items():
        docs.append(Document(
            page_content=(
                f"Sector: {name}. Portfolio weight: {s['weight_pct']}%. Market value: {_money(s['value'])}. "
                f"Average P&L across positions: {s['avg_pnl_pct']}%. Unrealized P&L: {_money(s['unrealized_pnl'])}. "
                f"Positions: {', '.join(s['tickers'])}."
            ),
            metadata={"source_id": f"sector:{name}", "source": "portfolio", "type": "sector_summary", "sector": name},
        ))
    for p in data["positions"]:
        docs.append(Document(
            page_content=(
                f"Position {p['ticker']} ({p['sector']}). Shares: {p['shares']:,}. Avg cost: ${p['avg_cost']}. "
                f"Current price: ${p['current_price']}. Market value: {_money(p['market_value'])}. "
                f"Unrealized P&L: {_money(p['unrealized_pnl'])} ({p['pnl_pct']}%). "
                f"Portfolio weight: {p['weight_pct']}%. Beta: {p['beta']}. "
                f"52-week high: ${p['52w_high']}, 52-week low: ${p['52w_low']}. "
                f"P/E ratio: {p['pe_ratio']}. Analyst rating: {p['analyst_rating']}."
            ),
            metadata={"source_id": f"position:{p['ticker']}", "source": "portfolio", "type": "position",
                      "ticker": p["ticker"], "sector": p["sector"]},
        ))
    return docs


def load_market_documents() -> List[Document]:
    sig = metrics.technical_signals(load_market_data())
    docs = []
    for t, s in sorted(sig.items()):
        docs.append(Document(
            page_content=(
                f"Market data {t} as of {s['date']}: close ${s['close']}, RSI-14 {s['rsi_14']}, "
                f"SMA-50 {s['sma_50']}, SMA-200 {s['sma_200']}, MACD signal {s['macd_signal']}. "
                f"Price change since {s['period_start']}: {s['period_return_pct']}%. "
                f"Signals: {'; '.join(s['signals'])}. Technical status: {s['status']}."
            ),
            metadata={"source_id": f"market:{t}", "source": "market_data", "type": "technicals", "ticker": t},
        ))
    return docs


def load_earnings_documents() -> List[Document]:
    docs = []
    for t in load_earnings():
        docs.append(Document(
            page_content=(
                f"Earnings {t['ticker']} {t['quarter']} (reported {t['date']}). "
                f"Revenue: ${t['revenue_actual']}B (estimate: ${t['revenue_estimate']}B). "
                f"EPS: ${t['eps_actual']} (estimate: ${t['eps_estimate']}). Result: {t['beat_miss']}. "
                f"Transcript summary: {t['transcript_summary']}"
            ),
            metadata={"source_id": f"earnings:{t['ticker']}:{t['quarter']}", "source": "earnings",
                      "type": "earnings_transcript", "ticker": t["ticker"], "quarter": t["quarter"],
                      "date": t["date"]},
        ))
    return docs


def load_macro_documents() -> List[Document]:
    docs = []
    for m in load_macro():
        indicators = ", ".join(f"{k}: {v}" for k, v in m.items() if k not in ("date", "commentary"))
        docs.append(Document(
            page_content=f"Macro indicators for {m['date']} - {indicators}. Commentary: {m.get('commentary', '')}",
            metadata={"source_id": f"macro:{m['date']}", "source": "macro", "type": "macro_indicator", "date": m["date"]},
        ))
    return docs


def load_risk_documents(portfolio: dict | None = None) -> List[Document]:
    """Risk metrics computed by src/utils/metrics.py from the bundled positions."""
    data = portfolio or load_portfolio()
    pos = data["positions"]
    v95 = metrics.calculate_portfolio_var(pos, 0.95)
    v99 = metrics.calculate_portfolio_var(pos, 0.99)
    cvar = metrics.calculate_portfolio_cvar(pos, 0.95)
    stress = metrics.run_stress_tests(pos)
    conc = metrics.concentration_check(data)
    sector_breaches = ", ".join(f"{b['sector']} {b['weight_pct']}%" for b in conc["sector_breaches"]) or "none"
    name_breaches = ", ".join(f"{b['ticker']} {b['weight_pct']}%" for b in conc["single_name_breaches"]) or "none"
    docs = [
        Document(
            page_content=(
                f"Portfolio parametric VaR (weighted beta {v95['weighted_beta']} x assumed 1.2% market vol): "
                f"1-day 95% VaR {_money(abs(v95['portfolio_var_1d']))} ({v95['portfolio_var_1d_pct']}% of value), "
                f"10-day 95% VaR {_money(abs(v95['portfolio_var_10d']))}; "
                f"1-day 99% VaR {_money(abs(v99['portfolio_var_1d']))}. Risk level: {v95['risk_level']}."
            ),
            metadata={"source_id": "risk:var", "source": "metrics", "type": "risk_metric"},
        ),
        Document(
            page_content=(
                f"Portfolio Monte Carlo CVaR / Expected Shortfall at 95% ({cvar['n_simulations']} seeded simulations, "
                f"fat-tail mixture): 1-day VaR {_money(abs(cvar['var_1d']))}, 1-day CVaR {_money(abs(cvar['cvar_1d']))} "
                f"({cvar['cvar_1d_pct']}% of value), 10-day CVaR {_money(abs(cvar['cvar_10d']))}. "
                f"Tail ratio CVaR/VaR: {cvar['tail_ratio']} ({cvar['tail_ratio_interpretation']}). "
                f"CVaR risk level: {cvar['risk_level']}."
            ),
            metadata={"source_id": "risk:cvar", "source": "metrics", "type": "risk_metric"},
        ),
        Document(
            page_content=(
                f"Stress test summary: worst case scenario {stress['summary']['worst_case_scenario']} with portfolio P&L "
                f"{_money(stress['summary']['worst_case_pnl'])} ({stress['summary']['worst_case_pnl_pct']}%). "
                f"Scenarios tested: {stress['summary']['scenarios_tested']}. Severe or catastrophic scenarios: "
                f"{stress['summary']['catastrophic_scenarios']}."
            ),
            metadata={"source_id": "risk:stress", "source": "metrics", "type": "risk_metric"},
        ),
        Document(
            page_content=(
                f"Concentration check versus illustrative policy limits (sector {conc['sector_limit_pct']}%, "
                f"single name {conc['single_name_limit_pct']}%): largest sector weight {conc['max_sector_weight_pct']}%, "
                f"largest single-name weight {conc['max_single_name_weight_pct']}%. Sector breaches: "
                f"{sector_breaches}. "
                f"Single-name breaches: "
                f"{name_breaches}."
            ),
            metadata={"source_id": "risk:concentration", "source": "metrics", "type": "risk_metric"},
        ),
    ]
    for key, s in stress["scenarios"].items():
        worst = ", ".join(f"{w['ticker']} {_money(w['pnl_impact'])}" for w in s["worst_5_positions"])
        docs.append(Document(
            page_content=(
                f"Stress scenario {s['scenario_name']}: market shock {s['market_shock_pct']}%, portfolio P&L "
                f"{_money(s['portfolio_pnl'])} ({s['portfolio_pnl_pct']}%), severity {s['severity']}. "
                f"Worst positions: {worst}."
            ),
            metadata={"source_id": f"risk:stress:{key}", "source": "metrics", "type": "stress_scenario"},
        ))
    for p in pos:
        r = metrics.score_position_risk(p)
        docs.append(Document(
            page_content=(
                f"Risk score for {r['ticker']}: {r['risk_score']}/100, level {r['risk_level']}. "
                f"Flags: {'; '.join(r['flags']) or 'none'}. Recommended action: {r['recommended_action']}."
            ),
            metadata={"source_id": f"risk:score:{r['ticker']}", "source": "metrics", "type": "risk_score",
                      "ticker": r["ticker"]},
        ))
    return docs


def load_all_documents() -> List[Document]:
    """Load all financial documents (one Document per record)."""
    docs: List[Document] = []
    docs.extend(load_portfolio_documents())
    docs.extend(load_risk_documents())
    docs.extend(load_market_documents())
    docs.extend(load_earnings_documents())
    docs.extend(load_macro_documents())
    return docs


def load_chunks(chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> List[Document]:
    """Split every document into token-window chunks (short records -> 1 chunk)."""
    chunks = []
    for doc in load_all_documents():
        parts = split_text(doc.page_content, chunk_size, overlap)
        for i, part in enumerate(parts):
            sid = doc.metadata["source_id"] if len(parts) == 1 else f"{doc.metadata['source_id']}#{i}"
            chunks.append(Document(page_content=part, metadata={**doc.metadata, "source_id": sid, "chunk": i}))
    return chunks
