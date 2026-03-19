"""
FinTech AI - Document Loader
Converts portfolio, market, earnings, and macro JSON into LangChain Documents.
Mirrors EduAI's document_loader.py pattern.
"""

import json
import os
from typing import List
from langchain.schema import Document

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")


def load_portfolio_documents() -> List[Document]:
    """Convert portfolio positions into searchable LangChain Documents."""
    path = os.path.join(DATA_DIR, "portfolio.json")
    with open(path) as f:
        data = json.load(f)

    docs = []
    fund_name = data.get("fund_name", "Fund")
    aum = data.get("aum", 0)

    # Fund-level document
    docs.append(Document(
        page_content=(
            f"Fund: {fund_name}. Total AUM: ${aum:,.0f}. "
            f"As of: {data.get('as_of_date')}. "
            f"Number of positions: {len(data['positions'])}."
        ),
        metadata={"source": "portfolio", "type": "fund_summary"}
    ))

    # Per-position documents
    for p in data["positions"]:
        content = (
            f"Ticker: {p['ticker']}. Sector: {p['sector']}. "
            f"Shares: {p['shares']:,}. Avg cost: ${p['avg_cost']}. "
            f"Current price: ${p['current_price']}. "
            f"Market value: ${p['market_value']:,.0f}. "
            f"Unrealized PnL: ${p['unrealized_pnl']:,.0f} ({p['pnl_pct']}%). "
            f"Portfolio weight: {p['weight_pct']}%. "
            f"Beta: {p['beta']}. "
            f"52-week high: ${p['52w_high']}, low: ${p['52w_low']}. "
            f"P/E ratio: {p['pe_ratio']}. "
            f"Analyst rating: {p['analyst_rating']}."
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source": "portfolio",
                "ticker": p["ticker"],
                "sector": p["sector"],
                "type": "position",
            }
        ))
    return docs


def load_earnings_documents() -> List[Document]:
    """Convert earnings transcripts into LangChain Documents."""
    path = os.path.join(DATA_DIR, "earnings_transcripts.json")
    with open(path) as f:
        data = json.load(f)

    docs = []
    for t in data:
        content = (
            f"Ticker: {t['ticker']}. Quarter: {t['quarter']}. "
            f"Revenue: ${t['revenue_actual']}B (estimate: ${t['revenue_estimate']}B). "
            f"EPS: ${t['eps_actual']} (estimate: ${t['eps_estimate']}). "
            f"Result: {t['beat_miss']}. "
            f"Transcript summary: {t['transcript_summary']}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source": "earnings",
                "ticker": t["ticker"],
                "quarter": t["quarter"],
                "beat_miss": t["beat_miss"],
                "type": "earnings_transcript",
            }
        ))
    return docs


def load_macro_documents() -> List[Document]:
    """Convert macro indicators into LangChain Documents."""
    path = os.path.join(DATA_DIR, "macro_indicators.json")
    with open(path) as f:
        data = json.load(f)

    docs = []
    for m in data:
        indicators = ", ".join(
            f"{k}: {v}" for k, v in m.items()
            if k not in ["date", "commentary"]
        )
        content = (
            f"Date: {m['date']}. Macro indicators - {indicators}. "
            f"Commentary: {m.get('commentary', '')}"
        )
        docs.append(Document(
            page_content=content,
            metadata={"source": "macro", "date": m["date"], "type": "macro_indicator"}
        ))
    return docs


def load_all_documents() -> List[Document]:
    """Load and combine all financial documents."""
    docs = []
    docs.extend(load_portfolio_documents())
    docs.extend(load_earnings_documents())
    docs.extend(load_macro_documents())
    print(f"Loaded {len(docs)} total financial documents.")
    return docs
