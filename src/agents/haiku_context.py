"""
FinTech AI - explicit labels and precomputed comparisons for the Claude Haiku 4.5 context.

Measured Haiku failures (heldout-20260930T1*-bedrock): window returns relabelled as "YTD",
derived differences ($ beats, bps conversions), inverted month-over-month directions and
incomplete lists. The fix is applied ONLY to the Haiku (Bedrock) prompt context:

  * the FAISS index / RAG documents and the Gemini agent inputs are unchanged, so the
    Gemini-approved versions stay valid and no index rebuild is needed;
  * every return/change field is labelled with its period / basis, and every comparison
    shown comes with a direction word and the exact difference computed by code.

The grader always receives exactly this labelled context, so it grades what Haiku saw.
"""

from __future__ import annotations

import re
from typing import List, Optional

from src.utils.data import load_earnings, load_macro, load_portfolio

RATE_FIELDS = {"Federal Funds Rate", "CPI YoY", "Core PCE", "10Y Treasury Yield", "2Y Treasury Yield",
               "Unemployment Rate", "GDP Growth QoQ"}

_MARKET_RE = re.compile(r"Market data (\S+) as of (\S+): (.*?)Price change since (\S+): (-?[\d.]+)%\.")
_POSITION_RE = re.compile(r"Unrealized P&L: (-?\$[\d,.]+) \((-?[\d.]+)%\)\.")
_FUND_RE = re.compile(r"Total unrealized P&L: (-?\$[\d,.]+)\.")
_SECTOR_RE = re.compile(r"Average P&L across positions: (-?[\d.]+)%\.")


def _direction(x: float, up: str = "up", down: str = "down", flat: str = "unchanged") -> str:
    return up if x > 0 else down if x < 0 else flat


def relabel_text(source_id: str, text: str) -> str:
    """Make the period / basis of every return field explicit (no numbers are changed)."""
    if source_id.startswith("market:"):
        def market(m):
            pct = float(m.group(5))
            return (f"Market data {m.group(1)} as of {m.group(2)}: {m.group(3)}Price change over the market-data "
                    f"window {m.group(4)} to {m.group(2)} (not year-to-date): {m.group(5)}% ({_direction(pct)}).")
        return _MARKET_RE.sub(market, text)
    if source_id.startswith("position:"):
        def position(m):
            pct = float(m.group(2))
            return (f"Unrealized P&L since average cost (not a period return): {m.group(1)} ({m.group(2)}% return "
                    f"since average cost; {_direction(pct, 'gain', 'loss', 'flat')}).")
        return _POSITION_RE.sub(position, text)
    if source_id == "fund:summary":
        return _FUND_RE.sub(lambda m: f"Total unrealized P&L since average cost: {m.group(1)}.", text)
    if source_id.startswith("sector:"):
        return _SECTOR_RE.sub(lambda m: (f"Average unrealized return since average cost across positions "
                                          f"(simple average): {m.group(1)}%."), text)
    return text


def _fmt(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".") if abs(x - round(x)) > 1e-9 else f"{x:.0f}"


def macro_change_text(date: str) -> Optional[str]:
    """Month-over-month comparison for one macro month, with direction words and exact differences."""
    rows = sorted(load_macro(), key=lambda m: m["date"])
    idx = next((i for i, m in enumerate(rows) if m["date"] == date), None)
    if idx is None or idx == 0:
        return None
    cur, prev = rows[idx], rows[idx - 1]
    parts = []
    for k, v in cur.items():
        if k in ("date", "commentary") or k not in prev:
            continue
        diff = round(v - prev[k], 2)
        unit = "percentage points" if k in RATE_FIELDS else "points"
        word = _direction(diff, "increased", "decreased", "was unchanged")
        parts.append(f"{k} {word} from {prev[k]} ({prev['date']}) to {v} ({date})"
                     + (f", a change of {_fmt(abs(diff))} {unit}" if diff else ""))
    return f"Month-over-month macro changes, {prev['date']} to {date}: " + "; ".join(parts) + "."


def earnings_surprise_text(e: dict) -> str:
    rev = round(e["revenue_actual"] - e["revenue_estimate"], 1)
    eps = round(e["eps_actual"] - e["eps_estimate"], 2)
    rev_word = "above" if rev > 0 else "below" if rev < 0 else "equal to"
    eps_word = "above" if eps > 0 else "below" if eps < 0 else "equal to"
    return (f"{e['ticker']} {e['quarter']}: revenue ${e['revenue_actual']}B was {rev_word} the ${e['revenue_estimate']}B "
            f"estimate" + (f" by ${_fmt(abs(rev))}B" if rev else "") + f"; EPS ${e['eps_actual']} was {eps_word} the "
            f"${e['eps_estimate']} estimate" + (f" by ${_fmt(abs(eps))}" if eps else "")
            + f"; reported result label: {e['beat_miss']}.")


def haiku_derived_macro() -> dict:
    last = sorted(load_macro(), key=lambda m: m["date"])[-1]["date"]
    return {"source_id": "derived:macro_changes", "text": macro_change_text(last)}


def haiku_earnings_flags(portfolio: dict) -> dict:
    held = {p["ticker"] for p in portfolio["positions"]}
    latest = {}
    for e in load_earnings():
        if e["ticker"] in held and (e["ticker"] not in latest or e["date"] > latest[e["ticker"]]["date"]):
            latest[e["ticker"]] = e
    return {"source_id": "derived:earnings_vs_estimates",
            "text": "Earnings versus estimates (computed by code): "
                    + " ".join(earnings_surprise_text(latest[t]) for t in sorted(latest))}


def label_agent_sources(sources: List[dict], portfolio: dict) -> List[dict]:
    """Haiku agent context: relabel ambiguous fields, replace comparisons with explicit versions."""
    out = []
    for s in sources:
        if s["source_id"] == "derived:earnings_vs_estimates":
            out.append(haiku_earnings_flags(portfolio))
        else:
            out.append({**s, "text": relabel_text(s["source_id"], s["text"])})
    if any(s["source_id"] == "derived:macro" for s in sources):
        out.append(haiku_derived_macro())
    return out


def portfolio_lists_text(portfolio: Optional[dict] = None) -> str:
    pf = portfolio or load_portfolio()
    pos = pf["positions"]
    losers = sorted((p for p in pos if p["unrealized_pnl"] < 0), key=lambda p: p["unrealized_pnl"])
    high_beta = sorted(p["ticker"] for p in pos if p["beta"] > 1.5)
    sells = sorted(p["ticker"] for p in pos if p["analyst_rating"] in ("Sell", "Strong Sell"))
    return ("Portfolio lists (computed by code over all positions): positions with an unrealized loss since average "
            "cost: " + ", ".join(f"{p['ticker']} ({p['pnl_pct']}%)" for p in losers)
            + f"; positions with beta above 1.5: {', '.join(high_beta)}; analyst rating Sell or Strong Sell: "
            + (", ".join(sells) or "none") + ".")


def augment_rag_sources(sources: List[dict]) -> List[dict]:
    """
    Haiku RAG context = the re-ranked sources (relabelled) plus code-computed companion records:
    month-over-month changes for each retrieved macro month, estimate comparisons for each retrieved
    earnings record, and full portfolio lists when any portfolio record was retrieved.
    """
    out, extra = [], []
    earnings = {(e["ticker"], e["quarter"]): e for e in load_earnings()}
    for s in sources:
        sid = s["source_id"]
        out.append({**s, "text": relabel_text(sid, s["text"])})
        if sid.startswith("macro:"):
            txt = macro_change_text(sid.split(":", 1)[1])
            if txt:
                extra.append({"source_id": f"derived:macro_changes:{sid.split(':', 1)[1]}", "text": txt})
        elif sid.startswith("earnings:"):
            _, t, q = sid.split(":", 2)
            if (t, q) in earnings:
                extra.append({"source_id": f"derived:earnings_vs_estimate:{t}:{q}",
                              "text": earnings_surprise_text(earnings[(t, q)])})
    if any(s["source_id"].split(":")[0] in ("fund", "sector", "position", "risk") for s in sources):
        extra.append({"source_id": "derived:portfolio_lists", "text": portfolio_lists_text()})
    for e in extra:
        e.setdefault("rerank_score", None)
        e.setdefault("faiss_score", None)
        e.setdefault("retrieved_by", "derived")
    return out + extra
