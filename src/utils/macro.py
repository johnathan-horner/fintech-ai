"""
FinTech AI - macro derivations and the regime rule (deterministic).

Everything an agent might otherwise compute itself (spreads, month-over-month
changes, regime flags) is computed here so prompts can contain the numbers and
forbid arithmetic.
"""

from __future__ import annotations

from typing import Dict, List

MACRO_FIELDS = (
    "Federal Funds Rate", "CPI YoY", "Core PCE", "10Y Treasury Yield", "2Y Treasury Yield",
    "Unemployment Rate", "GDP Growth QoQ", "ISM Manufacturing PMI", "Consumer Confidence Index", "VIX",
)


def macro_derived(macro: List[Dict]) -> Dict:
    rows = sorted(macro, key=lambda m: m["date"])
    last, prev = rows[-1], rows[-2]
    spread_pct = round(last["10Y Treasury Yield"] - last["2Y Treasury Yield"], 2)
    changes = {k: round(last[k] - prev[k], 2) for k in MACRO_FIELDS if k in last and k in prev}
    return {
        "date": last["date"],
        "previous_date": prev["date"],
        "latest": {k: last[k] for k in MACRO_FIELDS if k in last},
        "spread_10y_2y_pct": spread_pct,
        "spread_10y_2y_bps": round(spread_pct * 100, 1),
        "curve_inverted": spread_pct < 0,
        "mom_changes": changes,
        "pmi_expansion": last["ISM Manufacturing PMI"] > 50,
        "vix_regime": "High" if last["VIX"] >= 25 else "Medium" if last["VIX"] >= 18 else "Low",
        "cpi_trend": "Rising" if changes.get("CPI YoY", 0) > 0 else "Falling" if changes.get("CPI YoY", 0) < 0 else "Stable",
    }


def market_regime(macro: List[Dict]) -> Dict:
    d = macro_derived(macro)
    votes = sum([d["vix_regime"] == "High", d["curve_inverted"], not d["pmi_expansion"]])
    regime = "Risk-Off" if votes >= 2 else "Risk-On" if votes == 0 else "Transitional"
    return {
        **d,
        "regime": regime,
        "risk_off_votes": votes,
        "rule": "Risk-Off if >= 2 of {VIX >= 25, 10Y-2Y < 0, PMI <= 50}; Risk-On if none; else Transitional",
    }
