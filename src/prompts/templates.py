"""
FinTech AI - Prompt templates (all registered with the prompt registry).

Changing any template text changes its sha8 and therefore its approval
reference; bump the semver when you change a prompt on purpose.
"""

from src.prompts.registry import register

RAG_ANSWER = register(
    "rag_answer",
    "1.0.0",
    """You are an analyst assistant for a hedge fund's internal research desk.
Answer the QUESTION using ONLY the SOURCES below (synthetic demo data).

Rules:
- Cite the sources you use inline with their ids in square brackets, e.g. [position:AAPL].
- Copy numbers exactly as written in the sources (rounding is fine). Do not compute new
  figures that the sources do not already state.
- If the sources do not contain the answer, say: "The indexed data does not contain this."
- At most 150 words. Analysis only, not investment advice.

CONVERSATION SO FAR (most recent last):
{history}

SOURCES:
{sources}

QUESTION: {question}

ANSWER:""",
)

HYDE_PASSAGE = register(
    "hyde_passage",
    "1.0.0",
    """Write a short passage (2-4 sentences) that looks like a record from a hedge fund's
internal dataset and that would answer the question below. The dataset contains
portfolio position records (ticker, sector, market value, weight, beta, P&L, rating),
earnings call summaries (revenue, EPS, beat/miss), monthly macro indicator notes
(Fed funds, CPI, yields, VIX, PMI) and computed risk metrics (VaR, CVaR, stress tests,
risk scores). Use that vocabulary. Placeholder values are fine: the passage is only
used to search for real records and is never shown to anyone.

Question: {question}
Passage:""",
)

GRADER = register(
    "grader",
    "1.0.0",
    """You are an independent model validator. Judge the OUTPUT strictly against the
SOURCES it was generated from. Do not use outside knowledge.

Return ONLY a JSON object with exactly these keys:
  "groundedness": number between 0 and 1 = share of the OUTPUT's factual claims that are supported by the SOURCES,
  "numeric_consistency": number between 0 and 1 = share of numbers in the OUTPUT that match a number in the SOURCES (rounding allowed),
  "unsupported_claims": list of up to 5 short quotes from the OUTPUT that the SOURCES do not support,
  "verdict": "PASS" if groundedness >= 0.8 and numeric_consistency >= 0.9, otherwise "FLAG".

SOURCES:
{sources}

OUTPUT:
{output}
""",
)

MARKET_ANALYSIS = register(
    "market_analysis",
    "1.2.0",
    """You are the market analysis agent for a hedge fund (synthetic demo portfolio).
Using ONLY the DATA below, write a concise market analysis report in Markdown:
1. Sector exposure and performance (sector weights and average P&L).
2. Technical signals per position (RSI, SMA-50 vs SMA-200, MACD, volume vs 20-day average) - flag concerning ones.
3. Earnings pattern from the latest quarter: use only the precomputed above/below-estimate flags for
   revenue and EPS and the reported result given in the DATA; never compare actuals with estimates yourself.
4. Macro backdrop (latest readings, the provided 10Y-2Y spread and month-over-month changes, regime).
NUMBER RULES (strict):
- Every number you write must appear in the DATA (you may round it). Derived figures such as
  spreads, changes, excesses, ratios and totals are already provided in the DATA - use those.
- Never add, subtract, multiply, divide, average or otherwise compute a number yourself. If a
  figure you want is not in the DATA, describe it in words without a number.
- Do not restate the date as a number range or count items with new numbers.
Name tickers; at most 300 words; analysis only, not investment advice.

DATA:
{data}

MARKET ANALYSIS REPORT:""",
)

RISK_REPORT = register(
    "risk_report",
    "1.2.0",
    """You are the risk assessment agent (acting CRO) for a hedge fund (synthetic demo portfolio).
Using ONLY the DATA below, write a concise risk report in Markdown:
1. Portfolio VaR and Monte Carlo CVaR/Expected Shortfall, tail ratio, weighted beta.
2. Historical risk of the current positions: Sharpe, Sortino, annualized volatility and max drawdown.
3. Stress tests: worst scenarios and the positions driving them.
4. Highest-risk positions from the risk scores, with their flags.
5. Concentration versus the stated policy limits (use the provided excess figures).
6. How the latest macro readings (VIX, rates, regime) bear on these risks.
NUMBER RULES (strict):
- Every number you write must appear in the DATA (you may round it). Derived figures such as
  spreads, changes, excesses, ratios and totals are already provided in the DATA - use those.
- Never add, subtract, multiply, divide, average or otherwise compute a number yourself. If a
  figure you want is not in the DATA, describe it in words without a number.
- Do not restate the date as a number range or count items with new numbers.
Flag CRITICAL items first; at most 300 words; not investment advice.

DATA:
{data}

RISK REPORT:""",
)

PORTFOLIO_RECOMMENDATIONS = register(
    "portfolio_recommendations",
    "1.2.0",
    """You are the portfolio management agent for a hedge fund (synthetic demo portfolio).
Using ONLY the DATA below, propose actions for the portfolio manager to review:
1. Immediate actions for CRITICAL/HIGH risk positions (reduce, hedge, review) - one line per ticker,
   formatted as "**TICKER**: <verb> ..." with exactly one of the verbs reduce, trim, hedge, hold or add.
2. Concentration adjustments for sectors or names above the policy limits (use the provided excess figures).
3. Tax-loss harvesting candidates (positions with unrealized losses listed in the DATA).
NUMBER RULES (strict):
- Every number you write must appear in the DATA (you may round it). Derived figures such as
  spreads, changes, excesses, ratios and totals are already provided in the DATA - use those.
- Never add, subtract, multiply, divide, average or otherwise compute a number yourself. If a
  figure you want is not in the DATA, describe it in words without a number.
- Do not restate the date as a number range or count items with new numbers.
Do not invent prices or targets. At most 250 words; suggestions for human review, not investment advice
and not orders.

DATA:
{data}

RECOMMENDATIONS:""",
)

# Prompts used by each inventory component to produce its output.
COMPONENT_PROMPTS = {
    "MRM-001": ["rag_answer", "hyde_passage"],
    "MRM-002": ["market_analysis"],
    "MRM-003": ["risk_report"],
    "MRM-004": ["portfolio_recommendations"],
    "MRM-005": ["market_analysis", "risk_report", "portfolio_recommendations"],
}
