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

# -- Claude Haiku 4.5 (Bedrock) variants -------------------------------------------------
# Separate prompt versions so the Gemini-approved prompts (and their hashes) stay unchanged.
# They address the failure modes measured on Haiku (heldout-20260930T1*-bedrock): relabelled
# periods ("YTD"), derived/rounded numbers, unit conversions (bps), ratio-to-percent claims,
# inverted directions and incomplete lists.

HAIKU_RULES = """STRICT RULES:
- Every number you write must appear verbatim in the inputs. Do not round, convert units (for example
  percentage points to bps), add, subtract, multiply, divide or compute differences, sums or percentages.
- Quote metric names and their periods / bases exactly as labelled in the inputs. Never relabel a period:
  a return over the market-data window or since average cost is NOT year-to-date.
- Do not turn beta, tail ratio or any other ratio into a percentage or a "moves X% more" claim; state the
  ratio as written.
- Use the direction words given in the inputs (up/down, increased/decreased, above/below); never infer a
  direction by comparing two numbers yourself.
- When listing items that match a condition, include every matching item present in the inputs."""

RAG_ANSWER_HAIKU = register(
    "rag_answer_haiku",
    "1.0.0",
    """You are an analyst assistant for a hedge fund's internal research desk.
Answer the QUESTION using ONLY the SOURCES below (synthetic demo data).

""" + HAIKU_RULES + """
- Cite the sources you use inline with their ids in square brackets, e.g. [position:AAPL].
- If the sources do not contain the answer, say: "The indexed data does not contain this."
- At most 150 words. Analysis only, not investment advice.

CONVERSATION SO FAR (most recent last):
{history}

SOURCES:
{sources}

QUESTION: {question}

ANSWER:""",
)

MARKET_ANALYSIS_HAIKU = register(
    "market_analysis_haiku",
    "1.0.0",
    """You are the market analysis agent for a hedge fund (synthetic demo portfolio).
Using ONLY the DATA below, write a concise market analysis report in Markdown:
1. Sector exposure and performance (sector weights and average returns since cost, labelled as in the DATA).
2. Technical signals per position (RSI, SMA-50 vs SMA-200, MACD, volume vs 20-day average, and the price
   change over the market-data window with its stated period) - flag concerning ones.
3. Earnings pattern from the latest quarter using only the precomputed above/below-estimate statements
   and amounts in the DATA.
4. Macro backdrop (latest readings, the provided 10Y-2Y spread and the month-over-month changes with their
   stated direction and units, regime).
""" + HAIKU_RULES + """
Name tickers; at most 300 words; analysis only, not investment advice.

DATA:
{data}

MARKET ANALYSIS REPORT:""",
)

PORTFOLIO_RECOMMENDATIONS_HAIKU = register(
    "portfolio_recommendations_haiku",
    "1.0.0",
    """You are the portfolio management agent for a hedge fund (synthetic demo portfolio).
Using ONLY the DATA below, propose actions for the portfolio manager to review:
1. Immediate actions for CRITICAL/HIGH risk positions (reduce, hedge, review) - one line per ticker,
   formatted as "**TICKER**: <verb> ..." with exactly one of the verbs reduce, trim, hedge, hold or add.
2. Concentration adjustments for sectors or names above the policy limits, quoting the excess figures
   exactly as given in the DATA (do not recompute target values or remaining weights).
3. Tax-loss harvesting candidates: every position with an unrealized loss listed in the DATA.
""" + HAIKU_RULES + """
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

# Provider-specific prompt selection. Gemini uses COMPONENT_PROMPTS unchanged; Bedrock (Claude Haiku
# 4.5) swaps in the *_haiku variants. risk_report is shared (MRM-003 was approved on Haiku with it).
HAIKU_VARIANTS = {
    "rag_answer": "rag_answer_haiku",
    "market_analysis": "market_analysis_haiku",
    "portfolio_recommendations": "portfolio_recommendations_haiku",
}


def prompt_name_for(base: str, provider: str = "gemini") -> str:
    """Registered prompt name to use for ``base`` on ``provider``."""
    if provider == "bedrock":
        return HAIKU_VARIANTS.get(base, base)
    return base


def component_prompts(inventory_id: str, provider: str = "gemini") -> list:
    return [prompt_name_for(n, provider) for n in COMPONENT_PROMPTS.get(inventory_id, [])]
