# AI System Card: FinTech AI - Hedge Fund Intelligence Platform

## System Overview
- System Type: RAG-powered multi-agent financial analysis platform
- Primary LLM: Amazon Bedrock Claude Haiku 4.5 (`us.anthropic.claude-haiku-4-5-20251001-v1:0`, us-east-1) for answers, agents and HyDE; fallback Google Gemini free tier (`gemini-3.5-flash-lite`; HyDE `gemini-3.1-flash-lite`); demo mode last. The grader is always a different model from the one that answered. Embeddings: `gemini-embedding-001`. Model ids live in `src/llm/config.py`.
- Framework: LangChain + LangGraph for agent orchestration
- Version: 2.0.0
- Owner: Johnathan Horner, Shoot It Analytics LLC

## Intended Use
- Primary: AI-powered market analysis, portfolio management, and risk assessment for hedge funds (demonstrated on a synthetic portfolio)
- Users: Portfolio managers, risk analysts, investment researchers, compliance officers
- Out of scope: Not intended as automated trading system or standalone investment advice. Human oversight required for all investment decisions

## System Architecture

### RAG Pipeline
- **Embedding Model**: `gemini-embedding-001` (3072-dim; documents embedded as `RETRIEVAL_DOCUMENT`, questions as `RETRIEVAL_QUERY`, HyDE passages as `RETRIEVAL_DOCUMENT`).
- **Vector Store**: FAISS for semantic similarity search (`faiss-cpu` IndexFlatIP on L2-normalised vectors = cosine), manifest-verified (`faiss_index/manifest.json`).
- **Data Sources (indexed today)**: 117 records from the bundled synthetic data - fund and sector summaries, 15 positions, technicals for 20 tickers from `market_data.json`, 40 earnings-call summaries, 12 months of macro indicators, and risk metrics computed by `src/utils/metrics.py` (VaR, CVaR, stress scenarios, concentration, 15 risk scores).
- **Chunk Strategy**: 1000-token chunks with 200-token overlap (token count = regex `\w+|[^\w\s]` approximation; chunk size/overlap recorded in the index manifest; every current record fits in one chunk).
- **Retrieval**: Semantic similarity with relevance scoring - HyDE (hypothetical answer passage embedded, FAISS top 12, merged with the raw question's top 12) followed by a local cross-encoder re-ranker (`Xenova/ms-marco-MiniLM-L-6-v2` via fastembed ONNX); the top 4 re-ranked sources and their scores are passed to the answer prompt and shown in the UI.

### Multi-Agent Workflows
LangGraph StateGraph `market_analysis -> risk_assessment -> portfolio_management -> END` (inventory MRM-002, MRM-003, MRM-004; orchestrated by MRM-005).
- **Portfolio Risk Agent (MRM-003)**: VaR, Monte Carlo CVaR, stress testing and scenario analysis, Sharpe/Sortino/drawdown and concentration - all computed by the metrics engine (MRM-006); the agent narrates them.
- **Market Analysis Agent (MRM-002)**: sector insights, technicals incl. volume vs 20-day average, earnings above/below estimates and the macro backdrop with a rule-based regime (regime detection is the deterministic rule in `src/utils/macro.py`; the LLM narrates it).
- **Portfolio Management Agent (MRM-004)**: reduce / trim / hedge / hold suggestions and concentration adjustments for human review.
- Every derived figure (spreads, changes, excess over limits, Sharpe/Sortino/drawdown, volume ratios, above/below-estimate flags) is precomputed by code; agent prompts (v1.2.0) forbid arithmetic and the deterministic numeric check enforces it.

### LLM Integration
- **Primary Model**: Claude Haiku 4.5 on Amazon Bedrock (boto3 `converse`); on any Bedrock error, throttle, missing credentials or missing approval the call falls back to Gemini, then to demo mode. Every call has a max-output-token cap. Outputs are labelled with the provider + model that actually answered and the fallback reason.
- **Reasoning Tasks**: Market commentary, risk assessment, investment thesis generation
- **Guardrails**: Financial advice limitations, regulatory compliance, data privacy (regex PII filter + blocked topics in `src/guardrails/`; `AWS::Bedrock::Guardrail` in `infra/`)
- **Independent grader**: every output is scored by a different model against the exact inputs it was given (Haiku answer -> `gemini-3.1-flash-lite`; Gemini answer -> Haiku if available, else `gemini-3.1-flash-lite`; no different model -> UNGRADED, never self-graded), plus a deterministic numeric check. If the grader fails (e.g. 503 "high demand" after retries) the output is kept, labelled UNGRADED with the error, and never counts as a pass in the eval.
- **Live-call guard**: at most 10 live AI requests per Streamlit session (one user action = one request, even if it makes several model calls); fallback to labelled, data-derived demo output when the key is missing, the cap is reached, governance blocks the call, the free tier returns 429 (after one retry), or anything fails.

## Data Sources and Processing
- **Data**: all portfolio, market, earnings and macro data are synthetic files in `data/`, plus `data/price_history.json` (seeded synthetic daily prices used for the VaR backtest and PSI). No external data client exists (see Roadmap).
- **Data Quality**: Statistical validation, outlier detection, missing data handling (the data-quality validation test checks completeness, value ranges, weight totals and beta outliers)

## Performance Metrics
- **RAG Accuracy**: Retrieval relevance scoring and answer attribution (cross-encoder scores; answers cite source ids)
- **Model Validation**: Backtest accuracy, stability scores, data quality metrics

**Measured evaluation (held-out split)**

Gemini fallback path. Measured by `python scripts/run_eval.py --split heldout` on the held-out half of `data/eval/fintech_eval.jsonl`
(eval set v2: 127 items - 47 RAG questions with reference answers computed from the data + 20 portfolio variants for each of
MRM-002..005; seeded 50/50 split per component, documented in `src/eval/dataset.py`). Models: answers `gemini-3.5-flash-lite`,
grader/HyDE `gemini-3.1-flash-lite`, embeddings `gemini-embedding-001` (free tier), runs on 2026-09-29/30 UTC.
An item passes when the independent grader returns PASS with groundedness >= 0.8, the deterministic numeric check finds no unmatched
number, and (for exact-fact items) every computed reference value/term appears in the answer. APPROVED needs pass rate >= 0.85 and
mean groundedness >= 0.80; CONDITIONAL >= 0.70; otherwise FAILED. Orchestrator items pass only if all three sections pass.

| Component | n (held-out) | Passed | Mean groundedness | Exact-fact accuracy | Status | Eval run |
|---|---|---|---|---|---|---|
| MRM-001 RAG Q&A chain | 23 | 22/23 (0.9565) | 1.0000 | 0.9524 | **APPROVED** | `heldout-20260929T210703Z` |
| MRM-002 Market agent (market_analysis@1.2.0) | 10 | 10/10 (1.0000) | 1.0000 | n/a | **APPROVED** | `heldout-20260930T004656Z` |
| MRM-003 Risk agent (risk_report@1.2.0) | 10 | 10/10 (1.0000) | 1.0000 | n/a | **APPROVED** | `heldout-20260930T010751Z` |
| MRM-004 Portfolio agent (portfolio_recommendations@1.2.0) | 10 | 10/10 (1.0000) | 1.0000 | n/a | **APPROVED** | `heldout-20260930T010751Z` |
| MRM-005 Orchestrator (3 sections per item) | 10 | 10/10 (1.0000) | 0.9987 | n/a | **APPROVED** | `heldout-20260930T005058Z` |
| MRM-006 Metrics engine (VaR backtest) | 254 test days | - | - | - | **APPROVED** (Basel GREEN) | `heldout-20260929T210703Z` |

- MRM-006 out-of-sample 99% VaR backtest (2024-04-15 to 2026-03-19, 250-day rolling window, synthetic price history): historical_rolling: 3 exceptions / 254 days (last 250: 3, GREEN), Kupiec LR 0.0795, p 0.7779; parametric_rolling: 2 exceptions / 254 days (last 250: 2, GREEN), Kupiec LR 0.1251, p 0.7236; beta_proxy_engine: 2 exceptions / 254 days (last 250: 2, GREEN), Kupiec LR 0.1251, p 0.7236.
- History (all files kept in `data/eval/results/`): with v1.0.0 agent prompts the orchestrator FAILED 1/2 (`heldout-20260929T210703Z`) because the model computed a 10Y-2Y spread itself; the v1.1.0 market/risk/portfolio prompts were re-validated in `heldout-20260929T220706Z`, `...221455Z`, `...221956Z`. The fix (every derived number precomputed by code, prompts forbidding arithmetic, no non-authoritative upstream narratives in agent inputs, earnings above/below-estimate flags computed by code) is what the table above measures.
- `heldout-20260929T205406Z` (answers on `gemini-3.5-flash`) is incomplete: it stopped at item 21/34 on that model's free-tier limit of 20 requests/day and wrote no LLM approvals.
- MRM-001 was measured on 2026-09-29 and its prompts, model and index are unchanged since.
- Small n for the agents (10 each): treat as a smoke-level validation, not a statistically strong one. Approvals: `data/mrm/approvals.json` (written only by the harness).

Bedrock Claude Haiku 4.5 path: not yet evaluated (no Haiku approvals; the gate therefore routes live calls to the Gemini fallback). Dry-run cost estimate for the full held-out Haiku eval: $0.751 (per component $0.08-$0.28).

## Risk Management and Monitoring
- **Model Risk**: SR 11-7 compliance framework with independent validation (a separate grader model scores every output; approvals are earned only by the held-out eval harness and enforced before every model call)
- **Data Quality**: Automated validation rules and quality scoring
- **Performance Monitoring**: on-demand monitoring in the app/API - performance from the live trace log, PSI of portfolio returns, PSI of grader scores vs the held-out eval, validation expiry

## Ethical Considerations
- **Investment Advice**: System provides analysis only, not personalized investment advice
- **Market Manipulation**: Safeguards against generating misleading market information (grader + numeric consistency check flag unsupported numbers)
- **Transparency**: Clear attribution of AI-generated insights vs. quantitative calculations ("Live AI" / "Demo mode" badges, TraceRecord on every output)

## Regulatory Compliance
- **Model Risk Management**: SR 11-7-style framework implemented in code (inventory, held-out validation, approval gate, PSI monitoring, CRO reporting)
- **Data Governance**: 7-year retention, complete audit trails, change management (retention set in `infra/`: CloudTrail log group `RetentionDays.SEVEN_YEARS`, audit S3 bucket expiry 2555 days)

## Governance Framework
- **Change Management**: Controlled deployment with testing and validation (prompt registry `name@semver#sha8`; any model or prompt change blocks live use until re-approved)
- **Documentation**: Complete model inventory and methodology documentation
- **Incident Response**: Procedures for model failures and performance degradation

## Limitations and Known Issues
- **Synthetic data**: all portfolio, market, earnings, macro and price-history data are synthetic; backtests validate mechanics, not real-market calibration. The synthetic earnings "beat/miss" labels do not always agree with the revenue/EPS figures.
- **No live market data**: analysis is not real-time (see Roadmap)
- **Historical Bias**: Training on historical data may not reflect future market conditions
- **Complexity Risk**: Multi-agent workflows require careful orchestration and monitoring
- **Regulatory Evolution**: Ongoing compliance with evolving AI regulation in finance
- **Providers**: Bedrock throttling and Gemini free-tier rate limits / "high demand" 503s apply; calls fall back Bedrock -> Gemini -> data-derived templates
- **Bedrock Guardrails, KMS, CloudTrail**: defined in `infra/` (CDK) only; not applied to the live demo's Bedrock calls

## Usage Guidelines
- **Human Oversight**: All investment decisions require human review and approval
- **Risk Limits**: System outputs must be validated against portfolio risk limits
- **Documentation**: Investment rationale and AI contribution must be documented
- **Regular Validation**: Ongoing model performance monitoring and recalibration

## Roadmap (not yet implemented)
None of the following exists in the code today:
- **Live data sources**: Yahoo Finance / FRED market data with 15-minute delay, SEC EDGAR filings (10-K, 10-Q, 8-K) with daily updates, Federal Reserve / Bureau of Labor Statistics monthly indicators, real-time analysis.
- **Fundamental Analysis Agent**: financial metrics, valuation models, credit analysis.
- **Compliance Agent**: MRM reporting, regulatory validation, audit trail generation as an LLM agent (today these come from deterministic code in `src/mrm/`).
- **Sentiment analysis** as part of market analysis.
- **Bias detection**: regular evaluation across market conditions and asset classes.
- **Daily** model performance tracking (scheduled).
- **LangChain LCEL** chains.
- **Amazon Titan Text Embeddings** (the index is built on `gemini-embedding-001`).
- **MCP**: AI-powered / AI-enhanced tool output, positioning recommendations, performance attribution.
- **Client data protection**: encryption and access control for client portfolio data, and anonymization (today all data is synthetic JSON in the repo; KMS encryption is defined only in `infra/`).
- **SEC/FINRA** investment-adviser and fiduciary compliance review.
- **Model committee**: quarterly model performance and risk reviews (a process, not code).
- **Formal SR 11-7 compliance attestation** by an independent validation function.
