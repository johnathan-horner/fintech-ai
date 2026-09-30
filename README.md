#  FinTech AI - Hedge Fund Intelligence Platform

![CI](https://github.com/johnathan-horner/fintech-ai/actions/workflows/ci.yml/badge.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fintech-ai-horner.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![SR 11-7 MRM](https://img.shields.io/badge/MRM-SR%2011--7%20framework-green.svg)](docs/MRM_FRAMEWORK.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AI-Powered Market Analysis, Portfolio Management & Risk Assessment**

FinTech AI analyzes a synthetic hedge-fund portfolio with RAG + multi-agent AI: market analysis, risk assessment (VaR, CVaR, stress tests, risk scores) and portfolio recommendations, with every AI output independently graded, version-traced and gated by SR 11-7-style model approvals.

> **What runs today:** all data is **synthetic** and bundled in `data/`. The free demo runs on the
> **Google Gemini API free tier** (answers/agents `gemini-3.5-flash-lite`, independent grader/HyDE `gemini-3.1-flash-lite`,
> embeddings `gemini-embedding-001`). The **production target architecture** is Amazon Bedrock (Claude) +
> Titan embeddings, defined as AWS CDK code in [`infra/`](infra/) (synthesized, never deployed by this repo).
> Items that are planned but not built are listed under [Roadmap (not yet implemented)](#roadmap-not-yet-implemented).

---

##  Live Demo

**[-> Try the live demo on Streamlit Cloud](https://fintech-ai-horner.streamlit.app)**

The quant math (VaR, Monte Carlo CVaR, stress tests, out-of-sample VaR backtest, PSI, MRM validation) is always computed live from the bundled data. AI features call Gemini (free tier) when a key is configured and the component is approved by the eval harness; each output is labelled **Live AI** (with grader score + trace) or **Demo mode** (with the reason: no key, session cap of 10 live requests, blocked by model governance, 429 quota, or error). Demo-mode text is a deterministic template filled with values computed from the data - never canned numbers.

---

## The Problem

Hedge fund analysts spend hours manually reviewing fragmented data across Bloomberg terminals, SEC filings, earnings transcripts, and news feeds. Insights are reactive. Risk is assessed after the fact. Alpha opportunities are missed.

## The Solution

An AI system that:
- Analyzes market data, earnings, and macroeconomic signals on demand (bundled synthetic data)
- Proactively flags portfolio risk with quantitative scoring (VaR, Sharpe, Beta)
- Recommends specific portfolio actions (reduce, trim, hedge, hold, add) for human review
- Lets analysts ask natural language questions about any stock, sector, or macro theme in the indexed data
- Protects sensitive data with guardrails (PII filtering, blocked topics) designed around SEC/FINRA concerns

---

## Tech Stack

| Layer | Free demo (what runs) | Production target (`infra/`, not deployed) |
|---|---|---|
| AI/ML | Gemini API free tier: `gemini-3.5-flash-lite` (answers/agents), `gemini-3.1-flash-lite` (independent grader + HyDE), `gemini-embedding-001` | Amazon Bedrock (Claude Sonnet 4 / Claude 3.5 Haiku, Titan Embeddings v1) |
| RAG Framework | LangChain Core (Documents, messages, chat/embedding models via `langchain-google-genai`); HyDE + local cross-encoder re-ranker (`fastembed`, ms-marco-MiniLM-L-6-v2 ONNX) | same code, `LLM_PROVIDER=bedrock` (`langchain-aws`, optional) |
| Agent Orchestration | LangGraph (StateGraph) | same |
| Vector Store | FAISS (`faiss-cpu`, cosine) | FAISS index in S3 |
| API Backend | FastAPI | API Gateway + Lambda |
| Dashboard | Streamlit | Streamlit |
| Infrastructure | Streamlit Community Cloud | AWS CDK (Python) |
| Security | Regex PII filter + blocked topics, approval gate, grader | Bedrock Guardrails, KMS, IAM, CloudTrail |
| Data Sources | Bundled synthetic data (`data/*.json`) | Bundled synthetic data (live sources are on the roadmap) |

---

## Architecture

**Data Flow (free demo, what runs):**
`data/*.json` + computed metrics -> 1000-token chunks -> Gemini embeddings -> FAISS -> HyDE retrieval (k=12) -> cross-encoder re-rank (top 4) -> Gemini answer -> independent Gemini grader + numeric check -> TraceRecord -> Streamlit / FastAPI

**Production target (`infra/`, synthesized only):**
EventBridge schedule -> Lambda (ingest: records runs to KMS-encrypted S3; no market-data client yet); API Gateway -> Lambda (analysis: Bedrock Claude via Converse with a Bedrock Guardrail); KMS, CloudTrail (7-year retention), least-privilege IAM.

**Agent Orchestration:**
START -> Market Analysis Agent -> Risk Assessment Agent -> Portfolio Management Agent -> END

---

## Features

### RAG Pipeline
- Earnings transcripts, market data (technicals per ticker), portfolio positions, sector summaries, macro indicators and computed risk metrics indexed as LangChain Documents (117 records)
- Titan Embeddings (1536-dimensional vectors) stored in FAISS in the production target; the free demo embeds with `gemini-embedding-001` (3072 dimensions)
- Query transformation for financial terminology (e.g., "P/E compression" -> targeted search) - implemented as real **HyDE**: the HyDE model writes a hypothetical answer passage, the passage is embedded as a document and FAISS returns the top 12; the raw question's top 12 are merged in
- Re-ranking for most relevant financial documents - local cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2`; top 4 go to the answer prompt and their scores are shown in the UI
- Conversational memory for multi-turn analysis sessions (last 5 turns)
- Chunking: 1000-token chunks with 200-token overlap (regex word/punctuation token approximation, recorded in `faiss_index/manifest.json`); every bundled record is shorter than one chunk
- Index manifest (`embedding_model, dim, n_docs, data_sha256, corpus_sha256, chunk_size, chunk_overlap, built_at`) is verified on load; a stale index is rebuilt

### Multi-Agent System
- **Market Analysis Agent (MRM-002):** Analyzes price trends, volume (vs 20-day average), technical indicators, sector momentum, earnings above/below estimates and the macro backdrop
- **Risk Assessment Agent (MRM-003):** Reports VaR, Monte Carlo CVaR, Sharpe/Sortino, beta, max drawdown and stress tests computed by the metrics engine (MRM-006); flags high-risk positions
- **Portfolio Management Agent (MRM-004):** Recommends reduce / trim / hedge / hold actions and concentration adjustments based on computed risk scores, limits and technicals
- **Orchestrator (MRM-005):** LangGraph StateGraph chains agents sequentially with shared portfolio state

Every number an agent may use is precomputed by code (`src/agents/context.py`): parametric VaR, seeded Monte Carlo CVaR, stress tests, risk scores, concentration excess over limits (points and dollars), Sharpe / Sortino / volatility / max drawdown, volume vs 20-day average, earnings above/below-estimate flags, the 10Y-2Y spread, month-over-month macro changes and the rule-based regime. Agent prompts (v1.2.0) forbid computing any number that is not in the inputs, and the deterministic numeric check enforces it.

### Model Governance (SR 11-7)
- **Prompt registry** - every prompt is `name@semver#sha8` (`src/prompts/`).
- **Independent grader** - a second call to a *different* model scores every output against the exact sources it was given (groundedness, numeric consistency, unsupported claims, PASS/FLAG), plus a deterministic check that every metric-like number in the output rounds from a source number.
- **Grader outages** - if the grader model fails (e.g. HTTP 503 "high demand" after retries), the output is kept and labelled `UNGRADED` with the error; the eval harness treats such items as errors (one retry; no approval is written for a component with errored items).
- **TraceRecord** on every output (trace id, inventory id MRM-001..005, provider, model id, prompt versions, embedding model, index manifest hash, source ids, grader result, approval status, timestamp), shown in an expander and kept in a session log.
- **Approval gate** - `require_approved()` runs before any model call; the running model id + prompt versions must match an APPROVED/CONDITIONAL record in `data/mrm/approvals.json`, which only the eval harness writes. Otherwise the UI shows "Blocked by model governance: <reason>" and falls back to demo output. The MRM page marks components whose running version differs from the approved one as "re-validation required".
- **VaR backtest** - out-of-sample, rolling 250-day window, 99% historical + parametric VaR (and the engine's beta-proxy VaR) on the current positions over `data/price_history.json` (seeded synthetic, Student-t df=4, factor structure consistent with each beta); Kupiec POF and Basel traffic light. MRM-006 approval comes from the Basel zone.
- **Monitoring** - PSI of portfolio returns (first 250 days vs last 60), PSI of grader scores (held-out eval vs this session, "insufficient data" below n=30), performance from the actual trace log.

### API Endpoints
- `POST /chat` - Natural language Q&A about markets, stocks, or macro themes
- `POST /insights/market` - Market analysis powered by Market Analysis Agent
- `POST /insights/risk` - Portfolio risk scoring powered by Risk Assessment Agent
- `POST /insights/portfolio` - Portfolio recommendations powered by Portfolio Agent
- `GET /health` - Health check

The app exposes **22 endpoints** in total (counted by introspecting `app.routes`, see `src/api/main.py:list_endpoints`): the 5 above, `GET /portfolio/summary`, `GET /traces/recent`, and 15 `/mrm/*` routes (inventory, validation incl. `POST /mrm/validation/run-all`, monitoring, governance, reporting). Every AI endpoint returns `mode`, `reason`, `grader` and `trace`.

### Compliance
- Bedrock Guardrails filter sensitive financial PII (production: `AWS::Bedrock::Guardrail` in `infra/`; demo: regex PII filter + blocked-topic check in `src/guardrails/`)
- KMS encryption at rest with automatic key rotation
- TLS 1.2+ for all data in transit
- IAM least-privilege
- CloudTrail audit logging for all Bedrock API calls
- Synthetic data for development - no real client data
- 7-year audit retention in `infra/` (CloudTrail log group `SEVEN_YEARS`, audit bucket expiry 2555 days; values in `infra/retention_config.json`)

---

## Measured Evaluation (held-out split)

Measured by `python scripts/run_eval.py --split heldout` on the held-out half of `data/eval/fintech_eval.jsonl`
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
| MRM-003 Risk agent (risk_report@1.2.0) | 10 | 10/10 (1.0000) | 1.0000 | n/a | **APPROVED** | `heldout-20260929T235842Z` |
| MRM-004 Portfolio agent (portfolio_recommendations@1.2.0) | 10 | 10/10 (1.0000) | 0.9960 | n/a | **APPROVED** | `heldout-20260930T000258Z` |
| MRM-005 Orchestrator (3 sections per item) | 10 | 10/10 (1.0000) | 0.9987 | n/a | **APPROVED** | `heldout-20260930T005058Z` |
| MRM-006 Metrics engine (VaR backtest) | 254 test days | - | - | - | **APPROVED** (Basel GREEN) | `heldout-20260929T210703Z` |

- MRM-006 out-of-sample 99% VaR backtest (2024-04-15 to 2026-03-19, 250-day rolling window, synthetic price history): historical_rolling: 3 exceptions / 254 days (last 250: 3, GREEN), Kupiec LR 0.0795, p 0.7779; parametric_rolling: 2 exceptions / 254 days (last 250: 2, GREEN), Kupiec LR 0.1251, p 0.7236; beta_proxy_engine: 2 exceptions / 254 days (last 250: 2, GREEN), Kupiec LR 0.1251, p 0.7236.
- History (all files kept in `data/eval/results/`): with v1.0.0 agent prompts the orchestrator FAILED 1/2 (`heldout-20260929T210703Z`) because the model computed a 10Y-2Y spread itself; the v1.1.0 market/risk/portfolio prompts were re-validated in `heldout-20260929T220706Z`, `...221455Z`, `...221956Z`. The fix (every derived number precomputed by code, prompts forbidding arithmetic, no non-authoritative upstream narratives in agent inputs, earnings above/below-estimate flags computed by code) is what the table above measures.
- `heldout-20260929T205406Z` (answers on `gemini-3.5-flash`) is incomplete: it stopped at item 21/34 on that model's free-tier limit of 20 requests/day and wrote no LLM approvals.
- MRM-001 was measured on 2026-09-29 and its prompts, model and index are unchanged since. MRM-003/004 were measured before a refactor that changed how the (identical) prompt messages are sent; prompts, inputs and model are the same.
- Small n for the agents (10 each): treat as a smoke-level validation, not a statistically strong one. Approvals: `data/mrm/approvals.json` (written only by the harness).

---

## Roadmap (not yet implemented)

None of the following exists in the code today; they are planned (and were previously described as features).

- **Live financial data sources** - connecting to Yahoo Finance (delayed quotes), FRED, SEC EDGAR filings (10-K / 10-Q / 8-K) and Federal Reserve / BLS feeds; indexing SEC filings in the RAG corpus; real-time (15-minute delayed) market data and real-time analysis / risk scoring. Today all analysis runs on the bundled synthetic data.
- **Market data ingestion pipeline** - "Market Data APIs -> EventBridge -> Lambda -> S3 -> Titan Embeddings" with a real vendor client (the CDK ingest Lambda only records runs).
- **Fundamental Analysis Agent** - financial metrics, valuation models and credit analysis over filings/transcripts.
- **Compliance Agent** - an LLM agent for policy checks of recommendations, MRM reporting, regulatory validation and audit-trail generation (today MRM reports and audit trails come from deterministic code in `src/mrm/`).
- **Sentiment analysis** of earnings calls and news.
- **Bias detection** - regular counterfactual / cross-sector evaluation across market conditions and asset classes.
- **Daily performance tracking** - a scheduled job publishing monitoring results (today monitoring runs on demand in the app/API).
- **LangChain LCEL** chains (`prompt | model | parser`) - the chains currently call the models directly.
- **MCP AI features** - AI-powered sector commentary in `generate_market_insights()`, AI-enhanced fundamental analysis in `analyze_stock_fundamentals()`, positioning recommendations in `get_market_regime_analysis()`, and performance reporting / attribution in `generate_compliance_report()`.
- **Performance attribution** (e.g. Brinson sector attribution vs a benchmark).
- **Full exit / new-position recommendations** - the portfolio agent currently recommends reduce, trim, hedge, hold, or add only.
- **Formal SR 11-7 compliance attestation** - independent validation by a separate team and a model committee; today the repo implements the MRM framework in code (inventory, held-out validation, approval gate, PSI monitoring, CRO reporting).

---

## Quick Start

### Prerequisites
- Python 3.11
- Optional: a Google AI Studio API key on a project **without billing** (free tier) for live AI

### Install
```bash
git clone https://github.com/johnathan-horner/fintech-ai.git
cd fintech-ai
python3 -m venv .venv && source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt
```

### Configure (optional, for live AI)
Put the key in `.streamlit/secrets.toml` (see `.streamlit/secrets.toml.template`) or export `GEMINI_API_KEY` for CLI scripts.

### Data, index, eval
```bash
python scripts/generate_price_history.py      # seeded synthetic price history (already committed)
python scripts/build_eval_set.py              # eval questions with computed reference answers (already committed)
python scripts/build_index.py                 # FAISS index with Gemini embeddings (already committed; needs key)
python scripts/run_eval.py --split heldout    # held-out eval -> data/eval/results/, data/mrm/approvals.json (needs key)
python scripts/run_eval.py --components MRM-006   # VaR backtest approval only (no key)
```

### Run the Dashboard
```bash
streamlit run streamlit_app/streamlit_app.py
```

### Run the API (new terminal)
```bash
pip install -r requirements-api.txt
uvicorn src.api.main:app --reload
```

### Tests
```bash
pytest -q        # no network; Gemini calls are faked
```

---

## Sample Queries

| Question | What It Does |
|---|---|
| "What are my highest risk positions?" | RAG over the indexed position risk scores (beta, drawdown and concentration flags) |
| "How is the tech sector trending?" | RAG over sector summaries and per-ticker technicals |
| "Should I hedge my NVDA position?" | RAG over NVDA's position, risk score and technicals (analysis, not advice) |
| "What does the Fed decision mean for my bonds?" | RAG over macro indicators (the synthetic book holds no bonds, so the answer says so) |
| "Summarize AAPL's last earnings call" | RAG over indexed earnings transcripts |

---

## Project Structure

```
fintech-ai/
 data/                          # Synthetic data (never real client data)
    portfolio.json, market_data.json, earnings_transcripts.json, macro_indicators.json
    price_history.json          # seeded synthetic daily prices (scripts/generate_price_history.py)
    eval/fintech_eval.jsonl     # eval set, reference answers computed from the data
    eval/results/               # eval run outputs
    mrm/approvals.json          # approvals written by the eval harness
 faiss_index/                   # FAISS index + docstore + manifest (committed)
 src/
    llm/                        # config.py (provider/models), provider.py, guard.py (fallback, 429/503 retries)
    prompts/                    # versioned prompt registry + templates
    rag/                        # document_loader, chunking, vector_store (faiss), query_transform (HyDE),
                                #   reranker (cross-encoder), rag_chain
    agents/                     # market / risk / portfolio agents, orchestrator (LangGraph), context builders
    demo/templates.py           # deterministic, data-derived demo outputs
    mrm/                        # inventory + approval gate, grader, trace, approvals, validation, monitoring,
                                #   governance, reporting
    eval/                       # eval dataset + harness
    api/                        # FastAPI app + MRM routes
    guardrails/                 # PII filter + blocked topics
    utils/                      # metrics (VaR, CVaR, Sharpe, Beta, stress, volume, backtest inputs), macro, data
    mcp_tools.py                # MCP tool implementations (deterministic)
 streamlit_app/                 # streamlit_app.py (entry), demo_pages.py, mrm_dashboard.py, live_ai.py
 scripts/                       # build_index, build_eval_set, run_eval, generate_price_history
 infra/                         # AWS CDK (Python) production target: app.py, fintech_stack.py, lambda/
 mcp_server.py                  # FastMCP server
 tests/                         # pytest (no network)
 notebooks/rag_evaluation.ipynb
 docs/                          # compliance, MRM framework, ADRs
```

## Architecture Decisions

See [docs/decisions/](docs/decisions/) for architecture decision records explaining why each service and pattern was chosen.

## AI System Documentation

See [AI_SYSTEM_CARD.md](AI_SYSTEM_CARD.md) for detailed AI system documentation including RAG pipeline, multi-agent workflows, LLM usage, and governance framework.

## API Documentation

Run the FastAPI server with `uvicorn src.api.main:app --reload` and visit `/docs` for interactive Swagger documentation.

## 🔌 MCP Server Integration

The FinTech AI Platform includes a Model Context Protocol (MCP) server that exposes financial analysis functionality through a standardized interface for AI assistant integration. Every tool returns values computed from the bundled data, `src/utils/metrics.py` and `src/mrm` - no LLM calls, no randomness, same input -> same output.

### Available Tools

- `analyze_portfolio_risk()` - Portfolio risk analysis with VaR, CVaR, stress testing and position risk scores
- `generate_market_insights()` - Sector analysis computed from the bundled market data (weight, P&L, price performance, technical status)
- `optimize_portfolio_allocation()` - Modern portfolio theory optimization: long-only minimum-variance allocation with per-name caps on the synthetic price-history covariance
- `analyze_stock_fundamentals()` - Fundamentals from the bundled data (P/E, 52-week range, beta, rating, earnings surprises, technicals)
- `get_market_regime_analysis()` - Rule-based market regime classification from the latest macro indicators
- `generate_compliance_report()` - MRM risk and model-governance report (inventory, VaR backtest, PSI, concentration, governance)

### Available Resources

- `system://rag_pipeline` - RAG architecture and data source information
- `data://market_data` - Market data sources and quality controls

### Available Prompts

- `portfolio_analysis` - Portfolio analysis workflow prompt for an AI assistant that calls these tools
- `market_research` - Market research workflow prompt for an AI assistant that calls these tools

Run with: `pip install -r requirements-mcp.txt && python mcp_server.py` (separate environment: fastmcp 4 needs `websockets>=17`, which conflicts with Streamlit's pin).

## CI/CD Pipeline

GitHub Actions automatically runs on push/PR to main:
- Python 3.11 setup and dependency installation
- Code linting with flake8 (max line length 120)
- Import testing for core modules
- `pytest` (no network), a separate job that installs `requirements-mcp.txt` and imports `mcp_server`, and `cdk synth` of `infra/` (no AWS credentials, never deploys)

See [.github/workflows/ci.yml](.github/workflows/ci.yml) for full pipeline configuration.

---

*Built by Johnathan Horner*
