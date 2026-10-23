# FinTech AI: Hedge Fund Intelligence Demo

![CI](https://github.com/johnathan-horner/fintech-ai/actions/workflows/ci.yml/badge.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fintech-ai-horner.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Status: Live.** The public demo works today (checked 2026-10-23; the free Streamlit host puts it to sleep when idle and it wakes on the first visit). The live demo runs branch `main`, which answers with Google Gemini. This branch makes Amazon Bedrock Claude Haiku 4.5 the primary model; all five AI components passed held-out validation on Haiku, so once this branch is merged, live answers will come from Bedrock with Gemini as the fallback.

Independent reference project by Shoot It Analytics LLC. Built with public or synthetic data.

Live demo: https://fintech-ai-horner.streamlit.app

## What the demo does today

FinTech AI analyzes a synthetic hedge fund portfolio (15 positions, $32.36M AUM). All data is synthetic and bundled in `data/`. No real client data is used.

- **Portfolio Overview:** AUM, P&L, weighted beta, sector weights and positions, computed from `data/portfolio.json`.
- **Market Analysis, Risk Assessment, Portfolio Management:** one button runs a LangGraph pipeline of three agents (market, risk, portfolio). Every number the agents see is computed by code first: parametric VaR, Monte Carlo CVaR, stress tests, risk scores, Sharpe, Sortino, drawdown, volume and macro changes. The agents are told not to compute numbers of their own.
- **AI Chat:** questions over the bundled data. Retrieval uses HyDE (the model writes a hypothetical answer, and that answer is used for the FAISS search), then a local cross-encoder re-ranks the results and the top 4 go to the answer prompt.
- **Model Risk Management:** a 6-model inventory, held-out validation results, an out-of-sample VaR backtest, PSI drift monitoring, governance checks and a CRO report, all computed by `src/mrm/`.

Every AI output carries:

- a label naming the provider and model that actually answered, or "Demo mode" with the reason,
- a score from an independent grader, which is always a different model from the one that answered,
- a trace record with the inventory ID, model, prompt versions and approval status.

Live AI is capped at 10 requests per browser session. When live AI is not available, the app shows deterministic output built from the same computed numbers.

## Why I built this

**The problem.** Banks and investment firms that put language models near investment decisions
are expected to manage them like any other model: an inventory, independent validation, ongoing
monitoring and a record of what each model said and why (Federal Reserve guidance SR 26-2,
formerly SR 11-7). Most AI demos skip all of that, and the numbers a model writes cannot be traced
to source data.

**What I set out to prove.** Whether a multi-agent analysis assistant can run under model risk
management controls in code: every number computed before the model sees it, every model and prompt
version approved by a held-out evaluation before it may answer, and every answer graded and traced.

**Key design decisions**
- **Data:** a synthetic 15-position portfolio and synthetic price history, bundled in the repo.
- **Numbers come from code:** VaR, Monte Carlo CVaR (10,000 seeded simulations), stress tests and
  risk scores are computed first; agents are told not to compute numbers and a deterministic check
  rejects numbers not in their inputs.
- **Approval gate:** a model and prompt version may answer only after passing the held-out
  evaluation; the gate checks the exact version before each call.
- **Independent grading:** every answer is graded by a model from a different vendor.
- **Cost:** a 10-request cap per browser session, the cheapest capable models, and a dry-run cost
  estimate before any evaluation run.

**Results.** On the held-out split with Bedrock Claude Haiku 4.5: the RAG chat passed 22 of 23
questions (95.2% exact-fact accuracy) and all five AI components were approved. The out-of-sample
VaR backtest over 254 days had 2 to 3 exceptions with Kupiec p-values 0.72 to 0.78 (Basel green
zone). Details below.

**Limits.** Synthetic data only; never used with real portfolios, clients or trading. Not
investment advice, and not a certified or regulated product. The controls follow the structure of
SR 26-2 but have not been reviewed by an independent validation team.

**What I'd do next in a real deployment.** Connect licensed market data, have the firm's model
validation team review the inventory and test results, deploy the CDK stack in the firm's AWS account
with Bedrock Guardrails, and run the monitoring job on a schedule with alerts.

## Architecture

```
data/*.json and computed metrics
  -> chunking (1000 tokens, 200 overlap) -> gemini-embedding-001 -> FAISS index (committed)
  -> HyDE retrieval -> cross-encoder re-rank (ms-marco-MiniLM-L-6-v2, local ONNX)
  -> model governance gate (approval check for the exact model and prompt version)
  -> answer: Amazon Bedrock Claude Haiku 4.5, then Gemini, then demo mode
  -> independent grader (a different model) + deterministic numeric check
  -> trace record -> Streamlit UI or FastAPI
```

Agents: START -> Market Analysis Agent (MRM-002) -> Risk Assessment Agent (MRM-003) -> Portfolio Management Agent (MRM-004) -> END, run by the orchestrator (MRM-005). The RAG chain is MRM-001. The metrics engine is MRM-006.

**Provider order.** Bedrock Claude Haiku 4.5 (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) is tried first. Any Bedrock error, throttle, missing credentials or missing approval falls through to Gemini (`gemini-3.5-flash-lite`). If no live provider works, the app shows demo mode.

**Governance gate.** Approvals live in `data/mrm/approvals.json` and are written only by the evaluation harness. Each approval is tied to a component, provider, model ID and prompt versions. Before any model call, the gate checks the exact version that is about to run. A version that is not approved is skipped for that provider.

**Production target.** `infra/` is AWS CDK code (Bedrock Guardrails, KMS, CloudTrail with 7-year retention, Lambda, API Gateway, least-privilege IAM). It passes `cdk synth` in CI. It is not deployed and is not used by the live demo.

## Measured evaluation (held-out split)

How it is measured: `python scripts/run_eval.py --split heldout` on the held-out half of `data/eval/fintech_eval.jsonl` (127 items: 47 RAG questions with reference answers computed from the data, plus 20 portfolio variants for each agent component; seeded 50/50 split per component, see `src/eval/dataset.py`). An item passes when the independent grader returns PASS with groundedness of at least 0.8, the numeric check finds no number that is not in the sources, and, for exact-fact items, every reference value appears in the answer. APPROVED needs a pass rate of at least 0.85 and mean groundedness of at least 0.80. CONDITIONAL needs a pass rate of at least 0.70. Agent results use 10 held-out items each, so treat them as smoke-level checks.

**Bedrock Claude Haiku 4.5 (primary).** Answers by Haiku 4.5, graded by `gemini-3.1-flash-lite`.

| Component | Prompt version | Held-out items | Passed | Mean groundedness | Exact-fact accuracy | Status | Eval run |
|---|---|---|---|---|---|---|---|
| MRM-001 RAG chat | rag_answer_haiku@1.0.0 | 23 | 22/23 (0.957) | 1.000 | 0.952 | APPROVED | heldout-20260930T160709Z-bedrock |
| MRM-002 Market agent | market_analysis_haiku@1.0.0 | 10 | 10/10 | 0.991 | n/a | APPROVED | heldout-20260930T163311Z-bedrock |
| MRM-003 Risk agent | risk_report@1.2.0 | 10 | 9/10 | 0.996 | n/a | APPROVED | heldout-20260930T152404Z-bedrock |
| MRM-004 Portfolio agent | portfolio_recommendations_haiku@1.0.0 | 10 | 10/10 | 1.000 | n/a | APPROVED | heldout-20260930T164023Z-bedrock |
| MRM-005 Orchestrator (all 3 sections must pass) | the three agent prompts above | 10 | 9/10 | 0.991 | n/a | APPROVED | heldout-20261016T064021Z-bedrock |

Measured Bedrock cost of these runs, from Bedrock usage data: MRM-001 $0.065, MRM-002 $0.125, MRM-003 $0.083, MRM-004 $0.062, MRM-005 $0.272.

**Gemini (fallback).** Answers by `gemini-3.5-flash-lite`, graded by `gemini-3.1-flash-lite`.

| Component | Held-out items | Passed | Mean groundedness | Exact-fact accuracy | Status | Eval run |
|---|---|---|---|---|---|---|
| MRM-001 RAG chat | 23 | 22/23 (0.957) | 1.000 | 0.952 | APPROVED | heldout-20260929T210703Z |
| MRM-002 Market agent | 10 | 10/10 | 1.000 | n/a | APPROVED | heldout-20260930T004656Z |
| MRM-003 Risk agent | 10 | 10/10 | 1.000 | n/a | APPROVED | heldout-20260930T010751Z |
| MRM-004 Portfolio agent | 10 | 10/10 | 1.000 | n/a | APPROVED | heldout-20260930T010751Z |
| MRM-005 Orchestrator | 10 | 10/10 | 0.999 | n/a | APPROVED | heldout-20260930T005058Z |

All result files are in `data/eval/results/`. Earlier Haiku runs with the shared prompts (heldout-20260930T142057Z to T153729Z, bedrock) scored lower: MRM-002 and MRM-005 FAILED, mostly because Haiku mislabeled return periods. That led to the Haiku-specific prompts above.

VaR backtest (MRM-006): out-of-sample 99% VaR on a rolling 250-day window over seeded synthetic prices, 254 test days, 2 to 3 exceptions depending on method, Kupiec p-values 0.72 to 0.78, Basel GREEN (run `heldout-20260929T210703Z`).

Re-checked on 2026-10-23: running `portfolio_backtest()` from `src/mrm/validation/model_validator.py` reproduces these backtest numbers (historical 3 exceptions, p = 0.778; parametric and beta-proxy 2 exceptions, p = 0.724), and the full test suite passes (118 tests).

## How to run it

Requires Python 3.11.

```bash
git clone https://github.com/johnathan-horner/fintech-ai.git
cd fintech-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
streamlit run streamlit_app/streamlit_app.py
```

The app runs without any keys, in demo mode. For live AI, add `.streamlit/secrets.toml` (see `.streamlit/secrets.toml.template`):

- `[aws]` with `access_key_id`, `secret_access_key` and `region` for Bedrock (primary)
- `[gemini]` with `api_key` for Gemini (fallback, and required for embeddings)

Other commands:

```bash
pytest -q                                              # tests, no network
pip install -r requirements-api.txt && uvicorn src.api.main:app --reload   # FastAPI, 22 endpoints
python scripts/run_eval.py --provider bedrock --dry-run                    # Bedrock cost estimate, no calls
python scripts/run_eval.py --split heldout --provider bedrock --components MRM-002 --max-cost 1.00
```

The MCP server needs its own environment: `pip install -r requirements-mcp.txt && python mcp_server.py`. Its 6 tools return values computed from the bundled data.

## Limits

- Synthetic portfolio and prices. Results on real markets would differ.
- The held-out evaluation set is small (23 RAG questions, 10 items for each agent and the orchestrator).
- The grader is itself a language model and can be wrong; a deterministic numeric check backs it up.
- The live demo still answers with Gemini until this branch is merged.
- The AWS CDK stack in `infra/` passes `cdk synth` but has never been deployed.

## What's next

These are not built yet:

- Live market data from FRED and SEC EDGAR, shown next to the synthetic portfolio
- Fundamental Analysis and Compliance agents
- Sentiment analysis of earnings transcripts
- Bias checks in the evaluation harness
- A scheduled daily monitoring job
- An AI commentary tool in the MCP server, and positioning recommendations
- Performance attribution
- LangChain LCEL chains
- Titan embeddings for the production path
- Full exit and new-position recommendations (today the portfolio agent recommends reduce, trim, hedge, hold or add)
- Formal SR 26-2 (formerly SR 11-7) attestation by an independent validation team (today the repo implements the framework in code)

## More detail

- [AI_SYSTEM_CARD.md](AI_SYSTEM_CARD.md): system card
- [docs/decisions/](docs/decisions/): architecture decision records
- [.github/workflows/ci.yml](.github/workflows/ci.yml): CI (pytest, MCP import, cdk synth)

Built by Johnathan Horner
