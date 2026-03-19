#  FinTech AI - Hedge Fund Intelligence Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![SR 11-7 MRM](https://img.shields.io/badge/MRM-SR%2011--7%20Compliant-green.svg)](docs/MRM_FRAMEWORK.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AI-Powered Market Analysis, Portfolio Management & Risk Assessment**

FinTech AI connects to financial data sources, analyzes market conditions using RAG + multi-agent AI, and delivers institutional-grade insights - covering market analysis, portfolio optimization, and real-time risk scoring - all while maintaining SEC/FINRA compliance.

---

##  Live Demo

**[-> Try the live demo on Streamlit Cloud](https://your-app-name.streamlit.app)**

Demo mode runs on synthetic data with no AWS credentials required. The actual Python quant math (VaR, CVaR, stress tests, MRM validation) runs live - only the LLM narrative calls are mocked.

---

## The Problem

Hedge fund analysts spend hours manually reviewing fragmented data across Bloomberg terminals, SEC filings, earnings transcripts, and news feeds. Insights are reactive. Risk is assessed after the fact. Alpha opportunities are missed.

## The Solution

An AI system that:
- Analyzes market data, earnings, and macroeconomic signals in real-time
- Proactively flags portfolio risk with quantitative scoring (VaR, Sharpe, Beta)
- Recommends specific portfolio actions (rebalance, hedge, exit, enter)
- Lets analysts ask natural language questions about any stock, sector, or macro theme
- Protects sensitive data with SEC/FINRA-compliant guardrails

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI/ML | Amazon Bedrock (Claude 3 Sonnet, Titan Embeddings) |
| RAG Framework | LangChain 0.3, LCEL |
| Agent Orchestration | LangGraph (StateGraph) |
| Vector Store | FAISS |
| API Backend | FastAPI |
| Dashboard | Streamlit |
| Infrastructure | AWS CDK (Python) |
| Security | Bedrock Guardrails, KMS, IAM, CloudTrail |
| Data Sources | Yahoo Finance, SEC EDGAR, synthetic market data |

---

## Architecture

**Data Flow:**
Market Data APIs -> EventBridge -> Lambda -> S3 (KMS Encrypted) -> Titan Embeddings -> FAISS -> LangChain LCEL -> Bedrock Claude -> Guardrails -> FastAPI -> Streamlit

**Agent Orchestration:**
START -> Market Analysis Agent -> Risk Assessment Agent -> Portfolio Management Agent -> END

---

## Features

### RAG Pipeline
- SEC filings, earnings transcripts, and market data indexed as LangChain Documents
- Titan Embeddings (1536-dimensional vectors) stored in FAISS
- Query transformation for financial terminology (e.g., "P/E compression" -> targeted search)
- Re-ranking for most relevant financial documents
- Conversational memory for multi-turn analysis sessions

### Multi-Agent System
- **Market Analysis Agent:** Analyzes price trends, volume, technical indicators, sector momentum
- **Risk Assessment Agent:** Calculates VaR, Sharpe Ratio, Beta, drawdown risk; flags high-risk positions
- **Portfolio Management Agent:** Recommends rebalancing, hedges, entries/exits based on risk + market data
- **Orchestrator:** LangGraph StateGraph chains agents sequentially with shared portfolio state

### API Endpoints
- `POST /chat` - Natural language Q&A about markets, stocks, or macro themes
- `POST /insights/market` - Market analysis powered by Market Analysis Agent
- `POST /insights/risk` - Portfolio risk scoring powered by Risk Assessment Agent
- `POST /insights/portfolio` - Portfolio recommendations powered by Portfolio Agent
- `GET /health` - Health check

### Compliance
- Bedrock Guardrails filter sensitive financial PII
- KMS encryption at rest with automatic key rotation
- TLS 1.2+ for all data in transit
- IAM least-privilege
- CloudTrail audit logging for all Bedrock API calls
- Synthetic data for development - no real client data

---

## Quick Start

### Prerequisites
- Python 3.11+
- AWS CLI configured with Bedrock access
- Bedrock model access enabled (Claude 3 Sonnet, Titan Embeddings v1)

### Install
```bash
git clone https://github.com/yourusername/fintech-ai.git
cd fintech-ai
pip install boto3 langchain langchain-aws langchain-community langgraph faiss-cpu streamlit fastapi uvicorn faker yfinance pandas numpy "numpy<2"
```

### Generate Synthetic Data
```bash
python3 src/etl/generate_data.py
```

### Build Vector Store
```bash
python3 src/rag/vector_store.py
```

### Run the API
```bash
uvicorn src.api.main:app --reload
```

### Run the Dashboard (new terminal)
```bash
streamlit run streamlit_app/app.py
```

Open http://localhost:8501 and start analyzing.

---

## Sample Queries

| Question | What It Does |
|---|---|
| "What are my highest risk positions?" | Runs VaR + Beta analysis across portfolio |
| "How is the tech sector trending?" | Market analysis with momentum scoring |
| "Should I hedge my NVDA position?" | Portfolio agent recommends hedge strategy |
| "What does the Fed decision mean for my bonds?" | Macro impact analysis |
| "Summarize AAPL's last earnings call" | RAG over indexed earnings transcripts |

---

## Project Structure

```
fintech-ai/
 data/                         # Synthetic market + portfolio data
?    portfolio.json
?    market_data.json
?    earnings_transcripts.json
?    macro_indicators.json
 faiss_index/                  # FAISS vector embeddings
 src/
?    etl/
?   ?    generate_data.py      # Synthetic financial data generator
?    rag/
?   ?    document_loader.py    # JSON -> LangChain Documents
?   ?    vector_store.py       # FAISS + Titan Embeddings
?   ?    rag_chain.py          # LCEL chain with memory
?   ?    query_transform.py    # Financial query rewriting
?   ?    reranker.py           # LLM-based re-ranking
?    agents/
?   ?    market_agent.py       # Market analysis agent
?   ?    risk_agent.py         # Risk assessment agent
?   ?    portfolio_agent.py    # Portfolio management agent
?   ?    orchestrator.py       # Multi-agent orchestrator
?    api/
?   ?    main.py               # FastAPI endpoints
?    guardrails/
?   ?    bedrock_guardrails.py # SEC/FINRA compliance filtering
?    utils/
?        metrics.py            # Financial metrics (VaR, Sharpe, Beta)
 streamlit_app/
?    app.py                    # Analyst dashboard
 terraform/
?    main.py                   # AWS CDK stack
 notebooks/
?    rag_evaluation.ipynb      # RAG pipeline evaluation
 docs/
     COMPLIANCE.md             # SEC/FINRA compliance documentation
```

---

*Built by Johnathan Horner*
