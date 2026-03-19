# Git Commit Guide

This project uses Conventional Commits (https://www.conventionalcommits.org).
No emoji. No decorative characters. Plain English only.

## Format

    <type>(<scope>): <short summary>

    [optional body]

    [optional footer]

## Types

    feat      - A new feature
    fix       - A bug fix
    refactor  - Code change that is not a fix or feature
    docs      - Documentation only
    test      - Adding or updating tests
    chore     - Build process, dependency updates, config
    perf      - Performance improvement
    ci        - CI/CD pipeline changes

## Scope (optional, use the module or layer)

    mrm, agents, rag, api, etl, metrics, streamlit, infra, docs

## Examples

Good:
    feat(mrm): add CVaR/Expected Shortfall to model validator
    feat(metrics): implement Monte Carlo portfolio CVaR with fat-tail mixture
    feat(agents): wire stress test results into risk agent LLM prompt
    fix(mrm): correct Kupiec POF test chi-squared degrees of freedom
    refactor(metrics): extract full_risk_report as unified entry point
    docs(mrm): update MRM_FRAMEWORK with CVaR outstanding item closure
    chore: add .gitignore and requirements-streamlit.txt
    feat(streamlit): add demo mode with no-AWS fallback for Streamlit Cloud

Bad (do not use):
    added stuff
    fix bug
    WIP
    update
    feat: add CVaR to model validator (closes MRM item #1)  <- no parens needed for short scope

## Branch Naming

    feature/<short-description>       e.g. feature/cvar-stress-tests
    fix/<short-description>           e.g. fix/var-backtesting-kupiec
    chore/<short-description>         e.g. chore/streamlit-cloud-setup
    docs/<short-description>          e.g. docs/mrm-framework-update

## Suggested Initial Commit Sequence for This Project

    chore: initial project structure and requirements

    feat(etl): synthetic data generator for portfolio, market, earnings, macro

    feat(rag): FAISS vector store with Titan Embeddings and LCEL chain

    feat(metrics): VaR, Sharpe, Beta, max drawdown, Sortino, Calmar

    feat(metrics): CVaR/Expected Shortfall via Monte Carlo with fat-tail mixture

    feat(metrics): historical stress tests for GFC, COVID, rate shock, tech selloff

    feat(agents): Market Analysis Agent using LangGraph StateGraph

    feat(agents): Risk Assessment Agent with VaR, CVaR, and stress test integration

    feat(agents): Portfolio Management Agent with rebalancing recommendations

    feat(agents): multi-agent orchestrator chaining market, risk, portfolio agents

    feat(mrm): SR 11-7 model inventory with risk tiering and use policy

    feat(mrm): independent validation suite with conceptual soundness and backtesting

    feat(mrm): ongoing monitoring with PSI drift detection and expiry alerts

    feat(mrm): governance module with change management and audit trail

    feat(mrm): CRO report, model cards, and regulatory exam package

    feat(api): FastAPI backend with chat, insights, and MRM endpoints

    feat(streamlit): analyst dashboard with portfolio, risk, and MRM pages

    feat(streamlit): demo mode with no-AWS fallback for Streamlit Cloud deploy

    feat(infra): AWS CDK stack with S3, Lambda, API Gateway, KMS, CloudTrail

    docs: README, compliance documentation, MRM framework guide, deploy guide
