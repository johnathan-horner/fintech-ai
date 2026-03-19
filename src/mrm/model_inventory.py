"""
FinTech AI - Model Risk Management (MRM) Framework
Implements SR 11-7 / OCC 2011-12 guidance for AI model governance.

SR 11-7 defines three components of sound model risk management:
  1. Model Development, Implementation, and Use
  2. Model Validation (independent)
  3. Governance, Policies, and Controls

This module covers the Model Inventory - the authoritative registry of all
models in use, their risk tier, validation status, and lifecycle stage.
"""

import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum

INVENTORY_PATH = os.path.join(os.path.dirname(__file__), "../../../data/mrm_inventory.json")


#  Enums 

class ModelTier(str, Enum):
    """
    SR 11-7 risk-tiering. Higher tier = higher scrutiny + more frequent validation.
    Tier 1: High materiality - directly drives trading/credit decisions
    Tier 2: Medium materiality - supports decisions, human-in-the-loop
    Tier 3: Low materiality - informational/analytical only
    """
    TIER_1 = "Tier 1 - High"
    TIER_2 = "Tier 2 - Medium"
    TIER_3 = "Tier 3 - Low"


class ValidationStatus(str, Enum):
    APPROVED = "Approved"
    CONDITIONAL = "Conditional Approval"
    PENDING = "Pending Validation"
    FAILED = "Validation Failed"
    EXPIRED = "Validation Expired"
    IN_REVIEW = "Under Review"


class ModelStatus(str, Enum):
    DEVELOPMENT = "Development"
    STAGING = "Staging"
    PRODUCTION = "Production"
    RETIRED = "Retired"
    SUSPENDED = "Suspended"


class ModelType(str, Enum):
    RAG = "Retrieval-Augmented Generation"
    LLM_AGENT = "LLM Agent"
    QUANTITATIVE = "Quantitative/Statistical"
    ORCHESTRATION = "Multi-Agent Orchestration"


#  Model Definitions 

FINTECH_AI_MODELS = [
    {
        "model_id": "MRM-001",
        "name": "RAG Financial Q&A Chain",
        "type": ModelType.RAG,
        "tier": ModelTier.TIER_2,
        "description": (
            "LCEL RAG chain using Amazon Bedrock Claude 3 Sonnet + FAISS vector store. "
            "Answers analyst questions about portfolio, earnings, and macro data. "
            "Human-in-the-loop: analyst reviews and acts on outputs."
        ),
        "owner": "Johnathan Horner",
        "business_use": "Analyst decision support - market and portfolio Q&A",
        "inputs": ["Natural language query", "FAISS vector store", "Conversation history"],
        "outputs": ["Natural language analysis", "Source document references"],
        "dependencies": ["Amazon Bedrock Claude 3 Sonnet", "Titan Embeddings v1", "FAISS"],
        "data_sources": ["Synthetic portfolio data", "Synthetic earnings transcripts", "Synthetic macro data"],
        "status": ModelStatus.PRODUCTION,
        "validation_status": ValidationStatus.CONDITIONAL,
        "validation_date": "2025-01-15",
        "next_validation_due": "2026-01-15",
        "validation_frequency_months": 12,
        "last_reviewed": datetime.today().strftime("%Y-%m-%d"),
        "known_limitations": [
            "Hallucination risk on specific numeric outputs - must be cross-checked",
            "Context window limited to 5 conversation turns",
            "FAISS index requires rebuild when underlying data changes",
            "No real-time market data - index refresh latency up to 6 hours",
        ],
        "mitigations": [
            "Investment disclaimer appended to all outputs",
            "Source documents returned alongside every answer",
            "Human analyst review required before any trading action",
            "Guardrails filter blocked topics (insider trading, market manipulation)",
        ],
        "materiality_rationale": (
            "Tier 2: Supports decisions but does not execute trades. "
            "Analyst must independently validate any numeric output before acting."
        ),
    },
    {
        "model_id": "MRM-002",
        "name": "Market Analysis Agent",
        "type": ModelType.LLM_AGENT,
        "tier": ModelTier.TIER_2,
        "description": (
            "LangGraph LLM agent using Claude 3 Sonnet. Analyzes price trends, "
            "technical indicators (RSI, SMA, MACD), sector momentum, and earnings patterns."
        ),
        "owner": "Johnathan Horner",
        "business_use": "Market trend identification and technical alert generation",
        "inputs": ["Portfolio positions JSON", "FAISS-retrieved market context"],
        "outputs": ["Market analysis narrative", "Technical alerts list", "Sector breakdown"],
        "dependencies": ["Amazon Bedrock Claude 3 Sonnet", "FAISS", "MRM-001 vector store"],
        "data_sources": ["Synthetic market OHLCV data", "Synthetic earnings transcripts"],
        "status": ModelStatus.PRODUCTION,
        "validation_status": ValidationStatus.CONDITIONAL,
        "validation_date": "2025-01-15",
        "next_validation_due": "2026-01-15",
        "validation_frequency_months": 12,
        "last_reviewed": datetime.today().strftime("%Y-%m-%d"),
        "known_limitations": [
            "Technical indicators are derived from synthetic data - not real market feeds",
            "LLM may over-interpret noise as signal in volatile market regimes",
            "No backtesting infrastructure for signal validation",
            "Sector assignments are static, not dynamically updated",
        ],
        "mitigations": [
            "Outputs labeled as AI-generated analysis, not trading signals",
            "Technical alerts require analyst confirmation before action",
            "Sector breakdown cross-checked against portfolio management system",
        ],
        "materiality_rationale": (
            "Tier 2: Technical alerts are informational. "
            "No automated trading triggered by this model."
        ),
    },
    {
        "model_id": "MRM-003",
        "name": "Risk Assessment Agent",
        "type": ModelType.LLM_AGENT,
        "tier": ModelTier.TIER_1,
        "description": (
            "LangGraph agent calculating portfolio VaR, position-level risk scores, "
            "and generating risk reports. Directly informs risk limit decisions."
        ),
        "owner": "Johnathan Horner",
        "business_use": "Portfolio risk measurement, VaR calculation, risk limit monitoring",
        "inputs": ["Portfolio positions JSON", "Macro indicator data", "Market analysis output"],
        "outputs": ["Portfolio VaR (1-day, 10-day)", "Position risk scores (0-100)", "Risk report narrative"],
        "dependencies": [
            "Amazon Bedrock Claude 3 Sonnet",
            "src/utils/metrics.py (VaR, Sharpe, Beta calculations)",
            "MRM-002 Market Analysis Agent",
        ],
        "data_sources": ["Synthetic portfolio data", "Synthetic macro indicators"],
        "status": ModelStatus.PRODUCTION,
        "validation_status": ValidationStatus.PENDING,
        "validation_date": None,
        "next_validation_due": "2025-03-01",
        "validation_frequency_months": 6,
        "last_reviewed": datetime.today().strftime("%Y-%m-%d"),
        "known_limitations": [
            "VaR uses simplified parametric method (weighted beta * assumed market vol) - not full historical simulation",
            "Does not model correlation between positions (assumes linear additivity)",
            "LLM risk narrative may not reflect actual quantitative risk metrics",
            "No stress testing or scenario analysis implemented",
            "Backtesting of VaR predictions not yet implemented",
        ],
        "mitigations": [
            "VaR labeled as 'simplified estimate' in all outputs",
            "Risk scores are directional indicators, not precise measurements",
            "Independent risk team must validate VaR methodology before Tier 1 use",
            "PENDING validation - not approved for use in risk limit decisions until validated",
        ],
        "materiality_rationale": (
            "Tier 1: VaR outputs may inform risk limit decisions and regulatory reporting. "
            "Requires independent validation and approval before production use in limit monitoring."
        ),
    },
    {
        "model_id": "MRM-004",
        "name": "Portfolio Management Agent",
        "type": ModelType.LLM_AGENT,
        "tier": ModelTier.TIER_1,
        "description": (
            "LangGraph agent generating portfolio rebalancing recommendations, "
            "hedge strategies, entry/exit suggestions, and tax-loss harvesting opportunities."
        ),
        "owner": "Johnathan Horner",
        "business_use": "Portfolio optimization recommendations and rebalancing guidance",
        "inputs": ["Portfolio data", "Market analysis", "Risk assessment", "VaR output"],
        "outputs": ["Rebalancing plan", "Hedge recommendations", "Entry/exit suggestions", "Action items"],
        "dependencies": ["Amazon Bedrock Claude 3 Sonnet", "MRM-002", "MRM-003"],
        "data_sources": ["Synthetic portfolio data", "All upstream agent outputs"],
        "status": ModelStatus.STAGING,
        "validation_status": ValidationStatus.PENDING,
        "validation_date": None,
        "next_validation_due": "2025-03-01",
        "validation_frequency_months": 6,
        "last_reviewed": datetime.today().strftime("%Y-%m-%d"),
        "known_limitations": [
            "Recommendations are LLM-generated - not optimized via quantitative portfolio theory",
            "Does not account for transaction costs, market impact, or liquidity constraints",
            "Tax-loss harvesting suggestions require CPA/tax advisor review",
            "No integration with order management system (OMS) - manual execution only",
            "Dependent on MRM-003 (PENDING validation) - inherits its limitations",
        ],
        "mitigations": [
            "Status: STAGING - not in production until MRM-003 is validated",
            "All recommendations include mandatory investment disclaimer",
            "Portfolio manager must review and approve all actions before execution",
            "No automated order routing - human approval gate required",
        ],
        "materiality_rationale": (
            "Tier 1: Recommendations directly influence portfolio construction. "
            "Requires full validation of itself and all upstream models (MRM-002, MRM-003) "
            "before production promotion."
        ),
    },
    {
        "model_id": "MRM-005",
        "name": "Multi-Agent Orchestrator",
        "type": ModelType.ORCHESTRATION,
        "tier": ModelTier.TIER_1,
        "description": (
            "LangGraph StateGraph orchestrating MRM-002, MRM-003, and MRM-004 sequentially. "
            "Manages shared state and error propagation across the pipeline."
        ),
        "owner": "Johnathan Horner",
        "business_use": "End-to-end AI analysis pipeline execution",
        "inputs": ["Portfolio data JSON"],
        "outputs": ["Combined output of all three agents"],
        "dependencies": ["MRM-002", "MRM-003", "MRM-004", "LangGraph"],
        "data_sources": ["All upstream sources"],
        "status": ModelStatus.STAGING,
        "validation_status": ValidationStatus.PENDING,
        "validation_date": None,
        "next_validation_due": "2025-04-01",
        "validation_frequency_months": 6,
        "last_reviewed": datetime.today().strftime("%Y-%m-%d"),
        "known_limitations": [
            "Errors in upstream agents propagate - no compensation logic yet",
            "Sequential execution - if MRM-003 fails, MRM-004 receives incomplete data",
            "No circuit breaker for repeated failures",
            "Total latency 60-120 seconds - not suitable for real-time risk decisions",
        ],
        "mitigations": [
            "Error list captured in state and surfaced to dashboard",
            "Each agent fails gracefully with fallback messages",
            "Status: STAGING - production promotion blocked until all component models validated",
        ],
        "materiality_rationale": (
            "Tier 1: As orchestrator of Tier 1 models, inherits highest risk tier. "
            "Validation of orchestration logic required in addition to component model validation."
        ),
    },
    {
        "model_id": "MRM-006",
        "name": "Financial Metrics Engine",
        "type": ModelType.QUANTITATIVE,
        "tier": ModelTier.TIER_1,
        "description": (
            "Python-based quantitative metrics library. Calculates VaR (parametric), "
            "Sharpe Ratio (annualized), Beta (OLS regression), max drawdown, "
            "and position risk scores."
        ),
        "owner": "Johnathan Horner",
        "business_use": "Quantitative risk measurement supporting MRM-003",
        "inputs": ["Returns series", "Portfolio positions", "Market returns"],
        "outputs": ["VaR estimates", "Sharpe ratios", "Beta coefficients", "Risk scores"],
        "dependencies": ["Python standard library (math, statistics)"],
        "data_sources": ["Portfolio positions JSON"],
        "status": ModelStatus.PRODUCTION,
        "validation_status": ValidationStatus.PENDING,
        "validation_date": None,
        "next_validation_due": "2025-03-01",
        "validation_frequency_months": 12,
        "last_reviewed": datetime.today().strftime("%Y-%m-%d"),
        "known_limitations": [
            "VaR uses parametric normal distribution assumption - underestimates tail risk",
            "Sharpe calculation assumes i.i.d. returns - not valid in trending markets",
            "Beta estimated from limited synthetic data - unstable with short windows",
            "Position risk score weighting (beta 35%, PnL 30%, concentration 25%, rating 10%) is heuristic, not empirically calibrated",
            "No fat-tail or jump risk modeling (no CVaR/ES implementation)",
        ],
        "mitigations": [
            "All metric outputs include 'simplified estimate' labeling",
            "Independent quant validation of all formulas required before Tier 1 use",
            "Methodology documented in docs/MRM_VALIDATION.md for independent review",
        ],
        "materiality_rationale": (
            "Tier 1: Quantitative outputs used in risk decisions. "
            "Full model validation of statistical methodology required."
        ),
    },
]


#  Inventory Manager 

class ModelInventory:
    """
    SR 11-7 compliant model inventory.
    Tracks all models, their risk tiers, validation status, and lifecycle.
    """

    def __init__(self):
        self.models = {m["model_id"]: m for m in FINTECH_AI_MODELS}

    def get_all(self) -> list:
        return list(self.models.values())

    def get_by_id(self, model_id: str) -> Optional[dict]:
        return self.models.get(model_id)

    def get_by_tier(self, tier: ModelTier) -> list:
        return [m for m in self.models.values() if m["tier"] == tier]

    def get_by_status(self, status: ValidationStatus) -> list:
        return [m for m in self.models.values() if m["validation_status"] == status]

    def get_overdue_validations(self) -> list:
        """Return models whose validation is past due."""
        today = datetime.today().date()
        overdue = []
        for m in self.models.values():
            due = m.get("next_validation_due")
            if due and datetime.strptime(due, "%Y-%m-%d").date() < today:
                overdue.append(m)
        return overdue

    def get_production_models(self) -> list:
        return [m for m in self.models.values() if m["status"] == ModelStatus.PRODUCTION]

    def get_pending_validation(self) -> list:
        return [m for m in self.models.values()
                if m["validation_status"] in [ValidationStatus.PENDING, ValidationStatus.EXPIRED]]

    def summary_stats(self) -> dict:
        all_models = list(self.models.values())
        return {
            "total_models": len(all_models),
            "tier_1": len([m for m in all_models if m["tier"] == ModelTier.TIER_1]),
            "tier_2": len([m for m in all_models if m["tier"] == ModelTier.TIER_2]),
            "tier_3": len([m for m in all_models if m["tier"] == ModelTier.TIER_3]),
            "approved": len([m for m in all_models if m["validation_status"] == ValidationStatus.APPROVED]),
            "conditional": len([m for m in all_models if m["validation_status"] == ValidationStatus.CONDITIONAL]),
            "pending": len([m for m in all_models if m["validation_status"] == ValidationStatus.PENDING]),
            "failed": len([m for m in all_models if m["validation_status"] == ValidationStatus.FAILED]),
            "in_production": len([m for m in all_models if m["status"] == ModelStatus.PRODUCTION]),
            "in_staging": len([m for m in all_models if m["status"] == ModelStatus.STAGING]),
            "overdue_validations": len(self.get_overdue_validations()),
            "pending_validations": len(self.get_pending_validation()),
        }

    def export_json(self, path: str = INVENTORY_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"models": self.get_all(), "exported_at": datetime.today().isoformat()}, f, indent=2)
        print(f"Model inventory exported to {path}")


# Singleton
_inventory = None

def get_inventory() -> ModelInventory:
    global _inventory
    if _inventory is None:
        _inventory = ModelInventory()
    return _inventory


if __name__ == "__main__":
    inv = get_inventory()
    stats = inv.summary_stats()
    print("=== FinTech AI Model Inventory ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    inv.export_json()
