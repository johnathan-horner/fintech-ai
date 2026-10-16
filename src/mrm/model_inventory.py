"""
FinTech AI - Model Risk Management (MRM) Framework
Implements SR 11-7 / OCC 2011-12 guidance for AI model governance.

SR 11-7 defines three components of sound model risk management:
  1. Model Development, Implementation, and Use
  2. Model Validation (independent)
  3. Governance, Policies, and Controls

This module covers the Model Inventory - the authoritative registry of all
models in use, their risk tier, validation status, and lifecycle stage.

Validation status is NOT hand-written: it is derived from
data/mrm/approvals.json, which only the eval harness (scripts/run_eval.py)
writes. ``require_approved`` enforces the approved version before every
model call.
"""

import json
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from src.llm import config
from src.llm.guard import GovernanceBlocked
from src.mrm.approvals import approvals_mtime, approvals_path, as_list, load_approvals
from src.utils.data import DATA_DIR

INVENTORY_PATH = os.path.join(DATA_DIR, "mrm_inventory.json")


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


APPROVAL_TO_VALIDATION_STATUS = {
    "APPROVED": ValidationStatus.APPROVED,
    "CONDITIONAL": ValidationStatus.CONDITIONAL,
    "FAILED": ValidationStatus.FAILED,
}

LLM_COMPONENTS = ("MRM-001", "MRM-002", "MRM-003", "MRM-004", "MRM-005")


#  Model Definitions (static documentation; status fields are derived)

FINTECH_AI_MODELS = [
    {
        "model_id": "MRM-001",
        "name": "RAG Financial Q&A Chain",
        "type": ModelType.RAG,
        "tier": ModelTier.TIER_2,
        "description": (
            "RAG chain: HyDE retrieval over a FAISS index of the bundled synthetic data, "
            "local cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2), answer generation with the "
            "configured chat model (demo: Gemini 2.5 Flash free tier; production target: Amazon "
            "Bedrock Claude) and an independent grader model scoring every answer."
        ),
        "owner": "Johnathan Horner",
        "business_use": "Analyst decision support - market and portfolio Q&A",
        "inputs": ["Natural language query", "FAISS vector store", "Conversation history (last 5 turns)"],
        "outputs": ["Natural language analysis with source citations", "Re-ranked source list with scores", "Grader result"],
        "dependencies": ["Chat model (config)", "HyDE model (config)", "Embedding model (config)", "FAISS", "fastembed cross-encoder"],
        "data_sources": ["Synthetic portfolio data", "Synthetic market data", "Synthetic earnings transcripts", "Synthetic macro data", "Computed risk metrics"],
        "status": ModelStatus.PRODUCTION,
        "validation_frequency_months": 12,
        "known_limitations": [
            "Hallucination risk on specific numeric outputs - mitigated by grader + deterministic numeric check",
            "Conversation memory limited to the last 5 turns",
            "FAISS index must be rebuilt when underlying data or embedding model changes (manifest check enforces this)",
            "Bundled data is synthetic and static - no live market feed",
        ],
        "mitigations": [
            "Investment disclaimer appended to all API outputs",
            "Source ids and re-ranker scores returned alongside every answer",
            "Independent grader model (different model id) + numeric consistency check on every answer",
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
            "LangGraph agent node using the configured chat model. Analyses sector exposure, "
            "technical indicators (RSI, SMA, MACD), earnings beat/miss and macro backdrop from "
            "the bundled data and computed metrics."
        ),
        "owner": "Johnathan Horner",
        "business_use": "Market trend identification and technical alert generation",
        "inputs": ["Portfolio positions JSON", "Computed KPIs and technical signals", "Latest earnings and macro records"],
        "outputs": ["Market analysis narrative", "Technical alerts list", "Sector breakdown"],
        "dependencies": ["Chat model (config)", "src/utils/metrics.py", "LangGraph"],
        "data_sources": ["Synthetic market OHLCV data", "Synthetic earnings transcripts", "Synthetic macro data"],
        "status": ModelStatus.PRODUCTION,
        "validation_frequency_months": 12,
        "known_limitations": [
            "Technical indicators are derived from synthetic data - not real market feeds",
            "LLM may over-interpret noise as signal in volatile market regimes",
            "No backtesting infrastructure for signal validation",
            "Sector assignments are static, not dynamically updated",
        ],
        "mitigations": [
            "Outputs labeled as AI-generated analysis, not trading signals",
            "Technical alerts are computed deterministically; the LLM only narrates them",
            "Every output graded by an independent model against its exact inputs",
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
            "LangGraph agent node that narrates portfolio VaR, Monte Carlo CVaR, stress tests and "
            "position risk scores computed by MRM-006. Directly informs risk discussions."
        ),
        "owner": "Johnathan Horner",
        "business_use": "Portfolio risk narrative over computed VaR / CVaR / stress results",
        "inputs": ["Computed risk metrics (MRM-006)", "Macro indicator data", "Market analysis output"],
        "outputs": ["Risk report narrative", "Position risk scores (computed, 0-100)"],
        "dependencies": ["Chat model (config)", "MRM-006 Financial Metrics Engine", "MRM-002 Market Analysis Agent"],
        "data_sources": ["Synthetic portfolio data", "Synthetic macro indicators"],
        "status": ModelStatus.PRODUCTION,
        "validation_frequency_months": 6,
        "known_limitations": [
            "Portfolio VaR shown in the UI is the simplified beta x assumed-vol estimate from MRM-006",
            "LLM risk narrative may not reflect the quantitative metrics exactly - graded on every run",
            "Stress scenarios are stylised historical shocks applied through beta, not full revaluation",
        ],
        "mitigations": [
            "All numbers in the prompt come from src/utils/metrics.py",
            "Independent grader + numeric check flag unsupported numbers",
            "Live use blocked unless the exact model + prompt version is approved by the eval harness",
        ],
        "materiality_rationale": (
            "Tier 1: Risk narratives may inform risk limit discussions. "
            "Requires independent validation and approval before production use."
        ),
    },
    {
        "model_id": "MRM-004",
        "name": "Portfolio Management Agent",
        "type": ModelType.LLM_AGENT,
        "tier": ModelTier.TIER_1,
        "description": (
            "LangGraph agent node suggesting risk reductions, concentration adjustments and "
            "tax-loss harvesting candidates for portfolio manager review."
        ),
        "owner": "Johnathan Horner",
        "business_use": "Portfolio rebalancing suggestions for human review",
        "inputs": ["Portfolio data", "Computed risk scores", "Concentration vs policy limits", "Upstream agent summaries"],
        "outputs": ["Recommendation memo", "Action items (computed from risk scores)"],
        "dependencies": ["Chat model (config)", "MRM-002", "MRM-003", "MRM-006"],
        "data_sources": ["Synthetic portfolio data", "All upstream agent outputs"],
        "status": ModelStatus.STAGING,
        "validation_frequency_months": 6,
        "known_limitations": [
            "Recommendations are LLM-generated - not optimized via quantitative portfolio theory",
            "Does not account for transaction costs, market impact, or liquidity constraints",
            "Tax-loss harvesting suggestions require CPA/tax advisor review",
            "No integration with order management system (OMS) - manual execution only",
        ],
        "mitigations": [
            "All recommendations include mandatory investment disclaimer",
            "Portfolio manager must review and approve all actions before execution",
            "No automated order routing - human approval gate required",
        ],
        "materiality_rationale": (
            "Tier 1: Recommendations directly influence portfolio construction. "
            "Requires validation of itself and upstream models before production promotion."
        ),
    },
    {
        "model_id": "MRM-005",
        "name": "Multi-Agent Orchestrator",
        "type": ModelType.ORCHESTRATION,
        "tier": ModelTier.TIER_1,
        "description": (
            "LangGraph StateGraph running market_analysis -> risk_assessment -> portfolio_management -> END "
            "with shared state and error propagation."
        ),
        "owner": "Johnathan Horner",
        "business_use": "End-to-end AI analysis pipeline execution",
        "inputs": ["Portfolio data JSON"],
        "outputs": ["Combined output of all three agents with per-section traces"],
        "dependencies": ["MRM-002", "MRM-003", "MRM-004", "LangGraph"],
        "data_sources": ["All upstream sources"],
        "status": ModelStatus.STAGING,
        "validation_frequency_months": 6,
        "known_limitations": [
            "Errors in upstream agents propagate - sections fall back to deterministic text",
            "Sequential execution - total latency is the sum of three model calls plus three grader calls",
            "No circuit breaker beyond the per-session request cap",
        ],
        "mitigations": [
            "Error list captured in state and surfaced to dashboard",
            "Each agent fails gracefully with a deterministic, data-derived fallback",
            "Each agent is independently gated by its own approval record",
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
            "Python quantitative metrics library (src/utils/metrics.py): parametric VaR, Monte Carlo "
            "CVaR/Expected Shortfall (fat-tail mixture), historical and parametric rolling VaR, stress "
            "tests, Sharpe/Sortino/Calmar, Beta and position risk scores."
        ),
        "owner": "Johnathan Horner",
        "business_use": "Quantitative risk measurement supporting MRM-003",
        "inputs": ["Portfolio positions", "Synthetic daily price history (data/price_history.json)"],
        "outputs": ["VaR / CVaR estimates", "Stress test P&L", "Risk scores"],
        "dependencies": ["Python standard library (math, statistics)"],
        "data_sources": ["Portfolio positions JSON", "Synthetic price history"],
        "status": ModelStatus.PRODUCTION,
        "validation_frequency_months": 12,
        "known_limitations": [
            "Headline parametric VaR uses weighted beta x an assumed 1.2% market vol, not estimated covariance",
            "Sharpe calculation assumes i.i.d. returns - not valid in trending markets",
            "Backtest runs on synthetic price history, so it validates mechanics rather than real-market calibration",
            "Position risk score weighting (beta 35%, PnL 30%, concentration 25%, rating 10%) is heuristic, not empirically calibrated",
        ],
        "mitigations": [
            "Out-of-sample rolling-window VaR backtest with Kupiec POF and Basel traffic light",
            "CVaR/ES reported alongside VaR",
            "Methodology documented in docs/MRM_FRAMEWORK.md for independent review",
        ],
        "materiality_rationale": (
            "Tier 1: Quantitative outputs used in risk decisions. "
            "Full model validation of statistical methodology required."
        ),
    },
]


#  Current (running) version of each component

def current_version(inventory_id: str, provider: Optional[str] = None) -> dict:
    """model_id + prompt versions that would run for a component on ``provider`` (default gemini)."""
    from src.prompts import registry
    from src.prompts.templates import component_prompts

    prompts = registry.refs(component_prompts(inventory_id, provider or "gemini"))
    if inventory_id not in LLM_COMPONENTS:
        return {"model_id": "src/utils/metrics.py", "prompt_versions": prompts, "provider": "code"}
    provider = provider or "gemini"
    if inventory_id == "MRM-001":
        mid = (f"{config.model_id('chat', provider)}+hyde:{config.model_id('hyde', provider)}"
               f"+emb:{config.model_id('embedding')}")
    else:
        mid = config.model_id("chat", provider)
    return {"model_id": mid, "prompt_versions": prompts, "provider": provider}


def current_versions(inventory_id: str) -> Dict[str, dict]:
    if inventory_id not in LLM_COMPONENTS:
        return {"code": current_version(inventory_id)}
    return {p: current_version(inventory_id, p) for p in config.provider_order()}


def _add_months(date_str: str, months: int) -> str:
    d = datetime.fromisoformat(date_str[:19])
    return (d + timedelta(days=round(months * 30.44))).strftime("%Y-%m-%d")


_STATUS_RANK = {"APPROVED": 0, "CONDITIONAL": 1, "FAILED": 2}


#  Inventory Manager

class ModelInventory:
    """
    SR 11-7 compliant model inventory.
    Tracks all models, their risk tiers, validation status per provider, and lifecycle.
    """

    def __init__(self, approvals: Optional[dict] = None):
        self.approvals = load_approvals() if approvals is None else approvals
        self.models = {}
        for m in FINTECH_AI_MODELS:
            mid = m["model_id"]
            model = dict(m)
            records = as_list(self.approvals.get(mid))
            versions = current_versions(mid)
            per_provider = {p: check_approval(mid, v["model_id"], v["prompt_versions"], self.approvals, provider=p)
                            for p, v in versions.items()}
            allowed = [(p, r) for p, r in per_provider.items() if r["allowed"]]
            matching = [r for r in records if any(_matches(mid, r, p, v) for p, v in versions.items())]
            model["current_versions"] = versions
            model["current_version"] = next(iter(versions.values()))
            model["approved_versions"] = [
                {k: r.get(k) for k in ("provider", "model_id", "prompt_versions", "approved_at", "approved_by",
                                       "evidence", "status", "metrics")}
                for r in records
            ]
            model["provider_status"] = per_provider
            model["version_mismatch"] = bool(records) and not matching
            best = sorted(matching, key=lambda r: (_STATUS_RANK.get(r["status"], 9), -_ts(r)))
            appr = best[0] if best else (records[-1] if records else None)
            model["approved_version"] = (
                {k: appr.get(k) for k in ("provider", "model_id", "prompt_versions", "approved_at", "approved_by",
                                          "evidence", "status", "metrics")} if appr else None)
            if allowed:
                statuses = [r["status"] for _, r in allowed]
                model["validation_status"] = (ValidationStatus.APPROVED if "APPROVED" in statuses
                                              else ValidationStatus.CONDITIONAL)
            elif matching:
                model["validation_status"] = ValidationStatus.FAILED
            else:
                model["validation_status"] = ValidationStatus.PENDING
            model["validation_date"] = ((appr or {}).get("approved_at") or "")[:10] or None
            model["next_validation_due"] = (
                _add_months(appr["approved_at"], m["validation_frequency_months"])
                if appr and appr.get("approved_at") else None
            )
            self.models[mid] = model

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
            json.dump({"models": self.get_all(), "exported_at": datetime.today().isoformat()}, f, indent=2, default=str)
        print(f"Model inventory exported to {path}")


def _ts(record: dict) -> float:
    try:
        return datetime.fromisoformat((record.get("approved_at") or "")[:19]).timestamp()
    except ValueError:
        return 0.0


def _matches(inventory_id: str, record: dict, provider: str, version: dict) -> bool:
    same_provider = inventory_id not in LLM_COMPONENTS or record.get("provider", "gemini") == provider
    return (same_provider and record.get("model_id") == version["model_id"]
            and sorted(record.get("prompt_versions") or []) == sorted(version["prompt_versions"] or []))


#  Approval gate

def check_approval(inventory_id: str, model_id: str, prompt_versions: List[str], approvals: Optional[dict] = None,
                   provider: str = "gemini") -> dict:
    """
    Decide whether ``inventory_id`` may call ``model_id`` on ``provider`` with ``prompt_versions``.
    Approvals are keyed by (component, provider, model_id, prompt_versions).
    Returns {"allowed": bool, "status": str, "reason": str, "record": dict|None}.
    """
    approvals = load_approvals() if approvals is None else approvals
    records = as_list(approvals.get(inventory_id))
    label = config.provider_label(provider, model_id.split("+")[0]) if inventory_id in LLM_COMPONENTS else "code"
    version = {"model_id": model_id, "prompt_versions": prompt_versions}
    matching = sorted((r for r in records if _matches(inventory_id, r, provider, version)), key=_ts, reverse=True)
    if not matching:
        same_provider = [r for r in records if inventory_id not in LLM_COMPONENTS or r.get("provider") == provider]
        if not same_provider:
            return {"allowed": False, "status": "NOT_VALIDATED", "record": None,
                    "reason": f"{inventory_id} is not yet validated for {label} (no approval record from the eval harness)"}
        latest = max(same_provider, key=_ts)
        return {"allowed": False, "status": "VERSION_MISMATCH", "record": None,
                "reason": (f"{inventory_id} running version ({model_id}, prompts {sorted(prompt_versions or [])}) "
                           f"differs from approved ({latest.get('model_id')}, prompts "
                           f"{sorted(latest.get('prompt_versions') or [])})")}
    rec = matching[0]
    status = rec.get("status")
    if status not in ("APPROVED", "CONDITIONAL"):
        return {"allowed": False, "status": status, "record": rec,
                "reason": f"{inventory_id} on {label} validation status is {status} (eval run {rec.get('evidence')})"}
    return {"allowed": True, "status": status, "record": rec,
            "reason": f"{inventory_id} {status} for {label} (eval run {rec.get('evidence')})"}


def require_approved(inventory_id: str, model_id: str, prompt_versions: List[str], approvals: Optional[dict] = None,
                     provider: str = "gemini") -> dict:
    """Run BEFORE any model call. Raises GovernanceBlocked when not allowed."""
    result = check_approval(inventory_id, model_id, prompt_versions, approvals, provider=provider)
    if not result["allowed"]:
        raise GovernanceBlocked(result["reason"])
    return result


def require_current_approved(inventory_id: str, provider: str = "gemini") -> dict:
    """require_approved() for the version that would run on ``provider``."""
    cur = current_version(inventory_id, provider)
    return require_approved(inventory_id, cur["model_id"], cur["prompt_versions"], provider=provider)


# Singleton (rebuilt when approvals.json changes)
_inventory = None
_inventory_mtime = None


def get_inventory() -> ModelInventory:
    global _inventory, _inventory_mtime
    mtime = (approvals_path(), approvals_mtime(), tuple(config.provider_order()))
    if _inventory is None or mtime != _inventory_mtime:
        _inventory = ModelInventory()
        _inventory_mtime = mtime
    return _inventory


if __name__ == "__main__":
    inv = get_inventory()
    stats = inv.summary_stats()
    print("=== FinTech AI Model Inventory ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
