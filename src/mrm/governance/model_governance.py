"""
FinTech AI - MRM Governance
SR 11-7 governance: change management, approval workflow, audit trail,
model use policy, and escalation procedures.

Governance components:
  1. Change Management - track and approve all model changes
  2. Use Policy - defines who can use which models and under what conditions
  3. Escalation Procedures - what to do when alerts fire
  4. Audit Trail - immutable log of all governance actions
  5. Exception Management - handle use of models under conditional approval
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Optional
from enum import Enum

GOVERNANCE_LOG_PATH = os.path.join(os.path.dirname(__file__), "../../../logs/mrm_governance_log.json")
os.makedirs(os.path.dirname(GOVERNANCE_LOG_PATH), exist_ok=True)


#  Enums 

class ChangeType(str, Enum):
    MODEL_UPDATE = "Model Update"
    PARAMETER_CHANGE = "Parameter Change"
    DATA_SOURCE_CHANGE = "Data Source Change"
    INFRASTRUCTURE_CHANGE = "Infrastructure Change"
    THRESHOLD_CHANGE = "Threshold Change"
    NEW_USE_CASE = "New Use Case"
    DEPENDENCY_CHANGE = "Dependency Change"


class ApprovalStatus(str, Enum):
    PENDING = "Pending Approval"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    ESCALATED = "Escalated to CRO"


class EscalationLevel(str, Enum):
    L1_ANALYST = "L1 - Analyst Self-Resolution"
    L2_MODEL_OWNER = "L2 - Model Owner"
    L3_MRM_TEAM = "L3 - MRM Validation Team"
    L4_CRO = "L4 - Chief Risk Officer"
    L5_BOARD = "L5 - Board Risk Committee"


#  Use Policy 

MODEL_USE_POLICY = {
    "MRM-001": {
        "approved_users": ["analyst", "portfolio_manager", "risk_manager"],
        "approved_uses": [
            "Natural language Q&A about portfolio and market data",
            "Preliminary research and hypothesis generation",
            "Summarizing earnings transcripts and macro data",
        ],
        "prohibited_uses": [
            "Generating legally binding investment advice",
            "Direct trading signal generation without analyst review",
            "Client-facing reports without human review and approval",
        ],
        "conditions": [
            "All outputs must be reviewed by a qualified analyst before action",
            "Investment disclaimer must be attached to all client-shared outputs",
            "Model outputs may not be represented as guaranteed predictions",
        ],
        "approval_required_for": [],
    },
    "MRM-002": {
        "approved_users": ["analyst", "portfolio_manager", "risk_manager"],
        "approved_uses": [
            "Internal market trend analysis",
            "Technical alert generation for analyst review",
            "Sector performance comparison",
        ],
        "prohibited_uses": [
            "Automated trading signal routing",
            "External client reporting without human validation",
        ],
        "conditions": [
            "Technical alerts require analyst confirmation before any action",
            "Outputs labeled as AI-generated - not verified market data",
        ],
        "approval_required_for": ["New use cases not listed above"],
    },
    "MRM-003": {
        "approved_users": ["risk_manager", "portfolio_manager"],
        "approved_uses": [
            "Internal portfolio risk monitoring (directional/informational only)",
            "Risk dashboard display for internal stakeholders",
            "Pre-trade risk screening (human-reviewed)",
        ],
        "prohibited_uses": [
            "Regulatory VaR reporting to external bodies (not validated)",
            "Automated risk limit enforcement",
            "Any use in client-facing regulatory documents",
        ],
        "conditions": [
            "PENDING VALIDATION - use is conditional and informational only",
            "VaR outputs labeled 'simplified estimate - not for regulatory reporting'",
            "Independent validation must be completed before Tier 1 promotion",
            "CRO sign-off required for any expansion of use cases",
        ],
        "approval_required_for": [
            "Any use in regulatory reporting",
            "Use in automated risk limit decisions",
            "Expansion beyond internal dashboard",
        ],
    },
    "MRM-004": {
        "approved_users": ["portfolio_manager"],
        "approved_uses": [
            "Internal portfolio rebalancing discussion support",
            "Generating recommendation memos for portfolio manager review",
        ],
        "prohibited_uses": [
            "Automated order generation or routing",
            "Client portfolio management without human override",
            "Performance attribution or benchmark comparison reporting",
        ],
        "conditions": [
            "STATUS: STAGING - not in production",
            "All recommendations require portfolio manager sign-off",
            "No integration with OMS until fully validated",
            "MRM-003 validation must be completed first",
        ],
        "approval_required_for": ["Any production use"],
    },
    "MRM-005": {
        "approved_users": ["portfolio_manager", "risk_manager"],
        "approved_uses": ["Internal comprehensive analysis pipeline"],
        "prohibited_uses": ["Production use until all component models validated"],
        "conditions": ["STATUS: STAGING - requires validation of MRM-002, MRM-003, MRM-004, and MRM-005 itself"],
        "approval_required_for": ["Any production use"],
    },
    "MRM-006": {
        "approved_users": ["risk_manager", "analyst"],
        "approved_uses": [
            "Internal risk scoring and directional VaR estimation",
            "Portfolio risk dashboard display",
        ],
        "prohibited_uses": [
            "Regulatory capital calculations",
            "FRTB or Basel III VaR reporting",
            "Automated stop-loss triggering",
        ],
        "conditions": [
            "PENDING VALIDATION - quant methodology under independent review",
            "All outputs must include 'simplified estimate' disclaimer",
        ],
        "approval_required_for": ["Any regulatory use", "Automated risk decisions"],
    },
}


#  Change Management 

class ModelChangeRequest:
    """SR 11-7: All material model changes require review and approval."""

    def __init__(
        self,
        model_id: str,
        change_type: ChangeType,
        description: str,
        requestor: str,
        materiality: str = "Minor",
    ):
        self.change_id = f"CHG-{uuid.uuid4().hex[:8].upper()}"
        self.model_id = model_id
        self.change_type = change_type
        self.description = description
        self.requestor = requestor
        self.materiality = materiality  # Minor | Moderate | Major
        self.status = ApprovalStatus.PENDING
        self.created_at = datetime.today().isoformat()
        self.approved_by: Optional[str] = None
        self.approved_at: Optional[str] = None
        self.rejection_reason: Optional[str] = None
        self.requires_revalidation = materiality in ["Moderate", "Major"]

    def approve(self, approver: str):
        self.status = ApprovalStatus.APPROVED
        self.approved_by = approver
        self.approved_at = datetime.today().isoformat()
        log_governance_action(
            action="CHANGE_APPROVED",
            model_id=self.model_id,
            actor=approver,
            details={"change_id": self.change_id, "change_type": self.change_type, "description": self.description},
        )
        if self.requires_revalidation:
            log_governance_action(
                action="REVALIDATION_TRIGGERED",
                model_id=self.model_id,
                actor="SYSTEM",
                details={"change_id": self.change_id, "reason": f"Material change ({self.materiality}) requires re-validation."},
            )

    def reject(self, approver: str, reason: str):
        self.status = ApprovalStatus.REJECTED
        self.approved_by = approver
        self.approved_at = datetime.today().isoformat()
        self.rejection_reason = reason
        log_governance_action(
            action="CHANGE_REJECTED",
            model_id=self.model_id,
            actor=approver,
            details={"change_id": self.change_id, "reason": reason},
        )

    def to_dict(self) -> dict:
        return {
            "change_id": self.change_id,
            "model_id": self.model_id,
            "change_type": self.change_type,
            "description": self.description,
            "requestor": self.requestor,
            "materiality": self.materiality,
            "status": self.status,
            "requires_revalidation": self.requires_revalidation,
            "created_at": self.created_at,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "rejection_reason": self.rejection_reason,
        }


#  Audit Trail 

def log_governance_action(action: str, model_id: str, actor: str, details: dict):
    """Append an immutable governance action to the audit log."""
    entry = {
        "event_id": uuid.uuid4().hex,
        "timestamp": datetime.today().isoformat(),
        "action": action,
        "model_id": model_id,
        "actor": actor,
        "details": details,
    }

    # Load existing log
    log = []
    if os.path.exists(GOVERNANCE_LOG_PATH):
        try:
            with open(GOVERNANCE_LOG_PATH) as f:
                log = json.load(f)
        except Exception:
            log = []

    log.append(entry)

    with open(GOVERNANCE_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def get_audit_log(model_id: Optional[str] = None) -> List[dict]:
    """Retrieve the audit log, optionally filtered by model."""
    if not os.path.exists(GOVERNANCE_LOG_PATH):
        return []
    with open(GOVERNANCE_LOG_PATH) as f:
        log = json.load(f)
    if model_id:
        log = [e for e in log if e.get("model_id") == model_id]
    return log


#  Escalation Procedures 

ESCALATION_PROCEDURES = {
    "VAR_BREACH": {
        "description": "Actual portfolio loss exceeds VaR estimate",
        "trigger": "Daily P&L exceeds VaR at the stated confidence level for 3+ consecutive days",
        "immediate_action": "Notify risk manager and model owner within 1 hour",
        "escalation_path": [
            EscalationLevel.L2_MODEL_OWNER,
            EscalationLevel.L3_MRM_TEAM,
            EscalationLevel.L4_CRO,
        ],
        "documentation": "Complete incident report within 24 hours. Trigger re-validation within 5 business days.",
    },
    "MODEL_ERROR_CRITICAL": {
        "description": "Model error rate exceeds 15%",
        "trigger": "30-day rolling error rate > 15%",
        "immediate_action": "Suspend model use and notify model owner",
        "escalation_path": [
            EscalationLevel.L2_MODEL_OWNER,
            EscalationLevel.L3_MRM_TEAM,
        ],
        "documentation": "Root cause analysis required within 48 hours before model can be reinstated.",
    },
    "INPUT_DRIFT_CRITICAL": {
        "description": "Input distribution drift PSI > 0.25",
        "trigger": "PSI exceeds 0.25 on any monitored feature distribution",
        "immediate_action": "Flag all model outputs as 'UNDER REVIEW' in dashboard",
        "escalation_path": [
            EscalationLevel.L2_MODEL_OWNER,
            EscalationLevel.L3_MRM_TEAM,
        ],
        "documentation": "Re-validation required before continued Tier 1 use.",
    },
    "VALIDATION_EXPIRED": {
        "description": "Model validation period has expired",
        "trigger": "next_validation_due date has passed",
        "immediate_action": "For Tier 1: suspend production use. For Tier 2/3: add output warning label.",
        "escalation_path": [
            EscalationLevel.L2_MODEL_OWNER,
            EscalationLevel.L3_MRM_TEAM,
            EscalationLevel.L4_CRO,  # Tier 1 only
        ],
        "documentation": "Validation must be completed or model formally retired.",
    },
    "UNAUTHORIZED_USE": {
        "description": "Model used outside approved use policy",
        "trigger": "Query type or user role outside MODEL_USE_POLICY",
        "immediate_action": "Block request. Log incident. Notify model owner.",
        "escalation_path": [
            EscalationLevel.L2_MODEL_OWNER,
            EscalationLevel.L3_MRM_TEAM,
            EscalationLevel.L4_CRO,
        ],
        "documentation": "Mandatory incident report. Compliance review within 5 business days.",
    },
}


def check_use_policy(model_id: str, user_role: str, use_case: str) -> dict:
    """
    Check if a model use is within approved policy.
    Returns {compliant: bool, reason: str, conditions: list}
    """
    policy = MODEL_USE_POLICY.get(model_id)
    if not policy:
        return {"compliant": False, "reason": f"Model {model_id} has no use policy defined.", "conditions": []}

    if user_role not in policy.get("approved_users", []):
        log_governance_action(
            action="POLICY_VIOLATION_USER",
            model_id=model_id,
            actor=user_role,
            details={"use_case": use_case, "reason": f"User role '{user_role}' not in approved list."},
        )
        return {
            "compliant": False,
            "reason": f"User role '{user_role}' is not approved for model {model_id}.",
            "conditions": [],
        }

    for prohibited in policy.get("prohibited_uses", []):
        if any(word in use_case.lower() for word in prohibited.lower().split()[:3]):
            log_governance_action(
                action="POLICY_VIOLATION_USE",
                model_id=model_id,
                actor=user_role,
                details={"use_case": use_case, "prohibited_match": prohibited},
            )
            return {
                "compliant": False,
                "reason": f"Use case matches prohibited use: '{prohibited}'.",
                "conditions": [],
            }

    return {
        "compliant": True,
        "reason": "Use is within approved policy.",
        "conditions": policy.get("conditions", []),
    }


def get_governance_summary() -> dict:
    """Return a summary of governance status across all models."""
    from src.mrm.model_inventory import get_inventory, ValidationStatus, ModelTier

    inv = get_inventory()
    all_models = inv.get_all()

    tier1_in_prod_without_approval = [
        m for m in all_models
        if m["tier"] == ModelTier.TIER_1
        and m["status"].value in ["Production", "Staging"]
        and m["validation_status"] in [ValidationStatus.PENDING, ValidationStatus.FAILED, ValidationStatus.EXPIRED]
    ]

    return {
        "total_models": len(all_models),
        "tier1_violations": [
            {
                "model_id": m["model_id"],
                "name": m["name"],
                "issue": f"Tier 1 model in {m['status']} with validation status: {m['validation_status']}",
                "severity": "CRITICAL",
            }
            for m in tier1_in_prod_without_approval
        ],
        "pending_change_requests": 0,  # Would query change request store in prod
        "audit_log_entries": len(get_audit_log()),
        "escalation_procedures": list(ESCALATION_PROCEDURES.keys()),
    }


if __name__ == "__main__":
    summary = get_governance_summary()
    print("=== MRM Governance Summary ===")
    print(f"Total Models: {summary['total_models']}")
    print(f"Tier 1 Violations: {len(summary['tier1_violations'])}")
    for v in summary["tier1_violations"]:
        print(f"  [{v['severity']}] {v['model_id']}: {v['issue']}")
    print(f"Audit Log Entries: {summary['audit_log_entries']}")

    # Example: check use policy
    result = check_use_policy("MRM-003", "analyst", "regulatory VaR reporting")
    print(f"\nPolicy Check (MRM-003, analyst, regulatory VaR): {result}")
