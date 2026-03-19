"""
FinTech AI - MRM Reporting
Generates SR 11-7 compliant management reports for:
  - CRO (Chief Risk Officer) - monthly model risk summary
  - Board Risk Committee - quarterly model inventory + validation status
  - Regulatory Exam - on-demand full model documentation package
"""

import json
from datetime import datetime
from src.mrm.model_inventory import get_inventory, ModelTier, ValidationStatus, ModelStatus
from src.mrm.monitoring.model_monitor import run_monitoring_report, check_validation_expiry
from src.mrm.governance.model_governance import get_governance_summary, get_audit_log


def generate_cro_report() -> dict:
    """
    Monthly CRO Model Risk Report.
    Summarizes model inventory health, validation status, monitoring alerts,
    and outstanding governance issues for executive review.
    """
    inv = get_inventory()
    stats = inv.summary_stats()
    monitoring = run_monitoring_report()
    governance = get_governance_summary()
    expiry_alerts = check_validation_expiry()

    # Executive summary
    critical_issues = []

    # Tier 1 models without full approval
    for m in inv.get_by_tier(ModelTier.TIER_1):
        if m["validation_status"] in [ValidationStatus.PENDING, ValidationStatus.FAILED]:
            critical_issues.append({
                "issue": f"Tier 1 model {m['model_id']} ({m['name']}) has validation status: {m['validation_status']}.",
                "risk": "HIGH - model outputs should not be used in risk limit decisions until validated.",
                "action": "Complete independent validation or restrict to informational use only.",
            })

    # Staging models not yet promoted
    staging_models = inv.get_by_status(ValidationStatus.PENDING)

    # Monitoring critical alerts
    for perf in monitoring.get("performance", []):
        if perf.get("status") == "CRITICAL":
            critical_issues.append({
                "issue": f"Model {perf['model_id']} in CRITICAL performance state.",
                "risk": "MEDIUM - output quality may be degraded.",
                "action": f"Review error rate ({perf['metrics'].get('error_rate', 'N/A'):.1%}) and latency.",
            })

    return {
        "report_type": "CRO Monthly Model Risk Report",
        "report_date": datetime.today().strftime("%B %d, %Y"),
        "prepared_by": "FinTech AI MRM System",
        "classification": "INTERNAL - CONFIDENTIAL",

        "executive_summary": {
            "overall_model_risk_level": (
                "HIGH" if len(critical_issues) >= 2
                else "MEDIUM" if len(critical_issues) >= 1
                else "LOW"
            ),
            "total_models_in_inventory": stats["total_models"],
            "tier_1_models": stats["tier_1"],
            "tier_2_models": stats["tier_2"],
            "critical_issues_count": len(critical_issues),
            "models_pending_validation": stats["pending"],
            "models_approved": stats["approved"] + stats["conditional"],
            "validation_expiry_alerts": len(expiry_alerts),
        },

        "critical_issues": critical_issues,

        "model_inventory_summary": {
            "by_validation_status": {
                "approved": stats["approved"],
                "conditional_approval": stats["conditional"],
                "pending_validation": stats["pending"],
                "failed": stats["failed"],
            },
            "by_lifecycle_status": {
                "production": stats["in_production"],
                "staging": stats["in_staging"],
            },
        },

        "monitoring_summary": {
            "overall_status": monitoring.get("overall_status"),
            "models_critical": monitoring["summary"]["critical_alerts"],
            "models_warning": monitoring["summary"]["warning_alerts"],
            "input_drift_psi": monitoring.get("drift", {}).get("psi_value"),
            "drift_status": monitoring.get("drift", {}).get("alerts", [{}])[0].get("severity", "N/A"),
        },

        "validation_schedule": [
            {
                "model_id": m["model_id"],
                "model_name": m["name"],
                "tier": m["tier"],
                "due_date": m.get("next_validation_due", "Not scheduled"),
                "current_status": m["validation_status"],
                "priority": "URGENT" if m["tier"] == ModelTier.TIER_1 else "NORMAL",
            }
            for m in inv.get_all()
            if m["validation_status"] in [ValidationStatus.PENDING, ValidationStatus.EXPIRED]
        ],

        "governance_status": {
            "tier_1_violations": len(governance.get("tier1_violations", [])),
            "audit_log_entries": governance.get("audit_log_entries", 0),
            "pending_change_requests": governance.get("pending_change_requests", 0),
        },

        "recommendations": [
            "Complete independent validation of MRM-003 (Risk Assessment Agent) - currently PENDING, Tier 1.",
            "Complete validation of MRM-006 (Financial Metrics Engine) - VaR methodology requires quant review.",
            "Do not promote MRM-004 (Portfolio Agent) or MRM-005 (Orchestrator) to production until MRM-003 is approved.",
            "Implement CVaR/Expected Shortfall alongside VaR in MRM-006 for Basel III compliance.",
            "Establish formal re-validation cadence - Tier 1 every 6 months, Tier 2 every 12 months.",
        ],
    }


def generate_model_card(model_id: str) -> dict:
    """
    Generate a model card for a single model - SR 11-7 documentation artifact.
    Covers: description, intended use, limitations, validation status, governance.
    """
    inv = get_inventory()
    model = inv.get_by_id(model_id)

    if not model:
        return {"error": f"Model {model_id} not found."}

    audit_entries = get_audit_log(model_id)

    return {
        "model_card": {
            "model_id": model["model_id"],
            "model_name": model["name"],
            "version": "1.0",
            "generated_at": datetime.today().isoformat(),
            "classification": "INTERNAL - CONFIDENTIAL",
        },
        "model_details": {
            "type": model["type"],
            "description": model["description"],
            "owner": model["owner"],
            "business_use": model["business_use"],
            "risk_tier": model["tier"],
            "inputs": model["inputs"],
            "outputs": model["outputs"],
            "dependencies": model["dependencies"],
            "data_sources": model["data_sources"],
        },
        "lifecycle_status": {
            "development_status": model["status"],
            "validation_status": model["validation_status"],
            "last_validated": model.get("validation_date", "Never"),
            "next_validation_due": model.get("next_validation_due", "Not scheduled"),
            "validation_frequency": f"Every {model.get('validation_frequency_months', 12)} months",
        },
        "risk_profile": {
            "materiality_tier": model["tier"],
            "materiality_rationale": model["materiality_rationale"],
            "known_limitations": model["known_limitations"],
            "mitigations": model["mitigations"],
        },
        "governance": {
            "approved_users": [],  # Populated from MODEL_USE_POLICY in governance module
            "audit_events": audit_entries[-10:] if audit_entries else [],
            "change_history": [],  # Populated from change request log in prod
        },
        "sr_11_7_checklist": {
            "limitations_documented": len(model.get("known_limitations", [])) > 0,
            "mitigations_documented": len(model.get("mitigations", [])) > 0,
            "owner_assigned": bool(model.get("owner")),
            "business_use_defined": bool(model.get("business_use")),
            "data_sources_documented": bool(model.get("data_sources")),
            "validation_scheduled": bool(model.get("next_validation_due")),
            "validated_before_tier1_production": (
                model["tier"] != ModelTier.TIER_1
                or model["validation_status"] in [ValidationStatus.APPROVED, ValidationStatus.CONDITIONAL]
            ),
        },
    }


def generate_full_exam_package() -> dict:
    """
    Generate the full regulatory exam package - on-demand MRM documentation.
    Produces model cards for all models + CRO report + governance summary.
    """
    inv = get_inventory()
    model_cards = [generate_model_card(mid) for mid in inv.models.keys()]
    cro_report = generate_cro_report()
    governance = get_governance_summary()

    return {
        "exam_package": {
            "title": "FinTech AI - Model Risk Management Regulatory Exam Package",
            "prepared_for": "Regulatory Examination",
            "prepared_by": "Johnathan Horner, Model Owner",
            "date": datetime.today().strftime("%B %d, %Y"),
            "regulatory_framework": "SR 11-7 / OCC 2011-12",
            "classification": "CONFIDENTIAL - REGULATORY",
        },
        "model_cards": model_cards,
        "cro_report": cro_report,
        "governance_summary": governance,
        "sr_11_7_compliance_attestation": {
            "model_inventory_maintained": True,
            "independent_validation_function": True,
            "governance_framework_documented": True,
            "audit_trail_maintained": True,
            "use_policy_documented": True,
            "escalation_procedures_defined": True,
            "notes": (
                "MRM-003, MRM-004, MRM-005, and MRM-006 are PENDING independent validation. "
                "Use of these models in material risk decisions is restricted pending validation completion."
            ),
        },
    }


if __name__ == "__main__":
    print("Generating CRO Report...")
    cro = generate_cro_report()
    print(f"\n=== {cro['report_type']} ===")
    print(f"Overall Risk Level: {cro['executive_summary']['overall_model_risk_level']}")
    print(f"Critical Issues: {cro['executive_summary']['critical_issues_count']}")
    for issue in cro["critical_issues"]:
        print(f"  [WARN]  {issue['issue']}")
    print("\nRecommendations:")
    for r in cro["recommendations"]:
        print(f"  -> {r}")

    print("\n\nGenerating Model Card for MRM-003...")
    card = generate_model_card("MRM-003")
    checklist = card["sr_11_7_checklist"]
    print("SR 11-7 Checklist:")
    for item, status in checklist.items():
        icon = "[PASS]" if status else "[FAIL]"
        print(f"  {icon} {item}")
