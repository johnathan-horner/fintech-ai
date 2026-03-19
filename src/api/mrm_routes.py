"""
FinTech AI - MRM API Endpoints
Adds Model Risk Management routes to the FastAPI backend.
Mount this router in src/api/main.py.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.mrm.model_inventory import get_inventory
from src.mrm.validation.model_validator import run_validation_suite
from src.mrm.monitoring.model_monitor import run_monitoring_report, detect_input_drift
from src.mrm.governance.model_governance import (
    get_governance_summary,
    check_use_policy,
    get_audit_log,
    ModelChangeRequest,
    ChangeType,
)
from src.mrm.reporting.mrm_reporter import generate_cro_report, generate_model_card, generate_full_exam_package

router = APIRouter(prefix="/mrm", tags=["Model Risk Management"])


#  Request Models 

class PolicyCheckRequest(BaseModel):
    model_id: str
    user_role: str
    use_case: str


class ChangeRequestBody(BaseModel):
    model_id: str
    change_type: str
    description: str
    requestor: str
    materiality: str = "Minor"


#  Inventory 

@router.get("/inventory")
def get_model_inventory():
    """Return the full SR 11-7 model inventory."""
    inv = get_inventory()
    return {
        "models": inv.get_all(),
        "summary": inv.summary_stats(),
    }


@router.get("/inventory/{model_id}")
def get_model(model_id: str):
    """Return a single model from the inventory."""
    inv = get_inventory()
    model = inv.get_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")
    return model


@router.get("/inventory/tier/{tier}")
def get_models_by_tier(tier: str):
    """Return models filtered by risk tier (1, 2, or 3)."""
    from src.mrm.model_inventory import ModelTier
    tier_map = {"1": ModelTier.TIER_1, "2": ModelTier.TIER_2, "3": ModelTier.TIER_3}
    tier_enum = tier_map.get(tier)
    if not tier_enum:
        raise HTTPException(status_code=400, detail="Invalid tier. Use 1, 2, or 3.")
    inv = get_inventory()
    return inv.get_by_tier(tier_enum)


#  Validation 

@router.post("/validation/run/{model_id}")
def run_model_validation(model_id: str):
    """
    Run the full SR 11-7 validation suite for a model.
    Includes conceptual soundness, data quality, and (for MRM-006) quant tests.
    """
    inv = get_inventory()
    if not inv.get_by_id(model_id):
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")
    report = run_validation_suite(model_id)
    return report


@router.post("/validation/run-all")
def run_all_validations():
    """Run validation suite for all models in the inventory."""
    inv = get_inventory()
    results = {}
    for model_id in inv.models.keys():
        results[model_id] = run_validation_suite(model_id)
    return results


#  Monitoring 

@router.get("/monitoring/report")
def get_monitoring_report():
    """
    Get the full MRM monitoring report: performance, drift, expiry alerts.
    """
    return run_monitoring_report()


@router.get("/monitoring/drift/{model_id}")
def get_drift_report(model_id: str):
    """Get input distribution drift analysis (PSI) for a model."""
    return detect_input_drift(model_id)


@router.get("/monitoring/expiry-alerts")
def get_expiry_alerts():
    """Return validation expiry alerts for all models."""
    from src.mrm.monitoring.model_monitor import check_validation_expiry
    return check_validation_expiry()


#  Governance 

@router.get("/governance/summary")
def get_governance():
    """Return the governance summary including Tier 1 violations."""
    return get_governance_summary()


@router.post("/governance/policy-check")
def policy_check(request: PolicyCheckRequest):
    """
    Check if a proposed model use is within the approved use policy.
    Returns compliance status, reason, and applicable conditions.
    """
    return check_use_policy(request.model_id, request.user_role, request.use_case)


@router.get("/governance/audit-log")
def audit_log(model_id: Optional[str] = None):
    """Retrieve the immutable governance audit log."""
    return get_audit_log(model_id)


@router.post("/governance/change-request")
def submit_change_request(body: ChangeRequestBody):
    """Submit a model change request for review."""
    try:
        change_type = ChangeType(body.change_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid change_type. Valid values: {[e.value for e in ChangeType]}"
        )
    cr = ModelChangeRequest(
        model_id=body.model_id,
        change_type=change_type,
        description=body.description,
        requestor=body.requestor,
        materiality=body.materiality,
    )
    return {
        "change_id": cr.change_id,
        "status": cr.status,
        "requires_revalidation": cr.requires_revalidation,
        "message": (
            f"Change request {cr.change_id} submitted. "
            f"{'Re-validation will be triggered upon approval.' if cr.requires_revalidation else 'Minor change - no re-validation required.'}"
        ),
    }


#  Reporting 

@router.get("/reporting/cro-report")
def cro_report():
    """Generate the monthly CRO Model Risk Report."""
    return generate_cro_report()


@router.get("/reporting/model-card/{model_id}")
def model_card(model_id: str):
    """Generate a model card (SR 11-7 documentation) for a specific model."""
    card = generate_model_card(model_id)
    if "error" in card:
        raise HTTPException(status_code=404, detail=card["error"])
    return card


@router.get("/reporting/exam-package")
def exam_package():
    """
    Generate the full regulatory exam package.
    Includes all model cards, CRO report, and SR 11-7 compliance attestation.
    """
    return generate_full_exam_package()
