import json

import pytest

from src.llm.guard import GovernanceBlocked
from src.mrm.approvals import write_approval
from src.mrm.model_inventory import (
    ValidationStatus, check_approval, current_version, get_inventory, require_approved,
)
from src.prompts import registry


def _approve(comp, status="APPROVED", **over):
    cur = current_version(comp)
    rec = {"status": status, "model_id": cur["model_id"], "prompt_versions": cur["prompt_versions"],
           "approved_at": "2026-09-01T00:00:00+00:00", "approved_by": "test", "evidence": "heldout-test"}
    rec.update(over)
    return {comp: rec}


def test_gate_blocks_when_not_validated():
    cur = current_version("MRM-001")
    with pytest.raises(GovernanceBlocked, match="not yet validated"):
        require_approved("MRM-001", cur["model_id"], cur["prompt_versions"], approvals={})


def test_gate_passes_for_exact_approved_version():
    cur = current_version("MRM-002")
    res = require_approved("MRM-002", cur["model_id"], cur["prompt_versions"], approvals=_approve("MRM-002"))
    assert res["allowed"] and res["status"] == "APPROVED"
    res = require_approved("MRM-002", cur["model_id"], cur["prompt_versions"],
                           approvals=_approve("MRM-002", "CONDITIONAL"))
    assert res["status"] == "CONDITIONAL"


def test_gate_blocks_failed_and_version_mismatch():
    cur = current_version("MRM-003")
    with pytest.raises(GovernanceBlocked, match="FAILED"):
        require_approved("MRM-003", cur["model_id"], cur["prompt_versions"], approvals=_approve("MRM-003", "FAILED"))
    with pytest.raises(GovernanceBlocked, match="differs from approved"):
        require_approved("MRM-003", "some-other-model", cur["prompt_versions"], approvals=_approve("MRM-003"))
    with pytest.raises(GovernanceBlocked, match="prompts"):
        require_approved("MRM-003", cur["model_id"], ["risk_report@9.9.9#deadbeef"], approvals=_approve("MRM-003"))


def test_inventory_status_derived_from_approvals_file():
    inv = get_inventory()
    assert all(m["validation_status"] == ValidationStatus.PENDING for m in inv.get_all())
    rec = _approve("MRM-001")["MRM-001"]
    write_approval("MRM-001", rec)
    inv = get_inventory()
    m = inv.get_by_id("MRM-001")
    assert m["validation_status"] == ValidationStatus.APPROVED
    assert m["approved_version"]["evidence"] == "heldout-test"
    assert m["next_validation_due"] is not None
    assert check_approval("MRM-001", rec["model_id"], rec["prompt_versions"])["allowed"]


def test_write_approval_rejects_bad_status(tmp_path):
    with pytest.raises(ValueError):
        write_approval("MRM-001", {"status": "YES"}, path=str(tmp_path / "a.json"))


def test_prompt_registry_hashing():
    p = registry.get("rag_answer")
    assert p.ref == f"rag_answer@{p.version}#{registry.sha8(p.template)}"
    assert len(p.sha8) == 8
    other = registry.RegisteredPrompt("x", "1.0.0", p.template + " ")
    assert other.sha8 != p.sha8
    with pytest.raises(ValueError):
        registry.register("x_bad", "1.0", "t")
    registry.register("tmp_prompt", "1.0.0", "a")
    with pytest.raises(ValueError):
        registry.register("tmp_prompt", "1.0.0", "b")
    assert registry.refs(["grader", "rag_answer"]) == sorted([registry.get("grader").ref, p.ref])


def test_run_all_validations_returns_dict_keyed_by_model_id():
    from src.mrm.validation.model_validator import run_all_validations

    res = run_all_validations()
    assert set(res) == {"MRM-001", "MRM-002", "MRM-003", "MRM-004", "MRM-005", "MRM-006"}
    for mid, report in res.items():
        assert report["model_id"] == mid
        assert isinstance(report["tests"], list) and report["tests"]
        json.dumps(report, default=str)
    bt = [t for t in res["MRM-006"]["tests"] if t["test_name"].startswith("VaR Backtesting")][0]
    assert bt["details"]["methods"]["historical_rolling"]["test_observations"] >= 250
    ev = [t for t in res["MRM-001"]["tests"] if t["test_name"] == "Held-out Eval Evidence"][0]
    assert not ev["passed"]


def test_reports_generate_from_real_state():
    from src.mrm.reporting.mrm_reporter import generate_cro_report, generate_full_exam_package, generate_model_card

    cro = generate_cro_report(traces=[])
    assert cro["executive_summary"]["total_models_in_inventory"] == 6
    assert any("run_eval.py" in r for r in cro["recommendations"])
    pkg = generate_full_exam_package(traces=[])
    assert pkg["sr_11_7_compliance_attestation"]["independent_validation_function"] is True
    card = generate_model_card("MRM-006")
    assert card["model_card"]["running_version"]["model_id"] == "src/utils/metrics.py"


def test_change_request_counts_as_pending():
    from src.mrm.governance.model_governance import ChangeType, ModelChangeRequest, pending_change_requests

    before = pending_change_requests()
    cr = ModelChangeRequest("MRM-001", ChangeType.MODEL_UPDATE, "swap model", "tester")
    assert pending_change_requests() == before + 1
    cr.approve("mrm-lead")
    assert pending_change_requests() == before
