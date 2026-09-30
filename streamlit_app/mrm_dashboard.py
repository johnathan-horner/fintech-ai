"""
FinTech AI - MRM Dashboard Page (Streamlit)

Calls the real src/mrm modules directly (no HTTP, no mock data):
inventory with approval versions, validation suite (incl. the out-of-sample
VaR backtest), monitoring (PSI on real data + this session's trace log),
governance checks and generated reports.
"""

import pandas as pd
import streamlit as st

from src.mrm.governance.model_governance import (
    ChangeType, ModelChangeRequest, check_use_policy, get_audit_log, get_governance_summary,
)
from src.mrm.model_inventory import ValidationStatus, get_inventory
from src.mrm.monitoring.model_monitor import run_monitoring_report
from src.mrm.reporting.mrm_reporter import generate_cro_report, generate_full_exam_package, generate_model_card
from src.mrm.validation.model_validator import run_all_validations, run_validation_suite
from streamlit_app.formatting import md_escape, md_money

STATUS_BADGE = {
    ValidationStatus.APPROVED: "[APPROVED]",
    ValidationStatus.CONDITIONAL: "[CONDITIONAL]",
    ValidationStatus.PENDING: "[NOT YET VALIDATED]",
    ValidationStatus.FAILED: "[FAILED]",
    ValidationStatus.EXPIRED: "[EXPIRED]",
    ValidationStatus.IN_REVIEW: "[IN REVIEW]",
}


def _session_traces() -> list:
    return list(st.session_state.get("trace_log", []))


def _render_findings(test: dict):
    for finding in test.get("findings", []):
        f = md_money(finding)
        if finding.startswith(("FAIL", "CRITICAL")):
            st.error(f)
        elif finding.startswith("WARN"):
            st.warning(f)
        elif finding.startswith("PASS"):
            st.success(f)
        else:
            st.caption(f)
    for rec in test.get("recommendations", []):
        st.info(md_money(rec))
    methods = (test.get("details") or {}).get("methods")
    if methods:
        st.dataframe(pd.DataFrame([
            {"Method": k, "Test days": v["test_observations"], "Exceptions": v["exceptions"],
             "Last 250": v["exceptions_last_250"], "Zone": v["basel_zone"], "Kupiec LR": v["kupiec_lr"],
             "p-value": v["kupiec_p_value"]}
            for k, v in methods.items()
        ]), width="stretch", hide_index=True)


def _render_report(report: dict, expanded: bool = False):
    score = report.get("overall_score", 0)
    with st.expander(f"{report['model_id']} - score {score:.1%} - recommended: {report['recommended_status']}",
                     expanded=expanded):
        for test in report.get("tests", []):
            st.markdown(f"**{'[PASS]' if test['passed'] else '[FAIL]'} {test['test_name']}** - {test['score']:.1%}")
            _render_findings(test)


def render_mrm_page():
    st.title("Model Risk Management (MRM)")
    st.markdown("SR 11-7 / OCC 2011-12 model governance - inventory, validation, monitoring, and reporting. "
                "All values below are computed live from src/mrm.")

    inv = get_inventory()
    models = inv.get_all()
    model_ids = [m["model_id"] for m in models]

    inv_tab, validate_tab, monitor_tab, govern_tab, report_tab = st.tabs(
        ["Model Inventory", "Validation", "Monitoring", "Governance", "Reports"]
    )

    # -- Tab 1: Model Inventory
    with inv_tab:
        stats = inv.summary_stats()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Models", stats["total_models"])
        c2.metric("Tier 1 (High)", stats["tier_1"])
        c3.metric("Approved / Conditional", stats["approved"] + stats["conditional"])
        c4.metric("Not Yet Validated", stats["pending"])
        c5.metric("In Production", stats["in_production"])
        if stats["pending"]:
            st.warning(f"{stats['pending']} model(s) not yet validated. Live AI calls for unvalidated LLM components "
                       "are blocked by require_approved(); run scripts/run_eval.py --split heldout to earn approval.")
        st.markdown("---")
        for m in models:
            v_status = m["validation_status"]
            badge = "[RE-VALIDATION REQUIRED]" if m.get("version_mismatch") else STATUS_BADGE.get(v_status, "")
            with st.expander(f"{m['model_id']} - {m['name']} | {badge} {v_status.value} | "
                             f"{m['status'].value} | {m['tier'].value}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Type:** {m['type'].value}")
                    st.markdown(f"**Owner:** {m['owner']}")
                    st.markdown(f"**Validation status:** {v_status.value}")
                    st.markdown(f"**Last validated:** {m.get('validation_date') or 'Never'}")
                    st.markdown(f"**Next validation due:** {m.get('next_validation_due') or 'Not scheduled'}")
                with col2:
                    st.markdown(f"**Business use:** {m['business_use']}")
                    st.markdown(f"**Dependencies:** {', '.join(m['dependencies'])}")
                st.markdown("**Running version:**")
                st.json(m["current_version"])
                st.markdown("**Approved version:**")
                if m.get("version_mismatch"):
                    st.warning("The running model/prompt version differs from the approved version - live use is "
                               "blocked until scripts/run_eval.py re-validates it.")
                if m.get("approved_version"):
                    st.json(m["approved_version"])
                else:
                    st.caption("None - no approval record in data/mrm/approvals.json (written only by the eval harness).")
                st.markdown("**Known limitations:**")
                for lim in m.get("known_limitations", []):
                    st.markdown(md_money(f"- {lim}"))
                st.markdown("**Mitigations:**")
                for mit in m.get("mitigations", []):
                    st.markdown(md_money(f"- {mit}"))
                st.info(md_money(f"**Materiality rationale:** {m.get('materiality_rationale', '')}"))

    # -- Tab 2: Validation
    with validate_tab:
        st.subheader("Independent Model Validation")
        st.markdown("Conceptual soundness, data quality, out-of-sample 99% VaR backtest with Kupiec POF and Basel "
                    "traffic light (MRM-006), sensitivity analysis, and held-out eval evidence (LLM components).")
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_model = st.selectbox("Select Model to Validate", model_ids, key="val_model")
        with col2:
            run_all = st.checkbox("Run All Models", key="val_run_all")
        if st.button("Run Validation Suite", key="val_run"):
            with st.spinner("Running SR 11-7 validation tests..."):
                if run_all:
                    results = run_all_validations()
                    st.session_state["validation_results"] = results
                else:
                    st.session_state["validation_results"] = {selected_model: run_validation_suite(selected_model)}
        results = st.session_state.get("validation_results")
        if results:
            if len(results) > 1:
                st.dataframe(pd.DataFrame([
                    {"Model": mid, "Score": f"{r['overall_score']:.1%}", "All tests passed": r["all_tests_passed"],
                     "Recommended status": r["recommended_status"]}
                    for mid, r in results.items()
                ]), width="stretch", hide_index=True)
            for mid, report in results.items():
                _render_report(report, expanded=len(results) == 1)

    # -- Tab 3: Monitoring
    with monitor_tab:
        st.subheader("Ongoing Model Monitoring")
        st.caption("Performance comes from this session's trace log; MRM-006 drift is PSI of portfolio daily returns "
                   "(first 250-day window vs last 60 days of the synthetic price history); LLM drift is PSI of grader "
                   "scores vs the held-out eval run.")
        st.button("Refresh", key="mon_refresh")
        mon = run_monitoring_report(traces=_session_traces())
        overall = mon["overall_status"]
        (st.error if overall == "CRITICAL" else st.warning if overall == "WARNING" else st.success)(
            f"Overall monitoring status: {overall} ({mon['summary']['traced_requests']} traced requests this session)")

        st.markdown("**Input / output drift (PSI)**")
        rows = []
        for mid, d in mon["drift"].items():
            rows.append({"Model": mid, "Metric": d.get("drift_metric"), "Status": d.get("status"),
                         "PSI": d.get("psi_value") if d.get("psi_value") is not None else d.get("message", ""),
                         "Reference n": d.get("reference_n"), "Current n": d.get("current_n")})
        st.dataframe(pd.DataFrame(rows).astype(str), width="stretch", hide_index=True)
        d6 = mon["drift"].get("MRM-006", {})
        if d6.get("status") == "OK":
            st.caption(f"MRM-006 reference {d6['reference_period']}; current {d6['current_period']}.")
            for a in d6.get("alerts", []):
                (st.error if a["severity"] == "CRITICAL" else st.warning if a["severity"] == "WARNING" else st.info)(
                    a["message"])

        st.markdown("**Performance (this session)**")
        st.dataframe(pd.DataFrame([
            {"Model": p["model_id"], "Status": p["status"], **{k: v for k, v in p["metrics"].items() if k != "modes"}}
            for p in mon["performance"]
        ]).astype(str), width="stretch", hide_index=True)
        for p in mon["performance"]:
            for a in p["alerts"]:
                st.warning(f"{p['model_id']}: {a['message']}")

        if mon["validation_expiry"]:
            st.markdown("**Validation expiry alerts**")
            for a in mon["validation_expiry"]:
                (st.error if a["severity"] == "CRITICAL" else st.warning)(f"{a['model_id']} - {a['message']}")

    # -- Tab 4: Governance
    with govern_tab:
        st.subheader("Governance & Policy")
        gov = get_governance_summary()
        g1, g2, g3 = st.columns(3)
        g1.metric("Tier 1 violations", len(gov["tier1_violations"]))
        g2.metric("Pending change requests", gov["pending_change_requests"])
        g3.metric("Audit log entries", gov["audit_log_entries"])
        for v in gov["tier1_violations"]:
            st.error(f"{v['model_id']}: {v['issue']}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Policy Check")
            policy_model = st.selectbox("Model", model_ids, key="policy_model")
            user_role = st.selectbox("User Role", ["analyst", "portfolio_manager", "risk_manager", "admin"])
            use_case = st.text_input("Proposed Use Case", "internal market trend analysis")
            if st.button("Check Policy"):
                result = check_use_policy(policy_model, user_role, use_case)
                if result.get("compliant"):
                    st.success(f"COMPLIANT: {result['reason']}")
                    for cond in result.get("conditions", []):
                        st.info(f"Condition: {cond}")
                else:
                    st.error(f"NON-COMPLIANT: {md_escape(result.get('reason', 'Policy violation.'))}")
        with col2:
            st.markdown("### Submit Change Request")
            cr_model = st.selectbox("Model", model_ids, key="cr_model")
            cr_type = st.selectbox("Change Type", [c.value for c in ChangeType])
            cr_desc = st.text_area("Description", "Describe the change...")
            cr_materiality = st.select_slider("Materiality", ["Minor", "Moderate", "Major"])
            cr_requestor = st.text_input("Requestor", "analyst_1")
            if st.button("Submit Change Request"):
                cr = ModelChangeRequest(cr_model, ChangeType(cr_type), cr_desc, cr_requestor, cr_materiality)
                st.success(f"Change request {cr.change_id} submitted "
                           f"({'re-validation required on approval' if cr.requires_revalidation else 'minor change'}).")

        st.markdown("---")
        st.markdown("### Audit Log")
        audit_filter = st.selectbox("Filter by Model (optional)", ["All"] + model_ids)
        log = get_audit_log(None if audit_filter == "All" else audit_filter)
        if log:
            for entry in reversed(log[-20:]):
                st.markdown(f"`{entry['timestamp'][:19]}` - **{entry['action']}** | Model: {entry['model_id']} | "
                            f"Actor: {md_escape(entry['actor'])}")
        else:
            st.info("No audit log entries yet (policy violations and change requests are logged here).")

    # -- Tab 5: Reports
    with report_tab:
        st.subheader("MRM Reports")
        mc_model = st.selectbox("Model for model card", model_ids, key="mc_model_select")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("CRO Monthly Report"):
                report = generate_cro_report(traces=_session_traces())
                ex = report["executive_summary"]
                st.metric("Overall Model Risk", ex["overall_model_risk_level"])
                st.metric("Critical Issues", ex["critical_issues_count"])
                st.metric("Pending Validations", ex["models_pending_validation"])
                for issue in report["critical_issues"]:
                    st.error(issue["issue"])
                st.markdown("**Recommendations:**")
                for rec in report["recommendations"]:
                    st.markdown(md_money(f"- {rec}"))
        with col2:
            if st.button("Generate Model Card"):
                card = generate_model_card(mc_model)
                st.markdown("**SR 11-7 Checklist:**")
                for k, v in card["sr_11_7_checklist"].items():
                    st.markdown(f"{'[PASS]' if v else '[FAIL]'} {k.replace('_', ' ').title()}")
                with st.expander("Full Model Card JSON"):
                    st.json(card)
        with col3:
            if st.button("Full Exam Package"):
                pkg = generate_full_exam_package(traces=_session_traces())
                att = pkg["sr_11_7_compliance_attestation"]
                st.markdown("**SR 11-7 Compliance Attestation:**")
                for k, v in att.items():
                    if isinstance(v, bool):
                        st.markdown(f"{'[PASS]' if v else '[FAIL]'} {k.replace('_', ' ').title()}")
                st.warning(att["notes"])
                with st.expander("Full exam package JSON"):
                    st.json(pkg, expanded=False)
