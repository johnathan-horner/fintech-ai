"""
FinTech AI - MRM Dashboard Page (Streamlit)
Adds Model Risk Management visibility to the analyst dashboard.
Import and call render_mrm_page() from streamlit_app/app.py.
"""

import streamlit as st
import requests
import os

API_URL = os.getenv("FINTECH_API_URL", "http://localhost:8000")
DEMO_MODE = os.getenv("FINTECH_DEMO_MODE", "true").lower() != "false"

def get_mock_mrm_data(endpoint: str, method: str = "GET", body: dict = None) -> dict:
    """Mock data for MRM dashboard in demo mode"""
    if "/mrm/inventory" in endpoint:
        return {
            "summary": {
                "total_models": 5,
                "tier_1": 2,
                "approved": 3,
                "conditional": 1,
                "pending": 1,
                "in_production": 4
            },
            "models": [
                {
                    "model_id": "MRM-001-VAR",
                    "name": "Portfolio VaR Model",
                    "type": "Risk Model",
                    "owner": "Risk Team",
                    "tier": "Tier 1 - High",
                    "validation_status": "Approved",
                    "status": "Production",
                    "business_use": "Daily VaR calculation for portfolio risk management",
                    "next_validation_due": "2025-06-01",
                    "dependencies": ["Market Data API", "Portfolio Positions"],
                    "known_limitations": ["Does not account for liquidity risk", "Limited to normal market conditions"],
                    "mitigations": ["Manual liquidity adjustment", "Stress testing overlay"],
                    "materiality_rationale": "Critical for daily risk reporting and regulatory compliance"
                },
                {
                    "model_id": "MRM-002-CREDIT",
                    "name": "Credit Risk Scoring Model",
                    "type": "Credit Model",
                    "owner": "Credit Team",
                    "tier": "Tier 1 - High",
                    "validation_status": "Conditional Approval",
                    "status": "Production",
                    "business_use": "Corporate bond credit risk assessment",
                    "next_validation_due": "2025-04-15",
                    "dependencies": ["Credit Rating Data", "Financial Statements"],
                    "known_limitations": ["Limited emerging market coverage"],
                    "mitigations": ["Manual review for EM bonds"],
                    "materiality_rationale": "Used for material credit exposure decisions"
                }
            ]
        }
    elif "/mrm/validation/run" in endpoint:
        return {
            "overall_score": 0.82,
            "recommended_status": "Approved",
            "tests": [
                {"test_name": "Conceptual Soundness", "passed": True, "score": 0.85, "findings": ["PASS: Model methodology is sound"], "recommendations": []},
                {"test_name": "Data Quality", "passed": True, "score": 0.90, "findings": ["PASS: Data completeness >95%"], "recommendations": []},
                {"test_name": "VaR Backtesting", "passed": True, "score": 0.78, "findings": ["PASS: 4 exceptions in 250 days"], "recommendations": ["Monitor for trend"]}
            ]
        }
    elif "/mrm/monitoring/report" in endpoint:
        return {
            "overall_status": "WARNING",
            "performance": [
                {
                    "model_id": "MRM-001-VAR",
                    "status": "HEALTHY",
                    "metrics": {"total_calls_30d": 30, "error_rate": 0.02, "avg_latency_ms": 150, "p95_latency_ms": 300},
                    "alerts": []
                }
            ],
            "drift": {
                "psi_value": 0.15,
                "interpretation": "Moderate drift detected",
                "alerts": [{"severity": "WARNING", "message": "PSI above threshold"}]
            },
            "validation_expiry": [
                {"model_id": "MRM-002-CREDIT", "severity": "WARNING", "message": "Validation expires in 30 days"}
            ]
        }
    elif "/mrm/governance/policy-check" in endpoint:
        return {"compliant": True, "reason": "Model approved for internal use by analysts", "conditions": ["Must include disclaimer"]}
    elif "/mrm/governance/change-request" in endpoint:
        return {"change_id": "CR-2025-001", "message": "Change request submitted successfully"}
    elif "/mrm/governance/audit-log" in endpoint:
        return [
            {"timestamp": "2025-03-19T10:30:00", "action": "Model Validation", "model_id": "MRM-001-VAR", "actor": "system"},
            {"timestamp": "2025-03-18T14:15:00", "action": "Policy Check", "model_id": "MRM-002-CREDIT", "actor": "analyst_1"}
        ]
    elif "/mrm/reporting/cro-report" in endpoint:
        return {
            "executive_summary": {
                "overall_model_risk_level": "MEDIUM",
                "critical_issues_count": 1,
                "models_pending_validation": 1
            },
            "critical_issues": [{"issue": "MRM-002-CREDIT validation expires soon"}],
            "recommendations": ["Prioritize credit model validation", "Review data quality procedures"]
        }
    elif "/mrm/reporting/model-card" in endpoint:
        return {
            "model_id": "MRM-001-VAR",
            "sr_11_7_checklist": {
                "conceptual_soundness": True,
                "ongoing_monitoring": True,
                "developmental_evidence": True,
                "outcome_analysis": False
            }
        }
    elif "/mrm/reporting/exam-package" in endpoint:
        return {
            "sr_11_7_compliance_attestation": {
                "model_inventory_complete": True,
                "validation_policy_current": True,
                "independent_validation": True,
                "ongoing_monitoring": False,
                "notes": "Monitoring framework needs enhancement"
            }
        }
    return {"error": "Mock data not implemented for this endpoint"}

def call_mrm(endpoint: str, method: str = "GET", body: dict = None) -> dict:
    if DEMO_MODE:
        return get_mock_mrm_data(endpoint, method, body)

    try:
        if method == "POST":
            r = requests.post(f"{API_URL}{endpoint}", json=body, timeout=120)
        else:
            r = requests.get(f"{API_URL}{endpoint}", timeout=30)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def tier_badge(tier: str) -> str:
    colors = {"Tier 1 - High": "[HIGH]", "Tier 2 - Medium": "[MED]", "Tier 3 - Low": "[LOW]"}
    return colors.get(tier, "?")


def status_badge(status: str) -> str:
    badges = {
        "Approved": "[PASS]",
        "Conditional Approval": "[WARN]",
        "Pending Validation": "[PENDING]",
        "Validation Failed": "[FAIL]",
        "Validation Expired": "[CRITICAL]",
        "Under Review": "Search",
    }
    return badges.get(status, "?")


def render_mrm_page():
    st.title("MRM Model Risk Management (MRM)")
    st.markdown(
        "SR 11-7 / OCC 2011-12 compliant model governance - "
        "inventory, validation, monitoring, and reporting."
    )

    mrm_tab, validate_tab, monitor_tab, govern_tab, report_tab = st.tabs([
        " Model Inventory",
        "[VALIDATE] Validation",
        "[MONITOR] Monitoring",
        " Governance",
        " Reports",
    ])

    # - Tab 1: Model Inventory 
    with mrm_tab:
        st.subheader(" Model Inventory")
        data = call_mrm("/mrm/inventory")

        if "error" in data:
            st.error(f"Could not load inventory: {data['error']}")
        else:
            stats = data.get("summary", {})

            # Summary metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Models", stats.get("total_models", 0))
            c2.metric("Tier 1 (High)", stats.get("tier_1", 0))
            c3.metric("Approved", stats.get("approved", 0) + stats.get("conditional", 0))
            c4.metric("Pending Validation", stats.get("pending", 0))
            c5.metric("In Production", stats.get("in_production", 0))

            if stats.get("pending", 0) > 0:
                st.warning(
                    f"[WARN] {stats['pending']} model(s) with PENDING validation. "
                    "Tier 1 models must not be used in material risk decisions until validated."
                )

            st.markdown("---")

            # Model cards
            for model in data.get("models", []):
                tier = model.get("tier", "")
                v_status = model.get("validation_status", "")
                dev_status = model.get("status", "")

                with st.expander(
                    f"{tier_badge(tier)} **{model['model_id']}** - {model['name']} "
                    f"| {status_badge(v_status)} {v_status} | {dev_status}"
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Type:** {model.get('type', 'N/A')}")
                        st.markdown(f"**Owner:** {model.get('owner', 'N/A')}")
                        st.markdown(f"**Risk Tier:** {tier}")
                        st.markdown(f"**Validation Status:** {v_status}")
                        st.markdown(f"**Next Validation Due:** {model.get('next_validation_due', 'Not set')}")
                    with col2:
                        st.markdown(f"**Business Use:** {model.get('business_use', 'N/A')}")
                        st.markdown(f"**Dev Status:** {dev_status}")
                        st.markdown(f"**Dependencies:** {', '.join(model.get('dependencies', []))}")

                    st.markdown("**Known Limitations:**")
                    for lim in model.get("known_limitations", []):
                        st.markdown(f"  ? {lim}")

                    st.markdown("**Mitigations:**")
                    for mit in model.get("mitigations", []):
                        st.markdown(f"  ? {mit}")

                    st.info(f"**Materiality Rationale:** {model.get('materiality_rationale', '')}")

    # - Tab 2: Validation 
    with validate_tab:
        st.subheader("[VALIDATE] Independent Model Validation")
        st.markdown(
            "Run SR 11-7 validation tests: conceptual soundness, data quality, "
            "VaR backtesting (MRM-006), and sensitivity analysis."
        )

        inv_data = call_mrm("/mrm/inventory")
        model_ids = [m["model_id"] for m in inv_data.get("models", [])]

        col1, col2 = st.columns([2, 1])
        with col1:
            selected_model = st.selectbox("Select Model to Validate", model_ids)
        with col2:
            run_all = st.checkbox("Run All Models")

        if st.button("> Run Validation Suite"):
            endpoint = "/mrm/validation/run-all" if run_all else f"/mrm/validation/run/{selected_model}"
            method = "POST"

            with st.spinner("Running SR 11-7 validation tests..."):
                result = call_mrm(endpoint, method=method)

            if run_all:
                for mid, report in result.items():
                    score = report.get("overall_score", 0)
                    status = report.get("recommended_status", "")
                    color = "[LOW]" if score >= 0.85 else "[MED]" if score >= 0.65 else "[HIGH]"
                    with st.expander(f"{color} **{mid}** - Score: {score:.1%} | {status}"):
                        for test in report.get("tests", []):
                            icon = "[PASS]" if test["passed"] else "[FAIL]"
                            st.markdown(f"**{icon} {test['test_name']}** - Score: {test['score']:.1%}")
                            for finding in test.get("findings", []):
                                st.markdown(f"  -> {finding}")
            else:
                score = result.get("overall_score", 0)
                status = result.get("recommended_status", "")
                color = "[LOW]" if score >= 0.85 else "[MED]" if score >= 0.65 else "[HIGH]"

                st.metric("Overall Validation Score", f"{score:.1%}")
                st.metric("Recommended Status", status)

                for test in result.get("tests", []):
                    icon = "[PASS]" if test["passed"] else "[FAIL]"
                    with st.expander(f"{icon} {test['test_name']} - {test['score']:.1%}"):
                        for finding in test.get("findings", []):
                            if finding.startswith("FAIL") or finding.startswith("CRITICAL"):
                                st.error(finding)
                            elif finding.startswith("WARN"):
                                st.warning(finding)
                            else:
                                st.success(finding)
                        for rec in test.get("recommendations", []):
                            st.info(f" {rec}")

    # - Tab 3: Monitoring 
    with monitor_tab:
        st.subheader("[MONITOR] Ongoing Model Monitoring")
        st.markdown("30-day performance metrics, input drift (PSI), and validation expiry alerts.")

        if st.button("[REFRESH] Refresh Monitoring Report"):
            with st.spinner("Loading monitoring data..."):
                mon = call_mrm("/mrm/monitoring/report")

            # Overall status
            overall = mon.get("overall_status", "UNKNOWN")
            if overall == "CRITICAL":
                st.error(f"[HIGH] Overall Monitoring Status: {overall}")
            elif overall == "WARNING":
                st.warning(f"[MED] Overall Monitoring Status: {overall}")
            else:
                st.success(f"[LOW] Overall Monitoring Status: {overall}")

            # Performance
            st.subheader("Performance - Last 30 Days")
            for perf in mon.get("performance", []):
                status = perf.get("status", "HEALTHY")
                icon = "[HIGH]" if status == "CRITICAL" else "[MED]" if status == "WARNING" else "[LOW]"
                with st.expander(f"{icon} **{perf['model_id']}** - {status}"):
                    metrics = perf.get("metrics", {})
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Calls (30d)", metrics.get("total_calls_30d", 0))
                    c2.metric("Error Rate", f"{metrics.get('error_rate', 0):.1%}")
                    c3.metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0):.0f}ms")
                    c4.metric("P95 Latency", f"{metrics.get('p95_latency_ms', 0):.0f}ms")
                    for alert in perf.get("alerts", []):
                        st.warning(f"[WARN] {alert['message']}")

            # Drift
            drift = mon.get("drift", {})
            st.subheader("Input Drift (PSI)")
            psi = drift.get("psi_value", 0)
            psi_color = "[HIGH]" if psi > 0.25 else "[MED]" if psi > 0.10 else "[LOW]"
            st.metric(f"{psi_color} PSI (Beta Distribution)", f"{psi:.4f}")
            st.caption(drift.get("interpretation", ""))
            for alert in drift.get("alerts", []):
                if alert["severity"] == "CRITICAL":
                    st.error(f"[HIGH] {alert['message']}")
                elif alert["severity"] == "WARNING":
                    st.warning(f"[MED] {alert['message']}")
                else:
                    st.info(f"- {alert['message']}")

            # Expiry alerts
            expiry = mon.get("validation_expiry", [])
            if expiry:
                st.subheader("Validation Expiry Alerts")
                for a in expiry:
                    if a["severity"] == "CRITICAL":
                        st.error(f"[CRITICAL] **{a['model_id']}** - {a['message']}")
                    else:
                        st.warning(f"[ALERT] **{a['model_id']}** - {a['message']}")

    # - Tab 4: Governance 
    with govern_tab:
        st.subheader(" Governance & Policy")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Policy Check")
            policy_model = st.selectbox("Model", model_ids, key="policy_model")
            user_role = st.selectbox("User Role", ["analyst", "portfolio_manager", "risk_manager", "admin"])
            use_case = st.text_input("Proposed Use Case", "internal market trend analysis")
            if st.button("Search Check Policy"):
                result = call_mrm(
                    "/mrm/governance/policy-check",
                    method="POST",
                    body={"model_id": policy_model, "user_role": user_role, "use_case": use_case},
                )
                if result.get("compliant"):
                    st.success(f"[PASS] COMPLIANT: {result['reason']}")
                    for cond in result.get("conditions", []):
                        st.info(f"Condition: {cond}")
                else:
                    st.error(f"[FAIL] NON-COMPLIANT: {result.get('reason', 'Policy violation.')}")

        with col2:
            st.markdown("### Submit Change Request")
            cr_model = st.selectbox("Model", model_ids, key="cr_model")
            cr_type = st.selectbox("Change Type", [
                "Model Update", "Parameter Change", "Data Source Change",
                "Infrastructure Change", "New Use Case", "Dependency Change"
            ])
            cr_desc = st.text_area("Description", "Describe the change...")
            cr_materiality = st.select_slider("Materiality", ["Minor", "Moderate", "Major"])
            cr_requestor = st.text_input("Requestor", "analyst_1")
            if st.button("? Submit Change Request"):
                result = call_mrm(
                    "/mrm/governance/change-request",
                    method="POST",
                    body={
                        "model_id": cr_model,
                        "change_type": cr_type,
                        "description": cr_desc,
                        "requestor": cr_requestor,
                        "materiality": cr_materiality,
                    },
                )
                if "change_id" in result:
                    st.success(f"[PASS] {result['message']}")
                    st.code(f"Change ID: {result['change_id']}")
                else:
                    st.error(f"Failed: {result}")

        st.markdown("---")
        st.markdown("### Audit Log")
        audit_model_filter = st.selectbox("Filter by Model (optional)", ["All"] + model_ids)
        if st.button("? Load Audit Log"):
            mid = None if audit_model_filter == "All" else audit_model_filter
            endpoint = f"/mrm/governance/audit-log{'?model_id=' + mid if mid else ''}"
            log = call_mrm(endpoint)
            if isinstance(log, list) and log:
                for entry in reversed(log[-20:]):
                    st.markdown(
                        f"`{entry['timestamp'][:19]}` - **{entry['action']}** | "
                        f"Model: {entry['model_id']} | Actor: {entry['actor']}"
                    )
            elif isinstance(log, list):
                st.info("No audit log entries found.")
            else:
                st.error(str(log))

    # - Tab 5: Reports 
    with report_tab:
        st.subheader(" MRM Reports")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(" CRO Monthly Report"):
                with st.spinner("Generating CRO report..."):
                    report = call_mrm("/mrm/reporting/cro-report")
                ex_sum = report.get("executive_summary", {})
                level = ex_sum.get("overall_model_risk_level", "UNKNOWN")
                color = "[HIGH]" if level == "HIGH" else "[MED]" if level == "MEDIUM" else "[LOW]"
                st.metric(f"{color} Overall Model Risk", level)
                st.metric("Critical Issues", ex_sum.get("critical_issues_count", 0))
                st.metric("Pending Validations", ex_sum.get("models_pending_validation", 0))
                for issue in report.get("critical_issues", []):
                    st.error(f"[WARN] {issue['issue']}")
                st.markdown("**Recommendations:**")
                for rec in report.get("recommendations", []):
                    st.markdown(f"-> {rec}")

        with col2:
            if st.button(" Generate Model Card"):
                mc_model = st.session_state.get("mc_model_select", model_ids[0])
                with st.spinner(f"Generating model card for {mc_model}..."):
                    card = call_mrm(f"/mrm/reporting/model-card/{mc_model}")
                st.json(card)

        with col3:
            if st.button(" Full Exam Package"):
                with st.spinner("Generating regulatory exam package..."):
                    pkg = call_mrm("/mrm/reporting/exam-package")
                attestation = pkg.get("sr_11_7_compliance_attestation", {})
                st.markdown("**SR 11-7 Compliance Attestation:**")
                for k, v in attestation.items():
                    if isinstance(v, bool):
                        icon = "[PASS]" if v else "[FAIL]"
                        st.markdown(f"{icon} {k.replace('_', ' ').title()}")
                if attestation.get("notes"):
                    st.warning(attestation["notes"])

        # Model card selector (shared)
        st.markdown("---")
        mc_model_select = st.selectbox("Select model for card", model_ids, key="mc_model_select")
        if st.button(" View Model Card"):
            card = call_mrm(f"/mrm/reporting/model-card/{mc_model_select}")
            checklist = card.get("sr_11_7_checklist", {})
            st.markdown("**SR 11-7 Checklist:**")
            cols = st.columns(3)
            for i, (k, v) in enumerate(checklist.items()):
                icon = "[PASS]" if v else "[FAIL]"
                cols[i % 3].markdown(f"{icon} {k.replace('_', ' ').title()}")
            with st.expander("Full Model Card JSON"):
                st.json(card)
