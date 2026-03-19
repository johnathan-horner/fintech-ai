# FinTech AI - Model Risk Management Framework

## Overview

The FinTech AI MRM framework implements the three-pillar structure mandated by **SR 11-7 (Federal Reserve)** and **OCC 2011-12** guidance on model risk management:

| Pillar | SR 11-7 Section | Implementation |
|---|---|---|
| Model Development & Use | ?III | `model_inventory.py` - model registration, tiering, use policy |
| Model Validation | ?IV | `validation/model_validator.py` - independent test suite |
| Governance & Controls | ?V | `governance/model_governance.py` - change mgmt, audit, escalation |

---

## Model Inventory (6 Models)

| Model ID | Name | Tier | Status | Validation |
|---|---|---|---|---|
| MRM-001 | RAG Financial Q&A Chain | Tier 2 | Production | Conditional |
| MRM-002 | Market Analysis Agent | Tier 2 | Production | Conditional |
| MRM-003 | Risk Assessment Agent | **Tier 1** | Production | **PENDING** |
| MRM-004 | Portfolio Management Agent | **Tier 1** | Staging | **PENDING** |
| MRM-005 | Multi-Agent Orchestrator | **Tier 1** | Staging | **PENDING** |
| MRM-006 | Financial Metrics Engine | **Tier 1** | Production | **PENDING** |

[WARN] **CRITICAL**: Four Tier 1 models are PENDING validation. Per SR 11-7, these models must not be used in material risk limit decisions until independently validated.

---

## Validation Tests

The `model_validator.py` module runs four SR 11-7 mandated test categories:

### 1. Conceptual Soundness
- Verifies that the model design is appropriate for its stated purpose
- Checks: documentation completeness, limitation disclosure, use case alignment
- Flags Tier 1 models in production without approved validation

### 2. Data Quality
- Checks input completeness (all required fields present)
- Range validation (beta, weights, shares)
- Portfolio weight sum validation (~100%)
- Beta outlier detection (>2? from mean)

### 3. VaR Backtesting (MRM-006 only)
- Historical simulation: exceedance rate vs. expected 5% at 95% confidence
- Basel Traffic Light test: Green (<4 exceedances/250 days), Yellow (5-9), Red (>=10)
- Kupiec Proportion of Failures (POF) test: chi-squared test of exceedance rate

### 4. Sensitivity Analysis (MRM-006 only)
- Beta monotonicity: higher beta -> higher VaR and risk score
- Position removal: removing high-beta position reduces VaR
- PnL monotonicity: loss positions score higher risk than gain positions

---

## Monitoring

`monitoring/model_monitor.py` tracks:

| Metric | Warning Threshold | Critical Threshold |
|---|---|---|
| Error Rate | > 5% | > 15% |
| Avg Latency | > 10s | > 30s |
| Input Drift (PSI) | > 0.10 | > 0.25 |
| Validation Expiry | 60 days before due | Overdue |

### Input Drift (PSI)
Population Stability Index on the beta distribution:
- PSI < 0.10: Stable - no action
- PSI 0.10-0.25: Moderate drift - increase monitoring, schedule re-validation
- PSI > 0.25: Significant drift - mandatory re-validation before continued use

---

## Governance

### Use Policy
Each model has a defined use policy specifying:
- **Approved users** (by role)
- **Approved uses** (specific use cases)
- **Prohibited uses** (explicit prohibitions)
- **Conditions** (requirements for compliant use)
- **Approval required for** (use cases requiring CRO sign-off)

### Change Management
All material changes require a `ModelChangeRequest`:
- **Minor**: No re-validation required. Model owner approval.
- **Moderate**: Re-validation triggered. MRM team approval.
- **Major**: Re-validation triggered. CRO approval required.

### Escalation Procedures

| Trigger | Escalation Path |
|---|---|
| VaR breach (3+ consecutive days) | L2 -> L3 -> L4 (CRO) |
| Error rate > 15% | L2 -> L3 |
| PSI > 0.25 | L2 -> L3 |
| Validation expired (Tier 1) | L2 -> L3 -> L4 -> L5 (Board) |
| Unauthorized use | L2 -> L3 -> L4 |

### Audit Trail
All governance actions are logged immutably to `logs/mrm_governance_log.json`:
- Change request submissions and approvals
- Policy violation incidents
- Re-validation triggers

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/mrm/inventory` | Full model inventory + summary stats |
| GET | `/mrm/inventory/{model_id}` | Single model details |
| POST | `/mrm/validation/run/{model_id}` | Run validation suite |
| POST | `/mrm/validation/run-all` | Validate all models |
| GET | `/mrm/monitoring/report` | Full monitoring report |
| GET | `/mrm/monitoring/drift/{model_id}` | PSI drift analysis |
| GET | `/mrm/governance/summary` | Governance summary + Tier 1 violations |
| POST | `/mrm/governance/policy-check` | Check model use policy |
| GET | `/mrm/governance/audit-log` | Retrieve audit log |
| POST | `/mrm/governance/change-request` | Submit change request |
| GET | `/mrm/reporting/cro-report` | CRO monthly report |
| GET | `/mrm/reporting/model-card/{model_id}` | SR 11-7 model card |
| GET | `/mrm/reporting/exam-package` | Full regulatory exam package |

---

## Outstanding Validation Items

Before MRM-003, MRM-004, MRM-005, and MRM-006 can be fully approved for Tier 1 use:

1. **VaR Methodology (MRM-006)**: Independent quant validation of parametric VaR formula. Implement CVaR/Expected Shortfall alongside VaR for Basel III alignment.
2. **Risk Score Calibration (MRM-006)**: Empirically calibrate weighting factors (currently heuristic) using historical data.
3. **LLM Output Accuracy (MRM-003)**: Define and measure accuracy metrics for risk narrative outputs vs. quantitative ground truth.
4. **Backtesting Infrastructure**: Build real backtesting against historical P&L data (not synthetic) for VaR validation.
5. **Stress Testing**: Add scenario analysis (2008 GFC, COVID-19, rate shock scenarios) to MRM-003 and MRM-006.
6. **Orchestration Failure Modes (MRM-005)**: Implement circuit breaker pattern and compensation logic for agent failures.

---

*Maintained by Johnathan Horner. Reviewed annually and upon any material model change.*
*Regulatory framework: SR 11-7 (Federal Reserve, 2011) / OCC 2011-12.*
