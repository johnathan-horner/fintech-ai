# FinTech AI - SEC/FINRA Compliance Documentation

## Overview

FinTech AI is designed for internal use by qualified institutional investment professionals.
This document outlines compliance controls aligned with SEC and FINRA requirements.

---

## Regulatory Framework

| Regulation | Relevance | Control |
|---|---|---|
| SEC Rule 17a-4 | Books and records retention | CloudTrail + S3 Intelligent-Tiering with 7-year retention |
| FINRA Rule 4370 | Business continuity | Multi-AZ Lambda + S3 versioning |
| SEC Reg SP | Customer data privacy | KMS encryption + IAM least privilege |
| FINRA Rule 3110 | Supervisory systems | Full audit logging via CloudTrail |
| SEC AI Guidance (2024) | AI model governance | Bedrock Guardrails + disclaimer on all outputs |

---

## Data Controls

### Encryption
- All data at rest: AES-256 via AWS KMS with automatic key rotation (annual)
- All data in transit: TLS 1.2+ enforced at API Gateway level
- FAISS index: KMS-encrypted in S3

### Access Control
- IAM roles follow least-privilege principle - no wildcard `*` resource permissions
- Lambda execution roles scoped to specific Bedrock model ARNs
- API Gateway throttling: 100 RPS sustained, 200 burst

### Audit Trail
- All Bedrock API calls logged via CloudTrail
- Logs shipped to CloudWatch with 1-year retention
- S3 access logs enabled on all data buckets

---

## AI Output Governance

### Investment Disclaimer
All AI-generated outputs include a mandatory disclaimer:

> *"This AI-generated analysis is for informational purposes only and does not constitute
> investment advice. Past performance is not indicative of future results. All investment
> decisions should be made in consultation with a qualified financial advisor."*

### Blocked Topics
The Bedrock Guardrail configuration blocks queries and outputs related to:
- Insider trading
- Front running
- Market manipulation
- Pump and dump schemes

### PII Filtering
Bedrock Guardrails automatically redact:
- Social Security Numbers
- Credit/debit card numbers
- Email addresses (anonymized)
- Phone numbers (anonymized)

---

## Data Retention

| Data Type | Retention Period | Location |
|---|---|---|
| Market data (raw) | 7 years | S3 (Intelligent-Tiering after 30 days) |
| AI analysis outputs | 1 year | CloudWatch Logs |
| API access logs | 1 year | CloudWatch Logs |
| CloudTrail audit logs | 7 years | S3 (dedicated trail bucket) |
| FAISS vector index | Current only | S3 (versioned, prior versions expire after 90 days) |

---

## Development vs. Production

**Development:** All data is synthetic. Generated via `src/etl/generate_data.py` using Faker.
No real client, portfolio, or market data is used during development or testing.

**Production:** Real market data from licensed data providers (Bloomberg, Refinitiv, or Yahoo Finance API).
Client portfolio data must be stored in KMS-encrypted S3 only. No real data in version control.

---

*Maintained by Johnathan Horner. Review annually or upon regulatory updates.*
