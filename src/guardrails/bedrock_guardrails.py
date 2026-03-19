"""
FinTech AI - Compliance Guardrails
Bedrock Guardrails for SEC/FINRA compliance.
Filters sensitive client PII and blocks non-compliant outputs.
Mirrors EduAI's bedrock_guardrails.py.
"""

import boto3
import json
import re

REGION = "us-east-1"

# Guardrail config - create once via CDK/Terraform in production
GUARDRAIL_CONFIG = {
    "name": "fintech-ai-compliance",
    "description": "SEC/FINRA compliant guardrails for hedge fund AI",
    "blocked_patterns": [
        r"\b\d{3}-\d{2}-\d{4}\b",          # SSN
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
        r"\b[A-Z]{2}\d{6}\b",               # Passport numbers
    ],
    "blocked_topics": [
        "insider trading",
        "front running",
        "market manipulation",
        "pump and dump",
    ],
    "investment_disclaimer": (
        "\n\n[WARN] DISCLAIMER: This AI-generated analysis is for informational purposes only "
        "and does not constitute investment advice. Past performance is not indicative of "
        "future results. All investment decisions should be made in consultation with a "
        "qualified financial advisor. This system is for internal use only."
    )
}


def filter_pii(text: str) -> str:
    """Remove PII patterns from AI outputs."""
    filtered = text
    for pattern in GUARDRAIL_CONFIG["blocked_patterns"]:
        filtered = re.sub(pattern, "[REDACTED]", filtered)
    return filtered


def check_compliance(query: str) -> dict:
    """
    Check if a query contains blocked topics or compliance violations.
    Returns {compliant: bool, reason: str}
    """
    query_lower = query.lower()
    for topic in GUARDRAIL_CONFIG["blocked_topics"]:
        if topic in query_lower:
            return {
                "compliant": False,
                "reason": f"Query references blocked topic: '{topic}'. "
                          f"This system cannot assist with potentially illegal trading activities.",
            }
    return {"compliant": True, "reason": ""}


def apply_guardrails(response: str, add_disclaimer: bool = True) -> str:
    """
    Apply full compliance pipeline to an AI response:
    1. PII filtering
    2. Add investment disclaimer
    """
    cleaned = filter_pii(response)
    if add_disclaimer:
        cleaned += GUARDRAIL_CONFIG["investment_disclaimer"]
    return cleaned


def create_bedrock_guardrail():
    """
    Create the Bedrock Guardrail in AWS (run once during setup).
    Returns the guardrail_id.
    """
    client = boto3.client("bedrock", region_name=REGION)
    try:
        response = client.create_guardrail(
            name=GUARDRAIL_CONFIG["name"],
            description=GUARDRAIL_CONFIG["description"],
            sensitiveInformationPolicyConfig={
                "piiEntitiesConfig": [
                    {"type": "SSN", "action": "BLOCK"},
                    {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
                    {"type": "EMAIL", "action": "ANONYMIZE"},
                    {"type": "PHONE", "action": "ANONYMIZE"},
                ]
            },
            contentPolicyConfig={
                "filtersConfig": [
                    {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                    {"type": "VIOLENCE", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
                ]
            },
        )
        guardrail_id = response["guardrailId"]
        print(f"Bedrock guardrail created: {guardrail_id}")
        return guardrail_id
    except Exception as e:
        print(f"Guardrail creation failed (may already exist): {e}")
        return None
