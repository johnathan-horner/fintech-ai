"""
Analysis Lambda (production target): Bedrock Converse with the compliance
guardrail. Mirrors the Gemini demo path in src/ at the API level.
"""

import json
import os

import boto3  # provided by the Lambda Python runtime

_bedrock = boto3.client("bedrock-runtime")


def _response(status, body):
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}


def main(event, context):
    path = (event or {}).get("path", "")
    if path.endswith("/health"):
        return _response(200, {"status": "healthy", "model": os.environ["BEDROCK_CHAT_MODEL"]})
    try:
        body = json.loads((event or {}).get("body") or "{}")
    except ValueError:
        return _response(400, {"error": "invalid JSON"})
    question = body.get("question") or f"Summarise the {path.rsplit('/', 1)[-1]} view of the portfolio."
    resp = _bedrock.converse(
        modelId=os.environ["BEDROCK_CHAT_MODEL"],
        messages=[{"role": "user", "content": [{"text": question}]}],
        inferenceConfig={"maxTokens": 800, "temperature": 0.1},
        guardrailConfig={
            "guardrailIdentifier": os.environ["GUARDRAIL_ID"],
            "guardrailVersion": os.environ["GUARDRAIL_VERSION"],
        },
    )
    text = "".join(p.get("text", "") for p in resp["output"]["message"]["content"])
    return _response(200, {"answer": text, "stop_reason": resp.get("stopReason")})
