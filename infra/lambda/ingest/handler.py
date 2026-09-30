"""
Scheduled ingestion Lambda (production target).

No external market-data client is implemented in this repository, so this
handler only records an ingestion-run manifest in S3. A real deployment would
add a licensed data vendor client here.
"""

import json
import os
from datetime import datetime, timezone

import boto3  # provided by the Lambda Python runtime


def main(event, context):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    key = f"ingestion-runs/{now}.json"
    boto3.client("s3").put_object(
        Bucket=os.environ["MARKET_DATA_BUCKET"],
        Key=key,
        Body=json.dumps({"run_at": now, "source": "none-configured", "event": event}).encode(),
    )
    return {"status": "recorded", "key": key}
