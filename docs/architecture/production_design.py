"""FinTech AI - production design from infra/fintech_stack.py (CDK, synthesized in CI, never deployed)."""
import os

from diagrams.aws.compute import Lambda
from diagrams.aws.general import User
from diagrams.aws.integration import EventbridgeRule
from diagrams.aws.management import Cloudtrail, CloudwatchLogs
from diagrams.aws.ml import Bedrock
from diagrams.aws.network import APIGateway
from diagrams.aws.security import IAMRole, KMS
from diagrams.aws.storage import SimpleStorageServiceS3Bucket as Bucket

from archstyle import arch, aws_cloud, e, layer, linkedin

os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUT = "production-design"
STORAGE = "Storage (S3: versioned, SSE-KMS, public access blocked)"

with arch("FinTech AI", "Production design (not deployed)", OUT):
    client = User("API client")

    with aws_cloud():
        with layer("API"):
            api = APIGateway("API Gateway REST API\nstage prod, 20 rps / 40 burst\nPOST /chat, /insights/*, GET /health")
        with layer("Identity"):
            role = IAMRole("Lambda execution role\nInvokeModel (Haiku 4.5 only),\nApplyGuardrail, S3 rw, KMS")
        with layer("Scheduling"):
            rule = EventbridgeRule("EventBridge rule\ncron 06,12,18 UTC Mon-Fri")
        with layer("Compute (Lambda, Python 3.11)"):
            analysis = Lambda("AIAnalysisFn\n1024 MB, 5 min")
            ingest = Lambda("MarketDataIngestFn\n512 MB, 5 min")
        with layer("Generative AI"):
            llm = Bedrock("Bedrock Claude Haiku 4.5\nus. inference profile")
            guard = Bedrock("Bedrock Guardrail\nPII block/anonymize,\ndenied topics, content filters")
        with layer(STORAGE):
            audit = Bucket("AuditTrailBucket\nGlacier at 90 days,\nexpires at 2,555 days")
            faiss = Bucket("FaissIndexBucket")
            mkt = Bucket("MarketDataBucket")
        with layer("Audit and logging"):
            trail = Cloudtrail("CloudTrail\nmulti-Region, all mgmt events,\nLambda data events, file validation")
            logs = CloudwatchLogs("CloudWatch Logs\ntrail: 7 years\nLambda: 1 month")
        with layer("Encryption"):
            kms = KMS("KMS key\nrotation on")

    client >> e("HTTPS request") >> api
    api >> e("Lambda proxy integration") >> analysis
    role >> e("execution role") >> analysis
    role >> e("execution role") >> ingest
    rule >> e("invokes on schedule") >> ingest
    analysis >> e("Converse + guardrailConfig") >> llm
    llm >> e("applies guardrail\nto input + output", constraint="false") >> guard
    ingest >> e("PutObject run manifest") >> mkt
    analysis >> e("read/write grant only\n(handler never reads it)", style="dashed") >> faiss
    ingest >> e("function logs") >> logs
    analysis >> e("function logs") >> logs
    trail >> e("delivers log files") >> audit
    trail >> e("sends events", constraint="false") >> logs
    kms >> e("SSE-KMS for all 3 buckets", lhead=f"cluster_{STORAGE}") >> mkt
    kms >> e("encrypts guardrail") >> guard

linkedin(OUT)
