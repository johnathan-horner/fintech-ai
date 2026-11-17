"""FinTech AI: production design from infra/fintech_stack.py (AWS CDK). Not deployed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagrams import Diagram, Edge
from diagrams.aws.compute import Lambda
from diagrams.aws.integration import Eventbridge
from diagrams.aws.management import Cloudtrail, CloudwatchLogs
from diagrams.aws.ml import Bedrock
from diagrams.aws.network import APIGateway
from diagrams.aws.security import IAMRole, KMS
from diagrams.aws.storage import SimpleStorageServiceS3Bucket as S3
from diagrams.onprem.client import Client

from style import EDGE, GRAPH, NODE, area, flow, icon, primary, soft

with Diagram("FinTech AI  |  Production design (AWS CDK, not deployed)",
             filename=str(Path(__file__).with_name("production-design")), outformat="png", direction="TB",
             show=False, graph_attr=GRAPH, node_attr=NODE, edge_attr=EDGE):
    client = Client("Analyst app\n(HTTPS)")

    with area("AWS Cloud  (us-east-1)", "aws"):
        with area("API layer"):
            api = APIGateway("API Gateway REST API\n/chat, /insights, /health\n20 req/s throttle")
        with area("Compute"):
            analysis = Lambda("AI analysis\nLambda")
            role = IAMRole("Execution role\nHaiku 4.5 + ApplyGuardrail only")
        with area("Amazon Bedrock"):
            guard = icon("shield-check-red", "Bedrock Guardrail\nPII block, denied topics,\ncontent filters")
            bedrock = Bedrock("Claude Haiku 4.5\nUS inference profile")
        with area("Scheduled ingestion"):
            rule = Eventbridge("EventBridge rule\n6am, 12pm, 6pm\nweekdays")
            ingest = Lambda("Ingest\nLambda")
            market = S3("Market data bucket\nversioned, KMS")
        with area("Audit and encryption"):
            trail = Cloudtrail("CloudTrail\nmulti-Region,\nfile validation")
            audit = S3("Audit bucket\n7-year retention,\nGlacier after 90 days")
            logs = CloudwatchLogs("CloudWatch Logs\n7-year retention")
            kms = KMS("KMS key\nrotation on")

    client >> flow("POST /chat") >> api
    api >> flow("invokes") >> analysis
    role >> soft("permissions") >> analysis
    analysis >> primary("Converse API\n+ guardrailConfig") >> bedrock
    guard >> soft("screens input\nand output") >> bedrock
    rule >> flow("schedule") >> ingest
    ingest >> flow("writes run record") >> market
    trail >> flow("delivers logs") >> audit
    trail >> flow("sends events") >> logs
    kms >> soft("encrypts") >> audit
    kms >> soft("encrypts") >> market
    analysis >> Edge(style="invis") >> trail
