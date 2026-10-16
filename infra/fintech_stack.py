"""
FinTech AI - AWS CDK stack (PRODUCTION target architecture - code only).

This describes the paid production deployment: Amazon Bedrock Claude Haiku 4.5
(bedrock:InvokeModel on the us.* inference profile only) behind API Gateway/Lambda,
a Bedrock Guardrail, KMS encryption, CloudTrail with 7-year audit retention and an
EventBridge schedule. It is synthesized with `cdk synth` only; the live demo calls
Bedrock directly with its own credentials and does NOT use any of these resources
(Guardrail, KMS, CloudTrail are CDK-only). Never `cdk deploy` without a cost review.

The stack is environment-agnostic (no account/region lookups), so synth needs
no AWS credentials.
"""

import json
import os
import sys

import aws_cdk as cdk
from aws_cdk import (
    Aws,
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigateway as apigw,
    aws_bedrock as bedrock,
    aws_cloudtrail as cloudtrail,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_kms as kms,
    aws_lambda as lambda_,
    aws_logs as logs,
    aws_s3 as s3,
)
from constructs import Construct

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from src.llm.config import BEDROCK_MODELS, HAIKU_45, HAIKU_45_FOUNDATION_MODEL  # noqa: E402  (single source of truth)

# Regions the us.* Claude Haiku 4.5 inference profile routes to.
HAIKU_PROFILE_REGION = "us-east-1"
HAIKU_ROUTED_REGIONS = ("us-east-1", "us-east-2", "us-west-2")

with open(os.path.join(HERE, "retention_config.json")) as f:
    RETENTION = json.load(f)

GUARDRAIL_BLOCKED_TOPICS = {
    "InsiderTrading": "Requests to trade on or obtain material non-public information.",
    "MarketManipulation": "Requests to manipulate prices, e.g. pump and dump, spoofing or front running.",
}


class FinTechAIStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # -- KMS -------------------------------------------------------------------
        data_key = kms.Key(
            self, "FinTechDataKey",
            description="FinTech AI - KMS key for data encryption at rest",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # -- S3 --------------------------------------------------------------------
        bucket_defaults = dict(
            encryption=s3.BucketEncryption.KMS,
            encryption_key=data_key,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            minimum_tls_version=1.2,
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        market_data_bucket = s3.Bucket(
            self, "MarketDataBucket",
            lifecycle_rules=[s3.LifecycleRule(transitions=[s3.Transition(
                storage_class=s3.StorageClass.INTELLIGENT_TIERING, transition_after=Duration.days(30))])],
            **bucket_defaults,
        )
        faiss_bucket = s3.Bucket(self, "FaissIndexBucket", **bucket_defaults)

        # Audit bucket: 7-year retention (2555 days) for CloudTrail logs.
        audit_bucket = s3.Bucket(
            self, "AuditTrailBucket",
            lifecycle_rules=[s3.LifecycleRule(
                id="audit-7-year-retention",
                transitions=[s3.Transition(
                    storage_class=s3.StorageClass.GLACIER,
                    transition_after=Duration.days(RETENTION["audit_bucket_glacier_transition_days"]))],
                expiration=Duration.days(RETENTION["audit_bucket_expiration_days"]),
                noncurrent_version_expiration=Duration.days(RETENTION["audit_bucket_expiration_days"]),
            )],
            **bucket_defaults,
        )

        # -- Bedrock Guardrail (mirrors src/guardrails/bedrock_guardrails.py) -------
        guardrail = bedrock.CfnGuardrail(
            self, "ComplianceGuardrail",
            name="fintech-ai-compliance",
            description="SEC/FINRA compliant guardrails for hedge fund AI",
            blocked_input_messaging="This request is outside the approved use of this system.",
            blocked_outputs_messaging="The response was blocked by the compliance guardrail.",
            kms_key_arn=data_key.key_arn,
            sensitive_information_policy_config=bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(
                pii_entities_config=[
                    bedrock.CfnGuardrail.PiiEntityConfigProperty(type="US_SOCIAL_SECURITY_NUMBER", action="BLOCK"),
                    bedrock.CfnGuardrail.PiiEntityConfigProperty(type="CREDIT_DEBIT_CARD_NUMBER", action="BLOCK"),
                    bedrock.CfnGuardrail.PiiEntityConfigProperty(type="US_PASSPORT_NUMBER", action="BLOCK"),
                    bedrock.CfnGuardrail.PiiEntityConfigProperty(type="EMAIL", action="ANONYMIZE"),
                    bedrock.CfnGuardrail.PiiEntityConfigProperty(type="PHONE", action="ANONYMIZE"),
                ],
            ),
            topic_policy_config=bedrock.CfnGuardrail.TopicPolicyConfigProperty(
                topics_config=[
                    bedrock.CfnGuardrail.TopicConfigProperty(name=name, definition=definition, type="DENY")
                    for name, definition in GUARDRAIL_BLOCKED_TOPICS.items()
                ],
            ),
            content_policy_config=bedrock.CfnGuardrail.ContentPolicyConfigProperty(
                filters_config=[
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(type="HATE", input_strength="HIGH", output_strength="HIGH"),
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(type="VIOLENCE", input_strength="MEDIUM", output_strength="MEDIUM"),
                    bedrock.CfnGuardrail.ContentFilterConfigProperty(type="MISCONDUCT", input_strength="HIGH", output_strength="HIGH"),
                ],
            ),
        )
        guardrail_version = bedrock.CfnGuardrailVersion(
            self, "ComplianceGuardrailVersion", guardrail_identifier=guardrail.attr_guardrail_id,
            description="Initial published version",
        )

        # -- IAM (least privilege) ------------------------------------------------------
        lambda_role = iam.Role(
            self, "FinTechLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")],
        )
        # Claude Haiku 4.5 only: bedrock:InvokeModel (authorizes the non-streaming Converse API) on the
        # us.* inference profile and the foundation model in the 3 regions it routes to. Streaming
        # (InvokeModelWithResponseStream) and the global.* profile are intentionally not granted.
        # ("bedrock:Converse" is not a valid IAM action.)
        profile_arns = [f"arn:{Aws.PARTITION}:bedrock:{HAIKU_PROFILE_REGION}:{Aws.ACCOUNT_ID}:inference-profile/{HAIKU_45}"]
        fm_arns = [f"arn:{Aws.PARTITION}:bedrock:{r}::foundation-model/{HAIKU_45_FOUNDATION_MODEL}"
                   for r in HAIKU_ROUTED_REGIONS]
        lambda_role.add_to_policy(iam.PolicyStatement(
            sid="BedrockInvokeHaiku45",
            actions=["bedrock:InvokeModel"],
            resources=profile_arns + fm_arns,
        ))
        lambda_role.add_to_policy(iam.PolicyStatement(
            sid="BedrockApplyGuardrail",
            actions=["bedrock:ApplyGuardrail"],
            resources=[guardrail.attr_guardrail_arn],
        ))
        market_data_bucket.grant_read_write(lambda_role)
        faiss_bucket.grant_read_write(lambda_role)
        data_key.grant_encrypt_decrypt(lambda_role)

        common_env = {
            "MARKET_DATA_BUCKET": market_data_bucket.bucket_name,
            "FAISS_BUCKET": faiss_bucket.bucket_name,
            "LLM_PROVIDER": "bedrock",
            "BEDROCK_CHAT_MODEL": BEDROCK_MODELS["chat"],
            "BEDROCK_GRADER_MODEL": BEDROCK_MODELS["grader"],
            "GUARDRAIL_ID": guardrail.attr_guardrail_id,
            "GUARDRAIL_VERSION": guardrail_version.attr_version,
        }

        def fn(name: str, folder: str, timeout: Duration, memory: int) -> lambda_.Function:
            log_group = logs.LogGroup(self, f"{name}Logs", retention=logs.RetentionDays.ONE_MONTH,
                                      removal_policy=RemovalPolicy.DESTROY)
            return lambda_.Function(
                self, name,
                runtime=lambda_.Runtime.PYTHON_3_11,
                handler="handler.main",
                code=lambda_.Code.from_asset(os.path.join(HERE, "lambda", folder)),
                role=lambda_role,
                timeout=timeout,
                memory_size=memory,
                environment=common_env,
                log_group=log_group,
            )

        ingest_fn = fn("MarketDataIngestFn", "ingest", Duration.minutes(5), 512)
        analysis_fn = fn("AIAnalysisFn", "analysis", Duration.minutes(5), 1024)

        # -- API Gateway ------------------------------------------------------------------
        api = apigw.RestApi(
            self, "FinTechAIApi",
            rest_api_name="fintech-ai-api",
            description="FinTech AI - Hedge Fund Intelligence API",
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                throttling_rate_limit=20,
                throttling_burst_limit=40,
                logging_level=apigw.MethodLoggingLevel.INFO,
                data_trace_enabled=False,  # never log request bodies (financial data)
            ),
        )
        integration = apigw.LambdaIntegration(analysis_fn)
        api.root.add_resource("chat").add_method("POST", integration)
        insights = api.root.add_resource("insights")
        for name in ("market", "risk", "portfolio"):
            insights.add_resource(name).add_method("POST", integration)
        api.root.add_resource("health").add_method("GET", integration)

        # -- EventBridge: scheduled ingestion ------------------------------------------------
        rule = events.Rule(
            self, "MarketDataRefreshRule",
            schedule=events.Schedule.cron(minute="0", hour="6,12,18", week_day="MON-FRI"),
            description="Trigger market data ingestion on trading days",
        )
        rule.add_target(targets.LambdaFunction(ingest_fn))

        # -- CloudTrail: audit trail with 7-year retention -----------------------------------
        # Bedrock runtime calls (InvokeModel / Converse / ApplyGuardrail) are recorded as
        # CloudTrail management events, which this trail captures (read + write).
        trail_log_group = logs.LogGroup(
            self, "TrailLogGroup",
            retention=getattr(logs.RetentionDays, RETENTION["cloudtrail_log_group_retention"]),
            removal_policy=RemovalPolicy.RETAIN,
        )
        trail = cloudtrail.Trail(
            self, "FinTechAITrail",
            bucket=audit_bucket,
            encryption_key=data_key,
            is_multi_region_trail=True,
            include_global_service_events=True,
            enable_file_validation=True,
            management_events=cloudtrail.ReadWriteType.ALL,
            send_to_cloud_watch_logs=True,
            cloud_watch_log_group=trail_log_group,
        )
        trail.add_lambda_event_selector([ingest_fn, analysis_fn])

        # -- Outputs ------------------------------------------------------------------------
        cdk.CfnOutput(self, "ApiEndpoint", value=api.url)
        cdk.CfnOutput(self, "MarketDataBucketName", value=market_data_bucket.bucket_name)
        cdk.CfnOutput(self, "FaissIndexBucketName", value=faiss_bucket.bucket_name)
        cdk.CfnOutput(self, "AuditBucketName", value=audit_bucket.bucket_name)
        cdk.CfnOutput(self, "GuardrailId", value=guardrail.attr_guardrail_id)
        cdk.CfnOutput(self, "KmsKeyArn", value=data_key.key_arn)
