"""
FinTech AI - AWS CDK Infrastructure Stack
Mirrors EduAI's terraform/main.py pattern but for fintech workloads.
Provisions: S3, Lambda, API Gateway, EventBridge, KMS, CloudTrail, IAM.
"""

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    aws_iam as iam,
    aws_kms as kms,
    aws_logs as logs,
    aws_cloudtrail as cloudtrail,
    aws_events as events,
    aws_events_targets as targets,
    aws_s3_notifications as s3n,
    Duration,
    RemovalPolicy,
)
from constructs import Construct


class FinTechAIStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # ?? KMS Key ???????????????????????????????????????????????????????????
        data_key = kms.Key(
            self, "FinTechDataKey",
            description="FinTech AI - KMS key for data encryption at rest",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ?? S3 Buckets ????????????????????????????????????????????????????????
        # Market data bucket (raw ingestion)
        market_data_bucket = s3.Bucket(
            self, "MarketDataBucket",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=data_key,
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INTELLIGENT_TIERING,
                            transition_after=Duration.days(30),
                        )
                    ]
                )
            ],
        )

        # FAISS index bucket
        faiss_bucket = s3.Bucket(
            self, "FaissIndexBucket",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=data_key,
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ?? IAM Role for Lambda ???????????????????????????????????????????????
        lambda_role = iam.Role(
            self, "FinTechLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ],
        )

        # Bedrock access - least privilege
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
            ],
            resources=[
                f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
                f"arn:aws:bedrock:{self.region}::foundation-model/amazon.titan-embed-text-v1",
            ]
        ))

        # Bedrock Guardrails
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:ApplyGuardrail"],
            resources=["*"],
        ))

        # S3 access
        market_data_bucket.grant_read_write(lambda_role)
        faiss_bucket.grant_read_write(lambda_role)

        # KMS access
        data_key.grant_encrypt_decrypt(lambda_role)

        # ?? Lambda: Data Ingestion ????????????????????????????????????????????
        ingest_fn = lambda_.Function(
            self, "MarketDataIngestFn",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="handler.main",
            code=lambda_.Code.from_asset("lambda/ingest"),
            role=lambda_role,
            timeout=Duration.minutes(5),
            memory_size=1024,
            environment={
                "MARKET_DATA_BUCKET": market_data_bucket.bucket_name,
                "FAISS_BUCKET": faiss_bucket.bucket_name,
                "REGION": self.region,
            },
            log_retention=logs.RetentionDays.ONE_MONTH,
        )

        # ?? Lambda: AI Analysis ???????????????????????????????????????????????
        analysis_fn = lambda_.Function(
            self, "AIAnalysisFn",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="handler.main",
            code=lambda_.Code.from_asset("lambda/analysis"),
            role=lambda_role,
            timeout=Duration.minutes(15),
            memory_size=3008,
            environment={
                "MARKET_DATA_BUCKET": market_data_bucket.bucket_name,
                "FAISS_BUCKET": faiss_bucket.bucket_name,
                "REGION": self.region,
            },
            log_retention=logs.RetentionDays.ONE_MONTH,
        )

        # ?? API Gateway ???????????????????????????????????????????????????????
        api = apigw.RestApi(
            self, "FinTechAIApi",
            rest_api_name="fintech-ai-api",
            description="FinTech AI - Hedge Fund Intelligence API",
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
            ),
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                throttling_rate_limit=100,
                throttling_burst_limit=200,
                logging_level=apigw.MethodLoggingLevel.INFO,
                data_trace_enabled=False,  # Don't log request bodies (financial data)
            ),
        )

        analysis_integration = apigw.LambdaIntegration(analysis_fn)

        # Routes
        chat_resource = api.root.add_resource("chat")
        chat_resource.add_method("POST", analysis_integration)

        insights = api.root.add_resource("insights")
        market_r = insights.add_resource("market")
        market_r.add_method("POST", analysis_integration)
        risk_r = insights.add_resource("risk")
        risk_r.add_method("POST", analysis_integration)
        portfolio_r = insights.add_resource("portfolio")
        portfolio_r.add_method("POST", analysis_integration)

        health = api.root.add_resource("health")
        health.add_method("GET", analysis_integration)

        # ?? EventBridge: Scheduled Market Data Refresh ????????????????????????
        market_refresh_rule = events.Rule(
            self, "MarketDataRefreshRule",
            schedule=events.Schedule.cron(
                minute="0",
                hour="6,12,18",  # 6 AM, 12 PM, 6 PM UTC (pre/mid/post market)
                week_day="MON-FRI",
            ),
            description="Trigger market data ingestion on trading days",
        )
        market_refresh_rule.add_target(targets.LambdaFunction(ingest_fn))

        # ?? CloudTrail ????????????????????????????????????????????????????????
        trail_bucket = s3.Bucket(
            self, "CloudTrailBucket",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=data_key,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
        )

        trail = cloudtrail.Trail(
            self, "FinTechAITrail",
            bucket=trail_bucket,
            is_multi_region_trail=False,
            include_global_service_events=True,
            send_to_cloud_watch_logs=True,
            cloud_watch_log_group=logs.LogGroup(
                self, "TrailLogGroup",
                retention=logs.RetentionDays.ONE_YEAR,
                removal_policy=RemovalPolicy.RETAIN,
            ),
        )

        # Log all Bedrock API calls
        trail.add_event_selector(
            data_resource_type=cloudtrail.DataResourceType.LAMBDA_FUNCTION,
            data_resource_values=[
                ingest_fn.function_arn,
                analysis_fn.function_arn,
            ]
        )

        # ?? Outputs ???????????????????????????????????????????????????????????
        cdk.CfnOutput(self, "ApiEndpoint", value=api.url, description="FinTech AI API URL")
        cdk.CfnOutput(self, "MarketDataBucketName", value=market_data_bucket.bucket_name)
        cdk.CfnOutput(self, "FaissIndexBucketName", value=faiss_bucket.bucket_name)
        cdk.CfnOutput(self, "KmsKeyArn", value=data_key.key_arn)


app = cdk.App()
FinTechAIStack(
    app, "FinTechAIStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account") or "YOUR_AWS_ACCOUNT_ID",
        region=app.node.try_get_context("region") or "us-east-1",
    ),
)
app.synth()
