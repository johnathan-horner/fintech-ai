# Arrow check

Every arrow that touches an AWS service, with the official AWS page that confirms it and its direction. Arrows between non-AWS parts (Streamlit, Gemini, SQLite, scikit-learn and so on) were checked against the code only. Links come from docs.aws.amazon.com only.

## architecture.png (live)

| From | To | Action | AWS doc |
|---|---|---|---|
| Approval gate (Streamlit app) | Amazon Bedrock, Claude Haiku 4.5 | Converse API, non-streaming, us. inference profile | https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html |
| IAM user | Amazon Bedrock | Authorizes the call. Converse needs bedrock:InvokeModel, allowed only on the Haiku 4.5 profile | https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html |
| US inference profile | Claude Haiku 4.5 | Cross-Region inference routes the request inside the US | https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html |
| Amazon Bedrock | Independent grader | Converse response returns the answer to the caller | https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html |

## production-design.png (AWS CDK, not deployed)

| From | To | Action | AWS doc |
|---|---|---|---|
| Analyst app | API Gateway REST API | HTTPS request to the prod stage | https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html |
| API Gateway | AI analysis Lambda | Lambda proxy integration invokes the function | https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html |
| Execution role | Lambdas | Grants InvokeModel on Haiku 4.5 and ApplyGuardrail | https://docs.aws.amazon.com/lambda/latest/dg/lambda-intro-execution-role.html |
| AI analysis Lambda | Amazon Bedrock | Converse with guardrailConfig | https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html |
| Bedrock Guardrail | Claude Haiku 4.5 | Guardrail evaluates input and output during Converse | https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html |
| EventBridge rule | Ingest Lambda | Scheduled rule (cron, UTC) invokes the target | https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-create-rule-schedule.html |
| Ingest Lambda | Market data bucket | PutObject of a run record | https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObject.html |
| CloudTrail | Audit bucket | Delivers log files, KMS encrypted | https://docs.aws.amazon.com/awscloudtrail/latest/userguide/encrypting-cloudtrail-log-files-with-aws-kms.html |
| CloudTrail | CloudWatch Logs | Sends trail events to a log group | https://docs.aws.amazon.com/awscloudtrail/latest/userguide/send-cloudtrail-events-to-cloudwatch-logs.html |
| KMS key | S3 buckets | SSE-KMS default bucket encryption | https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingKMSEncryption.html |

## UNVERIFIED

None.
