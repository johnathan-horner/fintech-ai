# FinTech AI: arrow verification

Code read from `origin/main` (commit b0254c3) of github.com/johnathan-horner/fintech-ai unless a row says otherwise.
AWS docs were opened with WebFetch on 2026-11-16.

## Diagram 1: `docs/architecture/architecture.png` (What runs today)

| From | To | Action | Code evidence | AWS doc |
|---|---|---|---|---|
| Visitor (browser) | Streamlit app | HTTPS page load | docs/STREAMLIT_DEPLOY.md:18 (main file `streamlit_app/streamlit_app.py`); streamlit_app/streamlit_app.py:19-23; live URL answers 303 to the Streamlit auth page | N/A (not AWS) |
| Streamlit app | Compliance filter | checks chat question (topic blocklist); PII regex runs on every rendered output | streamlit_app/demo_pages.py:449-453 (`check_compliance`); streamlit_app/live_ai.py:149-152 (`filter_pii`); src/guardrails/bedrock_guardrails.py:36-57 | N/A (not AWS) |
| Streamlit app | Orchestrator MRM-005 | "Run multi-agent analysis" button runs the LangGraph pipeline | streamlit_app/demo_pages.py:135-157; src/agents/orchestrator.py:117-125 | N/A (not AWS) |
| Streamlit app | RAG chain MRM-001 | chat question | streamlit_app/demo_pages.py:455-460; src/rag/rag_chain.py:38-67 | N/A (not AWS) |
| Streamlit app | Metrics engine MRM-006 | computes dashboard metrics (VaR, CVaR, stress, backtest) | streamlit_app/demo_pages.py:80-119 | N/A (not AWS) |
| Orchestrator | Metrics engine | gets computed numbers for agent context | src/agents/orchestrator.py:28,88,101; src/agents/context.py | N/A (not AWS) |
| Orchestrator | Approval gate + provider chain | agent prompts go through the per-provider gate | src/agents/orchestrator.py:151-165; src/agents/base.py:59-63,104-112; src/llm/router.py:34-73 | N/A (not AWS) |
| RAG chain | Approval gate + provider chain | HyDE + answer calls go through the gate | src/rag/rag_chain.py:64-80; src/llm/router.py:49-56 | N/A (not AWS) |
| RAG chain | Cross-encoder re-rank | up to 24 candidates (12 HyDE + 12 raw-query, deduplicated), top 4 kept | src/rag/query_transform.py:22,60-74; src/rag/rag_chain.py:94-98; src/rag/reranker.py:14-15 | N/A (not AWS) |
| Cross-encoder re-rank | Hugging Face Hub | downloads ONNX model on first use | src/rag/reranker.py:4-7,26-28 (fastembed `TextCrossEncoder`) | N/A (not AWS) |
| Approval gate | approvals.json + faiss_index | reads approvals.json | src/mrm/approvals.py:21,51-54; src/mrm/model_inventory.py:438-466 | N/A (not AWS) |
| RAG chain | approvals.json + faiss_index | loads FAISS index (rebuilds only if stale) | src/rag/vector_store.py:140-167; faiss_index/manifest.json | N/A (not AWS) |
| Metrics engine | data/*.json | reads portfolio + prices | src/utils/data.py; streamlit_app/demo_pages.py:86-112 | N/A (not AWS) |
| Approval gate | Independent grader | answer + sources (grading runs inside each gated attempt) | src/rag/rag_chain.py:117-118; src/agents/base.py:70-72; src/llm/router.py:57-58 | N/A (not AWS) |
| Independent grader | Trace records | writes TraceRecord (in-process deque, max 500; per-session list) | src/mrm/trace.py:27,55-58; src/rag/rag_chain.py:120-145; streamlit_app/live_ai.py:87-88 | N/A (not AWS) |
| Approval gate | Provider clients | approved model call | src/llm/provider.py:129-164; src/llm/guard.py:182-186 | N/A (not AWS) |
| Independent grader | Provider clients | grading call on a model other than the answering one | src/mrm/grader.py:194-227; src/llm/router.py:76-93 | N/A (not AWS) |
| RAG chain | Provider clients | embeds HyDE passage + query | src/rag/query_transform.py:25-34,61-69; src/llm/provider.py:167-210 | N/A (not AWS) |
| Provider clients | Amazon Bedrock Claude Haiku 4.5 | bedrock-runtime `converse` (non-streaming) on modelId `us.anthropic.claude-haiku-4-5-20251001-v1:0`; client signs with static keys from `st.secrets["aws"]` (default region us-east-1). Used for primary answers, HyDE and for grading Gemini answers | src/llm/config.py:26-28,45-49,104-131; src/llm/provider.py:53-65,107-126; src/llm/router.py:88-90 | https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html (client calls Converse; modelId may be an inference profile ID; requires `bedrock:InvokeModel`) and https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html (a geographic profile routes within its geography, e.g. us-east-1, us-east-2, us-west-2 for a US profile) |
| Provider clients | Gemini API | generateContent (fallback answers, HyDE, grading) + embedContent (gemini-embedding-001) via langchain-google-genai, key from `st.secrets["gemini"]["api_key"]` | src/llm/config.py:35-40,93-99; src/llm/provider.py:146-174 | N/A (not AWS) |

## Diagram 2: `docs/architecture/production-design.png` (Production design (not deployed))

All rows below are drawn from `infra/fintech_stack.py` and `infra/lambda/*/handler.py`. None of these resources exist: `aws cloudformation describe-stacks --stack-name FinTechAIStack` returned "does not exist" in us-east-1, us-east-2 and us-west-2, and `list-guardrails`, `get-rest-apis`, `describe-trails` in us-east-1 returned empty lists (read-only calls, default CLI profile, 2026-11-16).

| From | To | Action | Code evidence | AWS doc |
|---|---|---|---|---|
| API client | API Gateway REST API | HTTPS request to stage `prod` (20 rps, burst 40): POST /chat, POST /insights/{market,risk,portfolio}, GET /health | infra/fintech_stack.py:191-208 | https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html (client submits API request to API Gateway) |
| API Gateway | AIAnalysisFn (Lambda) | Lambda proxy integration (API Gateway passes the request as the event) | infra/fintech_stack.py:203-208 (`LambdaIntegration(analysis_fn)`, proxy by default); infra/lambda/analysis/handler.py:18-26 reads `event["path"]`, `event["body"]` | https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html ("API Gateway passes to the integrated Lambda function an event object") |
| Lambda execution role | AIAnalysisFn, MarketDataIngestFn | execution role: InvokeModel on the Haiku 4.5 us. profile + 3 foundation-model ARNs, ApplyGuardrail, S3 read/write on 2 buckets, KMS encrypt/decrypt, basic logging | infra/fintech_stack.py:136-160,172-185 | https://docs.aws.amazon.com/lambda/latest/dg/lambda-intro-execution-role.html ("Lambda automatically assumes your execution role when you invoke your function") |
| EventBridge rule | MarketDataIngestFn | invokes on schedule cron(0 6,12,18 ? * MON-FRI) UTC | infra/fintech_stack.py:211-216 | https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-create-rule-schedule.html (scheduled rule with cron expression invokes its target; schedules are UTC) |
| AIAnalysisFn | Bedrock Claude Haiku 4.5 | bedrock-runtime `converse` with modelId from env `BEDROCK_CHAT_MODEL` (the us. profile) and `guardrailConfig` | infra/lambda/analysis/handler.py:11,27-35; infra/fintech_stack.py:162-170 | https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html (guardrailConfig field; inference profile as modelId) |
| Bedrock (Converse) | Bedrock Guardrail | guardrail evaluates input and model output during the Converse call | infra/fintech_stack.py:100-133 (CfnGuardrail + CfnGuardrailVersion); infra/lambda/analysis/handler.py:31-34 | https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html ("you include configuration information for the guardrail in calls to the Converse ... operations"; blocks content "entered by the user and ... generated by the model") |
| MarketDataIngestFn | MarketDataBucket | PutObject `ingestion-runs/<time>.json` run manifest (no data vendor wired) | infra/lambda/ingest/handler.py:16-24 | https://docs.aws.amazon.com/AmazonS3/latest/API/API_PutObject.html ("Adds an object to a bucket") |
| AIAnalysisFn | FaissIndexBucket (dashed) | read/write grant only; the handler never touches the bucket | infra/fintech_stack.py:83,159,164; infra/lambda/analysis/handler.py (no S3 call) | https://docs.aws.amazon.com/lambda/latest/dg/lambda-intro-execution-role.html (grant is a permission on the execution role, not a call) |
| AIAnalysisFn, MarketDataIngestFn | CloudWatch Logs | function logs to 1-month log groups | infra/fintech_stack.py:139,173-184 | https://docs.aws.amazon.com/lambda/latest/dg/monitoring-cloudwatchlogs.html (Lambda sends function logs to CloudWatch Logs; AWSLambdaBasicExecutionRole) |
| CloudTrail | AuditTrailBucket | delivers log files (KMS-encrypted, file validation on) | infra/fintech_stack.py:86-97,226-237 | https://docs.aws.amazon.com/awscloudtrail/latest/userguide/encrypting-cloudtrail-log-files-with-aws-kms.html ("log files ... delivered by CloudTrail to your bucket are encrypted by using ... SSE-KMS") |
| CloudTrail | CloudWatch Logs | sends events to the 7-year trail log group | infra/fintech_stack.py:221-236; infra/retention_config.json | https://docs.aws.amazon.com/awscloudtrail/latest/userguide/send-cloudtrail-events-to-cloudwatch-logs.html ("CloudTrail uses a CloudWatch Logs log group as a delivery endpoint") |
| KMS key | S3 storage layer (all 3 buckets) | SSE-KMS default bucket encryption | infra/fintech_stack.py:60-76,83,86-97 | https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingKMSEncryption.html (buckets can use SSE-KMS with a customer managed key) |
| KMS key | Bedrock Guardrail | encrypts the guardrail (`kms_key_arn`) | infra/fintech_stack.py:106 | https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-permissions-kms.html ("You encrypt your guardrails with customer managed AWS KMS keys") |

Label-only detail (no arrow): the CloudTrail label says "Lambda data events" from `trail.add_lambda_event_selector([ingest_fn, analysis_fn])` (infra/fintech_stack.py:237), confirmed against https://docs.aws.amazon.com/awscloudtrail/latest/userguide/logging-data-events-with-cloudtrail.html (Lambda `Invoke` is a data event; data events are not logged by default).

UNVERIFIED arrows: none.
