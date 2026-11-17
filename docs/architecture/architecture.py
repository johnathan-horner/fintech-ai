"""FinTech AI - what runs today (traced from origin/main of johnathan-horner/fintech-ai)."""
import os

from diagrams.aws.general import User
from diagrams.aws.ml import Bedrock

from archstyle import arch, aws_cloud, box, e, layer, linkedin, outside

os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUT = "architecture"

with arch("FinTech AI", "What runs today", OUT):
    user = User("Visitor (browser)")

    with outside("Streamlit Community Cloud"):
        with layer("App"):
            ui = box("Streamlit app, 6 pages\n10 live AI requests per session")
            filt = box("Compliance filter\ntopic blocklist on chat input\nPII regex on every output")
        with layer("Retrieval (RAG)", margin=34):
            rag = box("RAG chain MRM-001\nHyDE + FAISS search\n(index committed, 117 chunks)")
            rr = box("Cross-encoder re-rank\nms-marco-MiniLM-L-6-v2\nlocal ONNX, top 4")
        with layer("Governance", margin=34):
            gate = box("Approval gate + provider chain\nexact model + prompt version\nBedrock -> Gemini -> demo mode")
        with layer("Grading and audit trail", margin=34):
            grader = box("Independent grader\n(always a different model)\n+ deterministic numeric check")
            trace = box("Trace records\nin memory, per session")
        with layer("Agents (LangGraph)"):
            orch = box("Orchestrator MRM-005\nmarket -> risk -> portfolio\nagents MRM-002..004")
        with layer("Quant engine"):
            quant = box("Metrics engine MRM-006\nVaR, Monte Carlo CVaR (10,000 seeded)\nstress tests, VaR backtest, PSI")
        with layer("Storage: synthetic data (repo)"):
            data = box("data/*.json\nportfolio, prices, macro,\nmarket data, transcripts")
        with layer("Storage: approvals + index (repo)"):
            store = box("data/mrm/approvals.json\nfaiss_index/ (FAISS + docstore)")
        with layer("Model access"):
            clients = box("Provider clients\nboto3 bedrock-runtime\nlangchain-google-genai")

    with outside("Hugging Face Hub"):
        hf = box("Re-ranker model files\n(downloaded on first use)")

    with aws_cloud():
        with layer("Generative AI"):
            llm = Bedrock("Amazon Bedrock\nClaude Haiku 4.5\nus. cross-region profile")

    with outside("Google AI (Gemini API)"):
        gem = box("Gemini API\ngemini-3.5-flash-lite (fallback answers)\ngemini-3.1-flash-lite (grader)\ngemini-embedding-001")

    user >> e("HTTPS") >> ui
    ui >> e("chat question") >> rag
    ui >> e("Run multi-agent analysis") >> orch
    ui >> e("computes dashboards") >> quant
    ui >> e("checks chat question", constraint="false") >> filt
    rag >> e("HyDE + answer prompts") >> gate
    orch >> e("agent prompts") >> gate
    orch >> e("gets computed numbers") >> quant
    rag >> e("up to 24 candidates") >> rr
    rr >> e("downloads ONNX model") >> hf
    gate >> e("reads approvals.json") >> store
    rag >> e("loads FAISS index") >> store
    quant >> e("reads portfolio + prices") >> data
    gate >> e("answer + sources") >> grader
    grader >> e("writes TraceRecord", constraint="false") >> trace
    rag >> e("embed passage + query") >> clients
    gate >> e("approved model call") >> clients
    grader >> e("grading call (other model)") >> clients
    clients >> e("Converse, primary + grader") >> llm
    clients >> e("generateContent + embedContent") >> gem

linkedin(OUT)
