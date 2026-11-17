"""FinTech AI: what runs today (live on Streamlit Community Cloud)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagrams import Diagram, Edge
from diagrams.aws.ml import Bedrock
from diagrams.aws.security import IAM
from diagrams.onprem.client import User

from style import EDGE, GRAPH, NODE, area, flow, icon, primary, soft

with Diagram("FinTech AI  |  Live architecture", filename=str(Path(__file__).with_name("architecture")),
             outformat="png", direction="TB", show=False, graph_attr=GRAPH, node_attr=NODE, edge_attr=EDGE):
    user = User("Analyst\n(browser)")

    with area("Streamlit Community Cloud", "host"):
        app = icon("streamlit", "Streamlit app\n10 live AI requests per session")

        with area("Agents (LangGraph)"):
            agents = icon("langchain", "Market > Risk > Portfolio\nagents")
            metrics = icon("gauge-red", "Risk metrics\nVaR, CVaR, stress tests")

        with area("Retrieval (RAG)"):
            search = icon("search-ink", "HyDE search\nFAISS index, 117 chunks")
            rerank = icon("cpu-ink", "Cross-encoder\nre-rank, top 4")

        with area("Governance"):
            gate = icon("shield-check-red", "Approval gate\nmodel + prompt version")
            grader = icon("scale-red", "Independent grader\nnumeric check, PII filter")
            trace = icon("scroll-text-green", "Trace record\nper answer")

    with area("AWS Cloud  (us-east-1)", "aws"):
        iam = IAM("IAM user\nHaiku 4.5 only")
        bedrock = Bedrock("Amazon Bedrock\nClaude Haiku 4.5\nUS inference profile")

    with area("Google AI  (outside AWS)", "ext"):
        gemini = icon("googlegemini", "Gemini API\nembeddings, fallback,\ngrader")

    user >> flow("question, then answer\nwith model label + grade") >> app
    app >> flow("run analysis") >> agents
    metrics >> flow("numbers") >> agents
    app >> flow("chat question") >> search
    search >> flow("24 candidates") >> rerank
    search >> soft("embed query") >> gemini
    agents >> flow("agent prompts") >> gate
    rerank >> flow("prompt + context") >> gate
    gate >> primary("Converse API (primary)") >> bedrock
    iam >> soft("authorizes") >> bedrock
    gate >> soft("fallback only") >> gemini
    bedrock >> primary("answer") >> grader
    grader >> soft("grade (other model)") >> gemini
    grader >> flow("log") >> trace
    app >> Edge(style="invis") >> metrics
