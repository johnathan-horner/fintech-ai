import ast
import json
import os

import pytest

from src import mcp_tools

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


CALLS = [
    (mcp_tools.analyze_portfolio_risk, (), {}),
    (mcp_tools.analyze_portfolio_risk, (["AAPL", "UNH", "ZZZZ"], 0.99), {}),
    (mcp_tools.generate_market_insights, ("Technology", "1_month"), {}),
    (mcp_tools.optimize_portfolio_allocation, ({"AAPL": 0.4, "MSFT": 0.3, "UNH": 0.3}, "moderate"), {}),
    (mcp_tools.analyze_stock_fundamentals, ("MSFT",), {}),
    (mcp_tools.get_market_regime_analysis, (), {}),
    (mcp_tools.generate_compliance_report, (), {}),
    (mcp_tools.rag_pipeline_info, (), {}),
    (mcp_tools.market_data_sources, (), {}),
]


@pytest.mark.parametrize("fn,args,kwargs", CALLS)
def test_mcp_tools_deterministic(fn, args, kwargs):
    a = json.dumps(fn(*args, **kwargs), sort_keys=True, default=str)
    b = json.dumps(fn(*args, **kwargs), sort_keys=True, default=str)
    assert a == b


def test_no_random_in_mcp_code():
    for path in ("mcp_server.py", "src/mcp_tools.py", "src/mrm/monitoring/model_monitor.py",
                 "src/mrm/validation/model_validator.py"):
        tree = ast.parse(open(os.path.join(REPO, path)).read())
        names = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert "random" not in names and "random" not in mods, path


def test_mcp_values_match_metrics():
    from src.utils import metrics
    from src.utils.data import load_portfolio

    r = mcp_tools.analyze_portfolio_risk()
    assert r["parametric_var"] == metrics.calculate_portfolio_var(load_portfolio()["positions"], 0.95)
    r2 = mcp_tools.analyze_portfolio_risk(["AAPL", "UNH", "ZZZZ"])
    assert r2["positions_analyzed"] == ["AAPL", "UNH"] and r2["unknown_symbols"] == ["ZZZZ"]
    opt = mcp_tools.optimize_portfolio_allocation({"AAPL": 0.4, "MSFT": 0.3, "UNH": 0.3}, "aggressive")
    w = opt["optimized_allocation"]
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)
    assert opt["annualized_volatility_optimized"] <= opt["annualized_volatility_current"] + 1e-9
    ret = mcp_tools.market_data_sources()["compliance"]["audit_retention_configured_in_infra"]
    assert ret["audit_bucket_expiration_days"] == 2555


def test_api_imports_without_aws_and_routes():
    pytest.importorskip("fastapi")
    from src.api.main import app, list_endpoints

    paths = list_endpoints(app)
    assert ("POST", "/chat") in paths and ("GET", "/health") in paths
    assert ("POST", "/mrm/validation/run-all") in paths
    ops = sum(1 for v in app.openapi()["paths"].values() for m in v if m in ("get", "post", "put", "delete", "patch"))
    assert len(paths) == ops


def test_api_chat_demo_without_key():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from src.api.main import app

    c = TestClient(app)
    r = c.post("/chat", json={"question": "What is my CVaR?"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "demo" and "No LLM provider configured" in body["reason"]
    assert body["trace"]["mode"] == "demo"
    assert c.get("/health").json()["status"] == "healthy"
    assert c.post("/chat", json={"question": "help me with insider trading"}).status_code == 400
