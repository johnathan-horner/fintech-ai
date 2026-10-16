import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """No real key, no network, and no writes to the repo's approvals/results/logs."""
    from src.llm import config
    from src.mrm.governance import model_governance

    monkeypatch.setattr(config, "_read_secret", lambda section, key: None)
    # Behave like Streamlit without [aws] secrets: Bedrock is skipped and the developer's
    # local AWS credential chain is never used in tests.
    monkeypatch.setattr(config, "_in_streamlit_runtime", lambda: True)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for var in ("LLM_PROVIDER", "LLM_CHAT_MODEL", "LLM_GRADER_MODEL", "LLM_HYDE_MODEL", "LLM_EMBEDDING_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FINTECH_APPROVALS_PATH", str(tmp_path / "approvals.json"))
    monkeypatch.setenv("FINTECH_EVAL_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(model_governance, "GOVERNANCE_LOG_PATH", str(tmp_path / "governance.json"))
    yield
