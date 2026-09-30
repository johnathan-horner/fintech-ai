"""
FinTech AI - eval run results (data/eval/results/<run_id>.json).
"""

from __future__ import annotations

import glob
import json
import os
from typing import List, Optional

from src.utils.data import DATA_DIR

RESULTS_DIR = os.path.join(DATA_DIR, "eval", "results")


def results_dir() -> str:
    return os.environ.get("FINTECH_EVAL_RESULTS_DIR") or RESULTS_DIR


def list_runs(split: Optional[str] = "heldout", include_fake: bool = False) -> List[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir(), "*.json"))):
        try:
            with open(path) as f:
                run = json.load(f)
        except (OSError, ValueError):
            continue
        if split and run.get("split") != split:
            continue
        if run.get("fake") and not include_fake:
            continue
        run["_path"] = path
        runs.append(run)
    runs.sort(key=lambda r: r.get("started_at", ""))
    return runs


def latest_run_for(component: str, split: str = "heldout") -> Optional[dict]:
    for run in reversed(list_runs(split)):
        if component in run.get("components", {}):
            return run
    return None


def reference_grader_scores(component: str) -> List[float]:
    """Groundedness scores from the latest held-out run for ``component``."""
    run = latest_run_for(component)
    if not run:
        return []
    scores = []
    for item in run.get("items", []):
        if item.get("component") != component:
            continue
        for g in item.get("graders", []):
            if isinstance(g, dict) and isinstance(g.get("groundedness"), (int, float)):
                scores.append(float(g["groundedness"]))
    return scores
