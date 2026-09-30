"""
FinTech AI - approval records (data/mrm/approvals.json).

This file is written ONLY by the eval harness (scripts/run_eval.py). Each entry
records the exact model_id and prompt versions that were evaluated, the
resulting status and the evaluation run id used as evidence. If the file is
absent every LLM component is "not yet validated" and live calls are blocked.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from src.utils.data import DATA_DIR

APPROVALS_PATH = os.path.join(DATA_DIR, "mrm", "approvals.json")

VALID_STATUSES = ("APPROVED", "CONDITIONAL", "FAILED")


def approvals_path() -> str:
    return os.environ.get("FINTECH_APPROVALS_PATH") or APPROVALS_PATH


def load_approvals(path: Optional[str] = None) -> Dict[str, dict]:
    path = path or approvals_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    approvals = data.get("approvals", {}) if isinstance(data, dict) else {}
    return {k: v for k, v in approvals.items() if isinstance(v, dict) and v.get("status") in VALID_STATUSES}


def approvals_mtime(path: Optional[str] = None) -> float:
    path = path or approvals_path()
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def write_approval(inventory_id: str, record: dict, path: Optional[str] = None) -> None:
    """Upsert one approval record (used by scripts/run_eval.py only)."""
    if record.get("status") not in VALID_STATUSES:
        raise ValueError(f"invalid status {record.get('status')!r}")
    path = path or approvals_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"approvals": {}}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {"approvals": {}}
    data.setdefault("approvals", {})[inventory_id] = record
    data["note"] = "Written by scripts/run_eval.py from measured evaluation runs. Do not edit by hand."
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def prune_approvals(valid_ids, path: Optional[str] = None) -> list:
    """Remove records for inventory ids that no longer exist (e.g. a component was withdrawn)."""
    path = path or approvals_path()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    removed = sorted(k for k in data.get("approvals", {}) if k not in set(valid_ids))
    for k in removed:
        del data["approvals"][k]
    if removed:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
    return removed
