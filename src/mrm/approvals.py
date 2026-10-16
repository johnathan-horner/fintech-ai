"""
FinTech AI - approval records (data/mrm/approvals.json).

Written ONLY by the eval harness (scripts/run_eval.py). Approvals are keyed by
(component, provider, model_id, prompt_versions); a component may hold several
approved versions (e.g. Bedrock Claude Haiku 4.5 and the Gemini fallback).

File format (v2):
    {"approvals": {"MRM-001": [ {provider, model_id, prompt_versions, status, ...}, ... ]}}
Legacy v1 files ({"MRM-001": {...}}) are read as a one-element list with provider "gemini".
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from src.utils.data import DATA_DIR

APPROVALS_PATH = os.path.join(DATA_DIR, "mrm", "approvals.json")

VALID_STATUSES = ("APPROVED", "CONDITIONAL", "FAILED")


def approvals_path() -> str:
    return os.environ.get("FINTECH_APPROVALS_PATH") or APPROVALS_PATH


def _normalise(records) -> List[dict]:
    if isinstance(records, dict):
        records = [records]
    out = []
    for r in records or []:
        if isinstance(r, dict) and r.get("status") in VALID_STATUSES:
            out.append({"provider": "gemini", **r})
    return out


def _read(path: str) -> dict:
    if not os.path.exists(path):
        return {"approvals": {}}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {"approvals": {}}
    return data if isinstance(data, dict) else {"approvals": {}}


def load_approvals(path: Optional[str] = None) -> Dict[str, List[dict]]:
    """{component: [approval records]} (legacy single records normalised to lists)."""
    data = _read(path or approvals_path())
    return {k: v for k, v in ((k, _normalise(r)) for k, r in data.get("approvals", {}).items()) if v}


def as_list(records) -> List[dict]:
    """Accept a list, a single legacy record, or None (used for approvals passed in by callers/tests)."""
    return _normalise(records)


def approvals_mtime(path: Optional[str] = None) -> float:
    path = path or approvals_path()
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _same_version(a: dict, b: dict) -> bool:
    return (a.get("provider", "gemini") == b.get("provider", "gemini") and a.get("model_id") == b.get("model_id")
            and sorted(a.get("prompt_versions") or []) == sorted(b.get("prompt_versions") or []))


def write_approval(inventory_id: str, record: dict, path: Optional[str] = None) -> None:
    """Upsert one approval record for (component, provider, model_id, prompt_versions)."""
    if record.get("status") not in VALID_STATUSES:
        raise ValueError(f"invalid status {record.get('status')!r}")
    record = {"provider": "gemini", **record}
    path = path or approvals_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = _read(path)
    approvals = {k: _normalise(v) for k, v in data.get("approvals", {}).items()}
    existing = [r for r in approvals.get(inventory_id, []) if not _same_version(r, record)]
    approvals[inventory_id] = existing + [record]
    data["approvals"] = approvals
    data["format"] = "v2: {component: [records keyed by provider, model_id, prompt_versions]}"
    data["note"] = "Written by scripts/run_eval.py from measured evaluation runs. Do not edit by hand."
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def prune_approvals(valid_ids, path: Optional[str] = None) -> list:
    """Remove records for inventory ids that no longer exist (e.g. a component was withdrawn)."""
    path = path or approvals_path()
    if not os.path.exists(path):
        return []
    data = _read(path)
    removed = sorted(k for k in data.get("approvals", {}) if k not in set(valid_ids))
    for k in removed:
        del data["approvals"][k]
    if removed:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
    return removed
