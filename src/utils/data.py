"""
FinTech AI - bundled data access (read-only).

All files under data/ are synthetic. Existing files are never modified; the
price history file is produced by scripts/generate_price_history.py.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from functools import lru_cache

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

CORE_DATA_FILES = (
    "portfolio.json",
    "market_data.json",
    "earnings_transcripts.json",
    "macro_indicators.json",
)


def data_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


@lru_cache(maxsize=None)
def _load(name: str):
    with open(data_path(name)) as f:
        return json.load(f)


def load_portfolio() -> dict:
    return copy.deepcopy(_load("portfolio.json"))


def load_market_data() -> list:
    return copy.deepcopy(_load("market_data.json"))


def load_earnings() -> list:
    return copy.deepcopy(_load("earnings_transcripts.json"))


def load_macro() -> list:
    return copy.deepcopy(_load("macro_indicators.json"))


def load_price_history() -> dict:
    return copy.deepcopy(_load("price_history.json"))


def price_history_available() -> bool:
    return os.path.exists(data_path("price_history.json"))


def data_sha256(files=CORE_DATA_FILES) -> str:
    """sha256 over the raw bytes of the given data files (in order)."""
    h = hashlib.sha256()
    for name in files:
        h.update(name.encode())
        with open(data_path(name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()
