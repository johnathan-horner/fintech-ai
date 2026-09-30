#!/usr/bin/env python3
"""Generate data/price_history.json (SYNTHETIC, seeded, reproducible)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.etl.price_history import write_price_history  # noqa: E402
from src.utils.data import data_path, load_portfolio  # noqa: E402


def main() -> None:
    out = data_path("price_history.json")
    data = write_price_history(out, load_portfolio())
    n = len(data["dates"]) - 1
    print(f"Wrote {out}: {len(data['tickers'])} tickers x {n} daily returns (synthetic, seed={data['generator']['seed']})")


if __name__ == "__main__":
    main()
