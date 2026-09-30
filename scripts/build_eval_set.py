#!/usr/bin/env python3
"""Write data/eval/fintech_eval.jsonl (reference answers computed from the bundled data)."""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.dataset import DATASET_PATH, write_dataset  # noqa: E402


def main() -> None:
    items = write_dataset()
    counts = Counter((i["component"], i["split"]) for i in items)
    print(f"Wrote {DATASET_PATH}: {len(items)} items")
    for (comp, split), n in sorted(counts.items()):
        print(f"  {comp} {split}: {n}")


if __name__ == "__main__":
    main()
