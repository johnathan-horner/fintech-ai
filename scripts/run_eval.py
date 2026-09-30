#!/usr/bin/env python3
"""
Run the held-out eval and write approvals.

    python scripts/run_eval.py --split heldout            # all LLM components + MRM-006 backtest
    python scripts/run_eval.py --components MRM-006       # quant backtest only (no API key needed)
    python scripts/run_eval.py --fake                     # harness smoke test with a fake LLM (no approvals)

Free-tier friendly: sequential, throttled (--throttle seconds between items),
one retry on HTTP 429, and the run stops (without writing LLM approvals) if
429s persist. Needs GEMINI_API_KEY for LLM components.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.harness import LLM_COMPONENTS, live_deps_factory, run_eval  # noqa: E402
from src.llm import config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="heldout", choices=["heldout", "dev"])
    ap.add_argument("--components", nargs="*", default=None, help="e.g. MRM-001 MRM-006 (default: all)")
    ap.add_argument("--throttle", type=float, default=20.0, help="seconds to wait between items")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--fake", action="store_true", help="use a fake LLM/embedder (never writes approvals)")
    ap.add_argument("--no-approvals", action="store_true")
    ap.add_argument("--prune-approvals", action="store_true",
                    help="drop approval records for inventory ids that no longer exist, then exit")
    args = ap.parse_args()

    if args.prune_approvals:
        from src.mrm.approvals import prune_approvals
        from src.mrm.model_inventory import FINTECH_AI_MODELS
        removed = prune_approvals([m["model_id"] for m in FINTECH_AI_MODELS])
        print(f"Removed approval records for withdrawn components: {removed or 'none'}")
        return

    if not args.fake:
        # Eval runs tolerate longer free-tier "high demand" (503) spells than the interactive app.
        from src.llm import guard
        guard.TRANSIENT_BACKOFFS_S = (10.0, 30.0, 60.0)

    comps = args.components or list(LLM_COMPONENTS) + ["MRM-006"]
    needs_llm = any(c in LLM_COMPONENTS for c in comps)
    if args.fake:
        from tests.fakes import fake_deps_factory
        factory = fake_deps_factory
    else:
        if needs_llm and not config.has_credentials():
            sys.exit("No Gemini API key configured (GEMINI_API_KEY). Run with --components MRM-006 or --fake.")
        factory = live_deps_factory

    run = run_eval(args.split, comps, deps_factory=factory, throttle_s=0 if args.fake else args.throttle,
                   fake=args.fake, write_approvals=not args.no_approvals, max_items=args.max_items,
                   retry_pause_s=0 if args.fake else 60.0)
    print(f"\nRun {run['run_id']} -> {run['_path']}")
    if not run["complete"]:
        print(f"INCOMPLETE: {run['stopped_reason']} (LLM approvals not written)")
    for comp, s in run["components"].items():
        if comp == "MRM-006":
            print(f"  {comp}: {s['status']} (Basel zone {s['basel_zone']})")
        else:
            print(f"  {comp}: {s['status']} n={s['n_items']} pass_rate={s['pass_rate']} "
                  f"mean_groundedness={s['mean_groundedness']} exact_fact_accuracy={s['exact_fact_accuracy']}")


if __name__ == "__main__":
    main()
