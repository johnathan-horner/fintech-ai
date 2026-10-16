#!/usr/bin/env python3
"""
Run the held-out eval and write approvals.

    python scripts/run_eval.py --split heldout            # all LLM components + MRM-006 backtest
    python scripts/run_eval.py --components MRM-006       # quant backtest only (no API key needed)
    python scripts/run_eval.py --fake                     # harness smoke test with a fake LLM (no approvals)
    python scripts/run_eval.py --provider bedrock --dry-run          # cost estimate only, no model calls
    python scripts/run_eval.py --provider bedrock --components MRM-002 --max-cost 1.00

--provider picks the answering model (bedrock = Claude Haiku 4.5, gemini = Gemini free tier);
the grader is always the Gemini grader model (a different model). Bedrock runs record token
usage and cost from the converse usage metadata. Sequential and throttled; one retry on 429;
the run stops (without writing LLM approvals) if 429s persist or the cost cap is exceeded.
--dry-run refuses to proceed past the estimate and prints the per-component split needed to
keep every run <= $1.00.
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
    ap.add_argument("--provider", default="gemini", choices=["gemini", "bedrock"])
    ap.add_argument("--dry-run", action="store_true", help="estimate Bedrock cost from real prompt sizes; no calls")
    ap.add_argument("--max-cost", type=float, default=None, help="stop the run if Bedrock cost exceeds this (USD)")
    ap.add_argument("--prune-approvals", action="store_true",
                    help="drop approval records for inventory ids that no longer exist, then exit")
    args = ap.parse_args()

    if args.prune_approvals:
        from src.mrm.approvals import prune_approvals
        from src.mrm.model_inventory import FINTECH_AI_MODELS
        removed = prune_approvals([m["model_id"] for m in FINTECH_AI_MODELS])
        print(f"Removed approval records for withdrawn components: {removed or 'none'}")
        return

    if args.dry_run:
        import json

        from src.eval.harness import estimate_cost
        est = estimate_cost(args.split, [c for c in (args.components or LLM_COMPONENTS)], args.provider)
        print(json.dumps(est, indent=2))
        over = [c for c, v in est["components"].items() if v["cost_usd_est"] > 1.0]
        if over:
            print(f"STOP: single component(s) over $1.00: {over}")
        elif est["total_cost_usd_est"] > 1.0:
            print("Split by component: each component run is <= $1.00")
        else:
            print("A single run is estimated <= $1.00")
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
        if needs_llm and not config.gemini_available():
            sys.exit("No Gemini API key configured (GEMINI_API_KEY): needed for the grader and embeddings.")
        if needs_llm and args.provider == "bedrock" and not config.bedrock_available():
            sys.exit("No AWS credentials for Bedrock (default boto3 credential chain).")
        factory = live_deps_factory

    run = run_eval(args.split, comps, deps_factory=factory, throttle_s=0 if args.fake else args.throttle,
                   fake=args.fake, write_approvals=not args.no_approvals, max_items=args.max_items,
                   retry_pause_s=0 if args.fake else 60.0, provider=args.provider, max_cost_usd=args.max_cost)
    print(f"\nRun {run['run_id']} -> {run['_path']}")
    if not run["complete"]:
        print(f"INCOMPLETE: {run['stopped_reason']} (LLM approvals not written)")
    for comp, s in run["components"].items():
        if comp == "MRM-006":
            print(f"  {comp}: {s['status']} (Basel zone {s['basel_zone']})")
        else:
            print(f"  {comp}: {s['status']} n={s['n_items']} pass_rate={s['pass_rate']} "
                  f"mean_groundedness={s['mean_groundedness']} exact_fact_accuracy={s['exact_fact_accuracy']} "
                  f"bedrock_cost=${s['usage']['cost_usd']:.4f}")
    if run.get("usage"):
        u = run["usage"]
        print(f"  Bedrock usage: {u['bedrock_calls']} calls, {u['input_tokens']} in / {u['output_tokens']} out tokens, "
              f"${u['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
