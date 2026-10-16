"""
FinTech AI - provider chain and independent-grader selection.

``run_component`` tries each configured provider in order (Bedrock Claude Haiku 4.5,
then Gemini). A provider is skipped when the component's version for that provider is
not approved (reason recorded); on ANY error from a provider (throttling, access
denied, timeout, missing credentials, ...) the next provider is tried. If every
provider is skipped by governance, ``GovernanceBlocked`` is raised; if all fail,
``ProviderChainFailed`` - the caller (run_guarded) then shows demo output.

Query-embedding failures raise ``EmbeddingUnavailable``, which is NOT retried on
another provider (embeddings are always gemini-embedding-001): the chat goes
straight to demo mode with that reason.

``grader_candidates`` never returns the model that produced the output.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from src.llm import config
from src.llm.guard import GovernanceBlocked, sanitize_error


class ProviderChainFailed(RuntimeError):
    """Every provider in the chain failed."""


class EmbeddingUnavailable(RuntimeError):
    """Query/document embedding (gemini-embedding-001) failed - no provider fallback."""


def run_component(
    inventory_id: str,
    attempt: Callable[[str], dict],
    *,
    providers: Optional[List[str]] = None,
    approvals: Optional[dict] = None,
    enforce_approval: bool = True,
) -> dict:
    from src.mrm.model_inventory import check_approval, current_version

    providers = list(providers) if providers is not None else config.available_providers()
    if not providers:
        raise ProviderChainFailed("no LLM provider configured (no AWS credentials for Bedrock, no Gemini key)")
    reasons: List[str] = []
    governance_only = True
    for prov in providers:
        cur = current_version(inventory_id, prov)
        label = config.provider_label(prov, config.model_id("chat", prov))
        if enforce_approval:
            chk = check_approval(inventory_id, cur["model_id"], cur["prompt_versions"], approvals, provider=prov)
            if not chk["allowed"]:
                reasons.append(f"{label} not used: {chk['reason']}")
                continue
        try:
            result = attempt(prov)
        except (GovernanceBlocked, EmbeddingUnavailable):
            raise
        except Exception as exc:
            governance_only = False
            reasons.append(f"{label} failed: {sanitize_error(exc)}")
            continue
        result["provider"] = prov
        result["fallback_reason"] = "; ".join(reasons) or None
        if isinstance(result.get("trace"), dict):
            result["trace"]["fallback_reason"] = result["fallback_reason"]
        return result
    msg = "; ".join(reasons)
    if governance_only:
        raise GovernanceBlocked(msg)
    raise ProviderChainFailed(msg)


def grader_candidates(answer_provider: str, answer_model: str) -> List[Tuple[str, str]]:
    """
    (provider, model_id) graders that differ from the model that produced the output:
      Haiku answered  -> Gemini grader model (if a Gemini key exists)
      Gemini answered -> Haiku (if Bedrock is available), then the Gemini grader model if it differs
    Never returns ``answer_model`` itself.
    """
    out: List[Tuple[str, str]] = []
    gemini_grader = config.model_id("grader", "gemini")
    if answer_provider == "bedrock":
        if config.gemini_available():
            out.append(("gemini", gemini_grader))
    else:
        if config.bedrock_available():
            out.append(("bedrock", config.model_id("grader", "bedrock")))
        if config.gemini_available():
            out.append(("gemini", gemini_grader))
    return [(p, m) for p, m in out if m != answer_model]
