"""
FinTech AI - Prompt registry.

Every prompt template is registered as ``name@semver`` together with the
first 8 hex chars of the sha256 of its exact template text. The resulting
reference string ``name@semver#sha8`` is recorded in every TraceRecord and
in the approval record, so any edit to a prompt's text changes its reference
and blocks live use until the eval harness re-approves it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


@dataclass(frozen=True)
class RegisteredPrompt:
    name: str
    version: str
    template: str

    @property
    def sha8(self) -> str:
        return sha8(self.template)

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}#{self.sha8}"

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


_REGISTRY: Dict[str, RegisteredPrompt] = {}


def register(name: str, version: str, template: str) -> RegisteredPrompt:
    if not _SEMVER.match(version):
        raise ValueError(f"Prompt version must be semver (x.y.z), got {version!r}")
    existing = _REGISTRY.get(name)
    prompt = RegisteredPrompt(name=name, version=version, template=template)
    if existing and existing.template != template and existing.version == version:
        raise ValueError(f"Prompt {name}@{version} re-registered with different text; bump the version")
    _REGISTRY[name] = prompt
    return prompt


def get(name: str) -> RegisteredPrompt:
    _ensure_loaded()
    return _REGISTRY[name]


def refs(names: Iterable[str]) -> List[str]:
    """Sorted list of ``name@version#sha8`` for the given prompt names."""
    return sorted(get(n).ref for n in names)


def all_prompts() -> Dict[str, RegisteredPrompt]:
    _ensure_loaded()
    return dict(_REGISTRY)


def _ensure_loaded() -> None:
    # Importing the templates module registers every prompt.
    from src.prompts import templates  # noqa: F401
