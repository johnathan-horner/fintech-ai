"""
FinTech AI - Markdown-safe formatting helpers for Streamlit.

Streamlit markdown treats a pair of ``$`` as LaTeX delimiters, so any text
containing currency must escape ``$``. User-supplied text is fully escaped.
"""

from __future__ import annotations

import re

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|~<>$])")


def md_money(text: str) -> str:
    """Escape ``$`` (currency) so Streamlit does not render LaTeX. Idempotent."""
    if text is None:
        return ""
    return re.sub(r"(?<!\\)\$", r"\\$", str(text))


def md_escape(text: str) -> str:
    """Escape every Markdown control character (for untrusted user input)."""
    if text is None:
        return ""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def money(x: float, decimals: int = 0) -> str:
    """Currency string already escaped for st.markdown / st.metric labels."""
    s = f"{abs(x):,.{decimals}f}"
    return f"-\\${s}" if x < 0 else f"\\${s}"
