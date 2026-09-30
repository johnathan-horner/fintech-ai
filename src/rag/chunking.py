"""
FinTech AI - token-window text splitter.

Chunks are 1000 tokens with a 200-token overlap. Tokens are counted with a
simple, dependency-free approximation: every run of word characters and every
individual punctuation character is one token (regex ``\\w+|[^\\w\\s]``).
This over-counts slightly versus sub-word tokenizers for plain English and is
documented in the index manifest as ``tokenizer``. Chunks are cut on token
boundaries and keep the original text (character offsets), so short records
produce exactly one chunk.
"""

from __future__ import annotations

import re
from typing import List

CHUNK_SIZE_TOKENS = 1000
CHUNK_OVERLAP_TOKENS = 200
TOKENIZER = "regex \\w+|[^\\w\\s] (word/punctuation approximation)"

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    return sum(1 for _ in _TOKEN_RE.finditer(text))


def split_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> List[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    spans = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
    if len(spans) <= chunk_size:
        return [text.strip()] if text.strip() else []
    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(spans):
        end = min(start + chunk_size, len(spans))
        chunks.append(text[spans[start][0]:spans[end - 1][1]].strip())
        if end == len(spans):
            break
        start += step
    return chunks
