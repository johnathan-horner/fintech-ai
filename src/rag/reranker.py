"""
FinTech AI - cross-encoder re-ranker.

Scores (query, document) pairs with ms-marco-MiniLM-L-6-v2 running locally
through fastembed's ONNX export ``Xenova/ms-marco-MiniLM-L-6-v2`` (no torch,
no API calls). The model (~87 MB) is downloaded from Hugging Face on first use
and cached; Streamlit wraps ``get_reranker`` in ``st.cache_resource``.
"""

from __future__ import annotations

from typing import List, Tuple

RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 4


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL, encoder=None):
        self.model_name = model_name
        self._encoder = encoder

    @property
    def encoder(self):
        if self._encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._encoder = TextCrossEncoder(model_name=self.model_name)
        return self._encoder

    def score(self, query: str, texts: List[str]) -> List[float]:
        return [float(s) for s in self.encoder.rerank(query, texts)]

    def rerank(self, query: str, docs: List[dict], top_n: int = RERANK_TOP_N) -> List[Tuple[dict, float]]:
        """Return the top_n docs sorted by cross-encoder score (desc) with scores."""
        if not docs:
            return []
        scores = self.score(query, [d["text"] for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]


_reranker = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
