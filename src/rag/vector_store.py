"""
FinTech AI - FAISS vector store (faiss-cpu, no langchain-community).

Index layout (faiss_index/):
    index.faiss    - IndexFlatIP over L2-normalised embeddings (cosine similarity)
    docstore.json  - chunk texts + metadata, row-aligned with the index
    manifest.json  - {embedding_model, dim, n_docs, n_chunks, data_sha256,
                      corpus_sha256, chunk_size, chunk_overlap, tokenizer, built_at}

On load the manifest must match the configured embedding model and the
current data/corpus hashes; otherwise the index is rebuilt (needs a key).
Build locally with:  python scripts/build_index.py
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np

from src.llm.guard import call_with_retry
from src.rag.chunking import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, TOKENIZER
from src.utils.data import REPO_ROOT, data_sha256

INDEX_DIR = os.path.join(REPO_ROOT, "faiss_index")
# Gemini free tier counts every text in a batch as one embed request and allows
# 100 per minute, so the build sends <= 50 texts per batch and pauses between batches.
EMBED_BATCH = 50
EMBED_BATCH_PAUSE_S = 62.0


class IndexUnavailable(RuntimeError):
    """No usable index on disk (missing or stale) and it could not be built."""


def corpus_sha256(chunks) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.metadata["source_id"].encode())
        h.update(b"\x00")
        h.update(c.page_content.encode())
        h.update(b"\x01")
    return h.hexdigest()


def _normalise(vectors) -> np.ndarray:
    arr = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


@dataclass
class VectorIndex:
    index: object
    docs: List[dict]
    manifest: dict

    @property
    def manifest_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest()[:12]

    def search_by_vector(self, vector, k: int = 12) -> List[Tuple[dict, float]]:
        q = _normalise([vector])
        k = min(k, len(self.docs))
        scores, ids = self.index.search(q, k)
        return [(self.docs[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]


def build_index(embedder, chunks=None, path: str = INDEX_DIR, pause_s: Optional[float] = None,
                sleep=None) -> VectorIndex:
    """Embed every chunk (RETRIEVAL_DOCUMENT) and persist the FAISS index."""
    import time

    import faiss

    sleep = sleep or time.sleep
    if pause_s is None:
        pause_s = EMBED_BATCH_PAUSE_S if getattr(embedder, "model_name", "").startswith("gemini") else 0.0

    from src.rag.document_loader import load_chunks

    chunks = chunks if chunks is not None else load_chunks()
    texts = [c.page_content for c in chunks]
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        if i and pause_s:
            sleep(pause_s)
        batch = texts[i:i + EMBED_BATCH]
        vectors.extend(call_with_retry(lambda b=batch: embedder.embed_documents(b), backoff_s=max(pause_s, 3.0)))
    arr = _normalise(vectors)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)

    docs = [{"source_id": c.metadata["source_id"], "text": c.page_content, "metadata": c.metadata} for c in chunks]
    manifest = {
        "embedding_model": embedder.model_name,
        "dim": int(arr.shape[1]),
        "n_docs": len({c.metadata["source_id"].split("#")[0] for c in chunks}),
        "n_chunks": len(chunks),
        "data_sha256": data_sha256(),
        "corpus_sha256": corpus_sha256(chunks),
        "chunk_size": CHUNK_SIZE_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP_TOKENS,
        "tokenizer": TOKENIZER,
        "similarity": "cosine (IndexFlatIP on L2-normalised vectors)",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "docstore.json"), "w") as f:
        json.dump(docs, f)
    with open(os.path.join(path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return VectorIndex(index=index, docs=docs, manifest=manifest)


def manifest_problems(manifest: dict, embedding_model: str, chunks=None) -> List[str]:
    """Reasons the stored index does not match the current setup (empty = OK)."""
    from src.rag.document_loader import load_chunks

    problems = []
    if manifest.get("embedding_model") != embedding_model:
        problems.append(f"embedding model {manifest.get('embedding_model')} != {embedding_model}")
    if manifest.get("data_sha256") != data_sha256():
        problems.append("bundled data changed (data_sha256 mismatch)")
    chunks = chunks if chunks is not None else load_chunks()
    if manifest.get("corpus_sha256") != corpus_sha256(chunks):
        problems.append("document corpus changed (corpus_sha256 mismatch)")
    if manifest.get("chunk_size") != CHUNK_SIZE_TOKENS or manifest.get("chunk_overlap") != CHUNK_OVERLAP_TOKENS:
        problems.append("chunking parameters changed")
    return problems


def load_index(embedding_model: str, path: str = INDEX_DIR) -> VectorIndex:
    """Load and verify the index; raise IndexUnavailable if missing or stale."""
    import faiss

    mpath = os.path.join(path, "manifest.json")
    if not os.path.exists(mpath) or not os.path.exists(os.path.join(path, "index.faiss")):
        raise IndexUnavailable("FAISS index not built yet")
    with open(mpath) as f:
        manifest = json.load(f)
    problems = manifest_problems(manifest, embedding_model)
    if problems:
        raise IndexUnavailable("FAISS index is stale: " + "; ".join(problems))
    index = faiss.read_index(os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "docstore.json")) as f:
        docs = json.load(f)
    if index.ntotal != len(docs) or manifest.get("dim") != index.d:
        raise IndexUnavailable("FAISS index files are inconsistent")
    return VectorIndex(index=index, docs=docs, manifest=manifest)


def get_index(embedder, path: str = INDEX_DIR, auto_build: bool = True) -> VectorIndex:
    """Load the verified index, rebuilding it (needs credentials) if stale."""
    try:
        return load_index(embedder.model_name, path)
    except IndexUnavailable:
        if not auto_build:
            raise
        return build_index(embedder, path=path)


def index_status(embedding_model: Optional[str] = None, path: str = INDEX_DIR) -> dict:
    """Cheap status for the UI (no network)."""
    from src.llm import config

    embedding_model = embedding_model or config.model_id("embedding")
    mpath = os.path.join(path, "manifest.json")
    if not os.path.exists(mpath):
        return {"status": "missing", "problems": ["not built"], "manifest": None}
    with open(mpath) as f:
        manifest = json.load(f)
    problems = manifest_problems(manifest, embedding_model)
    return {"status": "ok" if not problems else "stale", "problems": problems, "manifest": manifest}


if __name__ == "__main__":
    from src.llm.provider import Embedder

    vi = build_index(Embedder())
    print(f"Built FAISS index: {vi.manifest['n_chunks']} chunks, dim {vi.manifest['dim']}")
