#!/usr/bin/env python3
"""
Build the FAISS index (faiss_index/) with Gemini embeddings.

Needs GEMINI_API_KEY in the environment (or ~/.streamlit/secrets.toml /
.streamlit/secrets.toml readable by st.secrets). Uses the free tier: one
batch embedding request per 100 chunks.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm import config  # noqa: E402
from src.llm.provider import Embedder  # noqa: E402
from src.rag.vector_store import INDEX_DIR, build_index, index_status  # noqa: E402


def main() -> None:
    if not config.gemini_available():
        sys.exit("No Gemini API key configured (set GEMINI_API_KEY). Nothing built.")
    status = index_status()
    if status["status"] == "ok" and "--force" not in sys.argv:
        print(f"Index at {INDEX_DIR} is current ({status['manifest']['n_chunks']} chunks). Use --force to rebuild.")
        return
    vi = build_index(Embedder())
    m = vi.manifest
    print(f"Built {INDEX_DIR}: {m['n_chunks']} chunks from {m['n_docs']} documents, dim {m['dim']}, "
          f"model {m['embedding_model']}, manifest hash {vi.manifest_hash}")


if __name__ == "__main__":
    main()
