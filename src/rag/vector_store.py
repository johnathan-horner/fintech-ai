"""
FinTech AI - FAISS Vector Store
Builds and loads the FAISS index using Amazon Titan Embeddings.
Mirrors EduAI's vector_store.py.
"""

import os
import boto3
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from src.rag.document_loader import load_all_documents

INDEX_PATH = os.path.join(os.path.dirname(__file__), "../../faiss_index")
EMBEDDING_MODEL = "amazon.titan-embed-text-v1"
REGION = os.getenv("AWS_REGION", "us-east-1")


def get_embeddings():
    bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)
    return BedrockEmbeddings(
        client=bedrock_client,
        model_id=EMBEDDING_MODEL,
    )


def build_vector_store():
    """Build FAISS index from all financial documents and persist to disk."""
    print("Loading financial documents...")
    docs = load_all_documents()

    print("Building FAISS index with Titan Embeddings...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    os.makedirs(INDEX_PATH, exist_ok=True)
    vectorstore.save_local(INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}")
    return vectorstore


def load_vector_store():
    """Load existing FAISS index from disk."""
    if not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        print("No FAISS index found. Building now...")
        return build_vector_store()

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"FAISS index loaded ({vectorstore.index.ntotal} vectors).")
    return vectorstore


if __name__ == "__main__":
    build_vector_store()
