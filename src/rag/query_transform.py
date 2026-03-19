"""
FinTech AI - Query Transformer
Rewrites vague analyst questions into targeted financial search queries.
Mirrors EduAI's query_transform.py using HyDE (Hypothetical Document Embeddings).
"""

import boto3
import json
from langchain_aws import ChatBedrock

REGION = "us-east-1"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

TRANSFORM_PROMPT = """You are a financial research assistant. 
Given a vague or natural-language question from a hedge fund analyst, 
rewrite it into a precise search query optimized for retrieving relevant 
financial documents (earnings transcripts, portfolio positions, macro data).

Rules:
- Include relevant ticker symbols if mentioned
- Include relevant financial terms (e.g., "EPS beat", "beta", "drawdown", "VaR")
- Keep the query under 30 words
- Return ONLY the rewritten query, nothing else

Original question: {question}
Rewritten query:"""


def get_llm():
    bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)
    return ChatBedrock(
        client=bedrock_client,
        model_id=MODEL_ID,
        model_kwargs={"max_tokens": 100, "temperature": 0.0},
    )


def transform_query(question: str) -> str:
    """Rewrite a natural language question into a financial search query."""
    try:
        llm = get_llm()
        prompt = TRANSFORM_PROMPT.format(question=question)
        response = llm.invoke(prompt)
        transformed = response.content.strip()
        print(f"Query transform: '{question}' -> '{transformed}'")
        return transformed
    except Exception as e:
        print(f"Query transform failed, using original: {e}")
        return question
