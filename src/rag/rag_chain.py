"""
FinTech AI - RAG Chain
LCEL chain with conversational memory for financial Q&A.
Mirrors EduAI's rag_chain.py.
"""

import boto3
from langchain_aws import ChatBedrock
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from src.rag.vector_store import load_vector_store
from src.rag.query_transform import transform_query

REGION = "us-east-1"
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

SYSTEM_PROMPT = """You are a senior hedge fund analyst AI assistant. 
You have access to portfolio positions, earnings transcripts, and macro economic data.

Answer the analyst's question using the provided context. Be precise and quantitative.
Always cite specific numbers (prices, percentages, ratios) when available.
If you identify risk, be direct about it. Format responses professionally.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Analysis:"""

PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=SYSTEM_PROMPT,
)


def build_rag_chain():
    """Build the conversational RAG chain."""
    bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)
    llm = ChatBedrock(
        client=bedrock_client,
        model_id=MODEL_ID,
        model_kwargs={"max_tokens": 1024, "temperature": 0.1},
    )

    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    return chain


# Module-level singleton
_chain = None


def get_rag_chain():
    global _chain
    if _chain is None:
        _chain = build_rag_chain()
    return _chain


def query(question: str) -> dict:
    """Run a financial question through the RAG chain."""
    transformed = transform_query(question)
    chain = get_rag_chain()
    result = chain({"question": transformed})
    return {
        "question": question,
        "answer": result["answer"],
        "sources": [doc.metadata for doc in result.get("source_documents", [])],
    }
