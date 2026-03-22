from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from apps.core.config import settings
from apps.core.prompts import rag_prompt, answer_grader_prompt
from apps.services.retriever import hybrid_retriever
from apps.services.reranker import reranker
from typing import List
import json


def get_llm():
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def format_context(documents: List[Document]) -> str:
    if not documents:
        return "No relevant documents found."
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Document {i} | Source: {source}, Page: {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)  # ← outside loop


def build_rag_chain():
    llm = get_llm()
    chain = (
        {
            "context": RunnableLambda(
                lambda q: format_context(
                    reranker.rerank(
                        q["question"], hybrid_retriever.invoke(q["question"])
                    )
                )
            ),
            "question": RunnableLambda(lambda q: q["question"]),
            "chat_history": RunnableLambda(lambda q: q.get("chat_history", [])),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return chain


def grade_answer(question: str, context: str, answer: str) -> dict:
    llm = get_llm()
    grader_chain = answer_grader_prompt | llm | StrOutputParser()

    result = grader_chain.invoke(
        {"question": question, "context": context, "answer": answer}
    )

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {"score": 0.5, "reason": "parse error"}


rag_chain = build_rag_chain()
