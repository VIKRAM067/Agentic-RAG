from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a study expert, very capable of teaching students to learn concepts easily.
    Answer only from the given context, dont hallucinate any answer.
    Always cite the sources like [Source: filename, Page: X].
    If context is insufficient say: I don't have enough information.""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "question: {question} \n\n context: {context}"),
    ]
)


answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert evaluator grading RAG system answers.

    Evaluate the answer based on:
        1. Faithfulness — is it grounded in the context or hallucinated?
        2. Relevance — does it actually answer the question?

        Return ONLY this JSON, nothing else:
            {{"score": 0.0-1.0, "faithful": true/false, "relevant": true/false, "reason": "one sentence"}}
            """,
        ),
        (
            "human",
            "question: {question} \n\n context: {context}\n\n answer: {answer} \n\n grade this answer",
        ),
    ]
)

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a query classifier for a RAG system.

    Your job is to decide the best retrieval strategy for the given question.

    Choose one of these strategies:
        - "hybrid"   → use for most questions including when user asks about content inside a specific file (e.g. "search in this pdf", "what does X say in Y.pdf")
        - "semantic" → use for conceptual questions (explain, why, how does)
        - "keyword"  → use for exact terms (formulas, names, codes)
        - "direct"   → use when no document search needed (greetings, small talk)
        - "files"    → use ONLY when user asks to LIST what documents are available (e.g. "what files do you have", "list your documents"). Do NOT use if the user is asking a question about content inside a specific file.

        Return ONLY this JSON:
            {{"strategy": "hybrid|semantic|keyword|direct|files", "reason": "one sentence"}}
            """,
        ),
        ("human", "Question: {question}"),
    ]
)


hyde_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that generates a plausible hypothetical answer to a question.
    Write it like a textbook excerpt, approximately 100 words.
    Always generate something — never say I don't know.""",
        ),
        ("human", "question: {question} \n\n Generate a hypothetical passage:"),
    ]
)
