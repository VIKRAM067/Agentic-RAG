from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from apps.agent.graph import agent
from apps.services.chain import format_context
from typing import List


def evaluate_rag(questions: List[str], ground_truths: List[str]) -> dict:

    answers = []
    contexts = []

    # run agent for each question
    for question in questions:
        result = agent.invoke(
            {
                "question": question,
                "strategy": "hybrid",
                "documents": [],
                "context": "",
                "answer": "",
                "grade": {},
                "retry_count": 0,
                "final_answer": "",
            }
        )

        answers.append(result["final_answer"])
        contexts.append([doc.page_content for doc in result["documents"]])

        # build RAGAS dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

        dataset = Dataset.from_dict(data)

        # run evaluation
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        return {
            "faithfulness": scores["faithfulness"],
            "answer_relevancy": scores["answer_relevancy"],
            "context_precision": scores["context_precision"],
            "context_recall": scores["context_recall"],
        }
