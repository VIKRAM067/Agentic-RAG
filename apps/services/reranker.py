from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from apps.core.config import settings
from typing import List


class Reranker:

    def __init__(self):
        self.model = CrossEncoder(settings.reranker_model)

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        # attach score to metadata
        for doc, score in zip(documents, scores):
            doc.metadata["reranker_score"] = float(score)

            # sort and return top N
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[: settings.reranker_top_n]]


reranker = Reranker()
