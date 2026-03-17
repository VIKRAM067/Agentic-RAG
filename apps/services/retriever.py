from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever
from apps.core.config import settings
from apps.services.vector_store import vector_store
from apps.services.embeddings import embeddingManager
from typing import List, Optional


class HybridRetriever(BaseRetriever):

    top_k: int = settings.top_k
    bm25_weight: float = settings.bm25_weight
    dense_weight: float = settings.dense_weight
    strategy: str = "hybrid"
    _bm25: Optional[object] = None

    class Config:
        arbitrary_types_allowed = True

    def _build_bm25(self) -> BM25Retriever:
        docs = vector_store.get_all_documents()

        if not docs:
            return None

        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = self.top_k
        return bm25

    def _dense_search(self, query: str) -> List[Document]:
        return vector_store.similarity_search(query)

    def _rrf_fusion(
        self, dense_docs: List[Document], bm25_docs: List[Document]
    ) -> List[Document]:
        scores = {}
        doc_map = {}

        for rank, doc in enumerate(dense_docs):
            key = doc.page_content[:100]  # Use a unique identifier for the document
            scores[key] = scores.get(key, 0) + self.dense_weight * (1 / (60 + rank + 1))
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + self.bm25_weight * (1 / (60 + rank + 1))
            doc_map[key] = doc

        sorted_keys = sorted(scores, key=scores.get, reverse=True)

        return [doc_map[key] for key in sorted_keys[: self.top_k]]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.strategy == "semantic":
            return self._dense_search(query)

        elif self.strategy == "keyword":
            bm25 = self._build_bm25()
            if not bm25:
                return self._dense_search(query)
            return bm25.invoke(query)

        else:
            dense_docs = self._dense_search(query)
            bm25 = self._build_bm25()
            if not bm25:
                return dense_docs
            bm25_docs = bm25.invoke(query)
            return self._rrf_fusion(dense_docs, bm25_docs)


hybrid_retriever = HybridRetriever()
