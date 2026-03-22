from sentence_transformers import SentenceTransformer
from apps.core.config import settings
import numpy as np
from typing import List


class EmbeddingManager:

    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query])[0]

    def embed_HyDE(self, question: str) -> np.ndarray:
        return self.model.encode([question])[0]


embeddingManager = EmbeddingManager()
