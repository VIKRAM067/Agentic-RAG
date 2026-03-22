import chromadb
from langchain_core.documents import Document
from apps.core.config import settings
from apps.services.embeddings import embeddingManager
from typing import List


class VectorStoreManager:

    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[Document]):
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = embeddingManager.embed_documents(texts)
        ids = [f"doc_{i}_{hash(text)}" for i, text in enumerate(texts)]
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def similarity_search(
        self, query: str, top_K: int = None, filter: dict = None
    ) -> List[Document]:
        embeddedQuery = embeddingManager.embed_query(query).tolist()
        searchResults = self.collection.query(
            query_embeddings=[embeddedQuery],
            n_results=top_K or settings.top_k,
            where=filter,
        )

        documents = []
        for text, metadata, distance in zip(
            searchResults["documents"][0],
            searchResults["metadatas"][0],
            searchResults["distances"][0],
        ):
            similarity = 1 - distance
            doc = Document(
                page_content=text, metadata={**metadata, "similarity_score": similarity}
            )
            documents.append(doc)
        return documents

    def get_all_documents(self) -> List[Document]:
        results = self.collection.get()
        documents = []
        for text, metadata in zip(results["documents"], results["metadatas"]):
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        return documents


vector_store = VectorStoreManager()
