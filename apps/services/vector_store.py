import chromadb
from langchain_core.documents import Document
from apps.core.config import settings
from apps.services.embeddings import embeddingManager
from typing import List
import uuid


class VectorStoreManager:

    def __init__(self):
        self.chromaDB_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        self.collection = self.chromaDB_client.get_or_create_collection(
            name=settings.chroma_collection_name
        )

    def add_documents(self, documents: List[Document]):
        for doc in documents:
            doc_embedding = embeddingManager.embed_texts([doc.page_content])[0]
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            self.collection.add(
                ids=[doc_id],
                embeddings=[
                    doc_embedding.tolist()
                ],  # should be a list of lists - its the chroma db client's format
                documents=[doc.page_content],
                metadatas=[doc.metadata],
            )
            print("Added documents to vector store")

    def similarity_search(self, query: str, top_K: int = None) -> List[Document]:
        embeddedQuery = embeddingManager.embed_query(query).tolist()
        searchResults = self.collection.query(
            query_embeddings=[embeddedQuery], n_results=top_K or settings.top_k
        )

        documents = []

        for text, metadata, distance in zip(
            searchResults["documents"][0],
            searchResults["metadatas"][0],
            searchResults["distances"][0],
        ):

            """We convert to similarity (higher = more similar)
            similarity = 1 - 0.1 = 0.9   # very similar ✅
            similarity = 1 - 0.9 = 0.1   # very different ✅"""

            similarity = 1 - distance  # convert distance to similarity score
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
