from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from apps.core.config import settings
from apps.services.vector_store import vector_store
from typing import List
from pathlib import Path


class IngestionManager:

    def __init__(self):
        self.textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.child_chunk_size, chunk_overlap=settings.chunk_overlap
        )

    def load_docs(self, file_path: str) -> List[Document]:
        load = PyMuPDFLoader(file_path)
        file_name = Path(file_path).name
        documents = load.load()
        for doc in documents:
            doc.metadata["source"] = file_name

        return documents  # ← outside loop

    def chunk_docs(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in documents:
            chunks = self.textSplitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk, metadata={**doc.metadata, "chunk_index": i}
                )
                chunked_docs.append(chunk_doc)
        return chunked_docs  # ← outside both loops

    def ingest(self, file_path: str) -> dict:
        loaded_docs = self.load_docs(file_path)
        chunked_docs = self.chunk_docs(loaded_docs)
        vector_store.add_documents(chunked_docs)

        return {
            "status": "success",
            "filename": Path(file_path).name,
            "pages": len(loaded_docs),
            "chunks": len(chunked_docs),
        }


ingestion_service = IngestionManager()
