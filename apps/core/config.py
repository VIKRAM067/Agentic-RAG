from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # app
    app_name: str = Field(default="Agentic RAG")
    app_version: str = Field(default="0.1.0")

    # LLM
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    llm_temperature: float = Field(default=0.3)
    llm_max_tokens: int = Field(default=1024)

    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # Vector Store
    chroma_persist_dir: str = Field(default="./data/vector_store")
    chroma_collection_name: str = Field(default="agentic_rag")

    # Retrieval
    top_k: int = Field(default=8)
    bm25_weight: float = Field(default=0.4)
    dense_weight: float = Field(default=0.6)

    # Reranker
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_top_n: int = Field(default=5)

    # Chunking
    parent_chunk_size: int = Field(default=1500)
    child_chunk_size: int = Field(default=300)
    chunk_overlap: int = Field(default=50)

    # Agent
    max_retries: int = Field(default=2)
    grade_threshold: float = Field(default=0.7)

    # this is not class inside a class, it's a pydantic settings config , where to fetch the information - from .env files
    class Config:
        env_file = ".env"  # read from this file
        case_sensitive = False  # # GROQ_API_KEY == groq_api_key


settings = Settings()
