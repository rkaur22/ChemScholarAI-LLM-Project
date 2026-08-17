"""
Central config for the chem-rag pipeline.

Style:
- Plain module-level constants for simple, rarely-changed values.
- `pydantic_settings.BaseSettings` subclasses, one per component, for anything
  that's environment-overridable and benefits from type validation. Pydantic
  auto-reads a matching env var (by field name, case-insensitive) from the
  environment or from `.env`, and raises a clear error at startup if the
  value can't be coerced to the declared type — instead of failing later,
  mid-pipeline, with a confusing error.
"""

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class DefaultSettings(BaseSettings):
    class Config:
        env_file = ".env"
        extra = "ignore"
        frozen = True

class ArxivSettings(DefaultSettings):
    # env_prefix means each field reads from a namespaced env var, e.g.
    # `search_category` as ARXIV_SEARCH_CATEGORY, not the bare SEARCH_CATEGORY.
    # This avoids collisions between components that share a field name
    class Config(DefaultSettings.Config):
        env_prefix = "ARXIV_"

    api_base_url: str = "https://export.arxiv.org/api/query"
    user_agent: str = "chem-rag-fetch/0.1"
    rate_limit_delay: float = 3.0  # seconds between requests
    page_size: int = 100  # results per API call
    max_results: int = 200  # total papers to fetch per run
    search_category: str = "cat:physics.chem-ph"
    timeout_seconds: int = 30
    output_dir: str = "data/json"
    output_file: str = "papers_compchem.jsonl"

class PostgresSettings(DefaultSettings):
    class Config(DefaultSettings.Config):
        env_prefix = "POSTGRES_"

    host: str = "localhost"
    port: int = 5432
    database: str = "chem_rag"
    user: str = "postgres"
    password: str = "postgres"

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

class EmbeddingSettings(DefaultSettings):
    class Config(DefaultSettings.Config):
        env_prefix = "EMBEDDING_"

    provider: str = "sentence-transformers" 
    model: str = "allenai/specter2"
    normalize: bool = True
    batch_size: int = 32

class GroqSettings(DefaultSettings):
    class Config(DefaultSettings.Config):
        env_prefix = "GROQ_"

    api_key: str = ""
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.2
    max_tokens: int = 1024

class RagSettings(DefaultSettings):
    class Config(DefaultSettings.Config):
        env_prefix = "RAG_"

    top_k: int = 4

class FaissSettings(DefaultSettings):
    class Config(DefaultSettings.Config):
        env_prefix = "FAISS_"

    index_path: str = "data/faiss/papers.index"

arxiv = ArxivSettings()
postgres = PostgresSettings()
embedding = EmbeddingSettings()
groq = GroqSettings()
rag = RagSettings()
faiss_cfg = FaissSettings()

# class MinioSettings(DefaultSettings):
#     endpoint: str = "http://localhost:9000"
#     access_key: str = "minio"
#     secret_key: str = "minio123"
#     bucket: str = "chem-rag-pdfs"