import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RAGStoreConfig:
    vector_backend: str = os.getenv("RAG_VECTOR_BACKEND", "local").lower()
    graph_backend: str = os.getenv("RAG_GRAPH_BACKEND", "local").lower()
    canonical_backend: str = os.getenv("RAG_CANONICAL_BACKEND", "local").lower()
    object_backend: str = os.getenv("RAG_OBJECT_BACKEND", "local").lower()

    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "legal_traffic_records")
    qdrant_timeout: float = float(os.getenv("QDRANT_TIMEOUT", "120"))

    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "legalrag")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    postgres_dsn: str = os.getenv(
        "POSTGRES_DSN",
        "postgresql://legalrag:legalrag@localhost:55432/legalrag",
    )

    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    minio_bucket: str = os.getenv("MINIO_BUCKET", "legal-rag-assets")
    minio_secure: bool = _bool_env("MINIO_SECURE", False)

    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    enable_embeddings: bool = _bool_env("RAG_ENABLE_EMBEDDINGS", False)
    allow_model_download: bool = _bool_env("RAG_ALLOW_MODEL_DOWNLOAD", False)
