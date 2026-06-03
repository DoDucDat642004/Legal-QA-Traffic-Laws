import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=False)

DEFAULT_VECTOR_BACKEND = "qdrant"
DEFAULT_EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
DEFAULT_EMBEDDING_BACKEND = "openvino"
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_QDRANT_COLLECTION = "legal_traffic_records_vi"
DEFAULT_OPENVINO_MODEL_DIR = "data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder"


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _str_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _lower_env(name: str, default: str) -> str:
    return _str_env(name, default).lower()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class RAGStoreConfig:
    vector_backend: str = field(default_factory=lambda: _lower_env("RAG_VECTOR_BACKEND", DEFAULT_VECTOR_BACKEND))
    graph_backend: str = field(default_factory=lambda: _lower_env("RAG_GRAPH_BACKEND", "local"))
    canonical_backend: str = field(default_factory=lambda: _lower_env("RAG_CANONICAL_BACKEND", "local"))
    object_backend: str = field(default_factory=lambda: _lower_env("RAG_OBJECT_BACKEND", "local"))

    qdrant_url: str = field(default_factory=lambda: _str_env("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str = field(default_factory=lambda: _str_env("QDRANT_API_KEY", ""))
    qdrant_collection: str = field(default_factory=lambda: _str_env("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION))
    qdrant_timeout: float = field(default_factory=lambda: _float_env("QDRANT_TIMEOUT", 120.0))

    neo4j_uri: str = field(default_factory=lambda: _str_env("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: _str_env("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _str_env("NEO4J_PASSWORD", "legalrag"))
    neo4j_database: str = field(default_factory=lambda: _str_env("NEO4J_DATABASE", "neo4j"))

    postgres_dsn: str = field(
        default_factory=lambda: _str_env(
            "POSTGRES_DSN",
            "postgresql://legalrag:legalrag@localhost:55432/legalrag",
        )
    )

    minio_endpoint: str = field(default_factory=lambda: _str_env("MINIO_ENDPOINT", "localhost:9000"))
    minio_access_key: str = field(default_factory=lambda: _str_env("MINIO_ACCESS_KEY", "minioadmin"))
    minio_secret_key: str = field(default_factory=lambda: _str_env("MINIO_SECRET_KEY", "minioadmin"))
    minio_bucket: str = field(default_factory=lambda: _str_env("MINIO_BUCKET", "legal-rag-assets"))
    minio_secure: bool = field(default_factory=lambda: _bool_env("MINIO_SECURE", False))

    embedding_model: str = field(default_factory=lambda: _str_env("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    embedding_backend: str = field(default_factory=lambda: _lower_env("RAG_EMBEDDING_BACKEND", DEFAULT_EMBEDDING_BACKEND))
    embedding_dimension: int = field(default_factory=lambda: _int_env("RAG_EMBEDDING_DIMENSION", DEFAULT_EMBEDDING_DIMENSION))
    embedding_batch_size: int = field(default_factory=lambda: _int_env("RAG_EMBEDDING_BATCH_SIZE", 64))
    enable_embeddings: bool = field(default_factory=lambda: _bool_env("RAG_ENABLE_EMBEDDINGS", True))
    allow_model_download: bool = field(default_factory=lambda: _bool_env("RAG_ALLOW_MODEL_DOWNLOAD", False))
    openvino_device: str = field(default_factory=lambda: _str_env("RAG_OPENVINO_DEVICE", "CPU"))
    openvino_model_dir: str = field(default_factory=lambda: _str_env("RAG_OPENVINO_MODEL_DIR", DEFAULT_OPENVINO_MODEL_DIR))
