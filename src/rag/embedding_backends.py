import logging
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Union, Protocol

import numpy as np

logger = logging.getLogger("EmbeddingBackends")


class Embedder(Protocol):
    """Protocol defining the interface for all embedding backends."""

    def get_embedding_dimension(self) -> int:
        """Returns the dimension of the generated embeddings."""
        ...

    def encode(
        self,
        texts: List[str],
        *,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> Union[np.ndarray, List[List[float]]]:
        """Encodes a list of texts into embeddings."""
        ...


def _bool_env(name: str, default: bool = False) -> bool:
    """Reads a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _model_cache_dir(model_name: str) -> Path:
    """Generates a cache directory path for a given model name."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    return Path(os.getenv("RAG_OPENVINO_MODEL_DIR", f"data/models/openvino/{slug}"))


def _normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    """Normalizes an array of vectors to unit length."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class SentenceTransformerEmbedder:
    """
    Standard embedding backend using the SentenceTransformers library.
    Supports various transformer-based models with CPU/GPU acceleration.
    """

    def __init__(self, model_name: str):
        """
        Initializes the SentenceTransformer model.

        Args:
            model_name: Name or path of the HuggingFace model.
        """
        from sentence_transformers import SentenceTransformer

        allow_model_download = _bool_env("RAG_ALLOW_MODEL_DOWNLOAD", False)
        if not allow_model_download:
            # Enforce offline mode if downloads are restricted
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        local_model_path = Path(model_name).expanduser()
        kwargs: dict[str, Any] = {"local_files_only": not allow_model_download}
        
        device = os.getenv("RAG_EMBEDDING_DEVICE")
        if device:
            kwargs["device"] = device

        self.model = SentenceTransformer(
            str(local_model_path) if local_model_path.exists() else model_name,
            **kwargs,
        )

        max_length = os.getenv("RAG_EMBEDDING_MAX_LENGTH")
        if max_length:
            self.model.max_seq_length = int(max_length)

    def get_embedding_dimension(self) -> int:
        """Returns the size of the embedding vector."""
        return int(self.model.get_embedding_dimension())

    def encode(
        self,
        texts: List[str],
        *,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        **_: Any,
    ) -> Union[np.ndarray, List[List[float]]]:
        """
        Encodes texts using the SentenceTransformer model.

        Args:
            texts: List of strings to encode.
            batch_size: Number of texts processed at once.
            convert_to_numpy: Whether to return a numpy array.
            show_progress_bar: Whether to display a TQDM progress bar.
            normalize_embeddings: Whether to L2-normalize vectors.

        Returns:
            A numpy array or list of floats containing the embeddings.
        """
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
        )
        return np.asarray(vectors, dtype="float32")


class OpenVINOEmbedder:
    """
    High-performance embedding backend optimized for Intel CPUs using OpenVINO.
    Uses mean-pooling on token embeddings for feature extraction.
    """

    def __init__(self, model_name: str):
        """
        Initializes and optionally exports the model to OpenVINO format.

        Args:
            model_name: HuggingFace model identifier.
        """
        try:
            from optimum.intel.openvino import OVModelForFeatureExtraction
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO backend requires 'openvino' and 'optimum-intel[openvino]'."
            ) from exc

        allow_model_download = _bool_env("RAG_ALLOW_MODEL_DOWNLOAD", False)
        if not allow_model_download:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        export_model = _bool_env("RAG_OPENVINO_EXPORT", False)
        self.model_dir = _model_cache_dir(model_name)
        source = self.model_dir if self.model_dir.exists() else model_name

        if not self.model_dir.exists() and not export_model:
            raise RuntimeError(
                f"OpenVINO model not found at {self.model_dir}. "
                "Run with RAG_OPENVINO_EXPORT=true or use sentence_transformers backend."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(source),
            local_files_only=not allow_model_download,
        )

        self.model = OVModelForFeatureExtraction.from_pretrained(
            str(source),
            export=bool(not self.model_dir.exists() and export_model),
            device=os.getenv("RAG_OPENVINO_DEVICE", "CPU"),
            compile=True,
            local_files_only=not allow_model_download,
        )

        if export_model and not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)
            logger.info("Successfully exported OpenVINO model to %s", self.model_dir)

        self.max_length = int(os.getenv("RAG_EMBEDDING_MAX_LENGTH", "512"))
        self.dimension = self._infer_dimension()

    def _infer_dimension(self) -> int:
        """Determines the embedding dimension by running a dummy inference."""
        vectors = self.encode(["dimension probe"], batch_size=1, show_progress_bar=False)
        return int(vectors.shape[1])

    def get_embedding_dimension(self) -> int:
        """Returns the pre-inferred embedding dimension."""
        return self.dimension

    def encode(
        self,
        texts: List[str],
        *,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        **_: Any,
    ) -> Union[np.ndarray, List[List[float]]]:
        """
        Performs inference using OpenVINO.

        Args:
            texts: Input texts.
            batch_size: Inference batch size.
            convert_to_numpy: Return numpy array.
            show_progress_bar: Show tqdm bar.
            normalize_embeddings: Normalize resulting vectors.

        Returns:
            Encoded embeddings.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        batches = range(0, len(texts), batch_size)
        if show_progress_bar:
            try:
                from tqdm.auto import tqdm
                batches = tqdm(list(batches), desc="OpenVINO Inference")
            except ImportError:
                pass

        out = []
        for start in batches:
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            
            outputs = self.model(**encoded)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs["last_hidden_state"]
            
            # Perform mean pooling using the attention mask
            hidden = np.asarray(hidden, dtype="float32")
            mask = np.asarray(encoded["attention_mask"], dtype="float32")[..., None]
            pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-12, None)
            out.append(pooled.astype("float32"))

        vectors = np.vstack(out).astype("float32")
        if normalize_embeddings:
            vectors = _normalize_embeddings(vectors).astype("float32")
        
        return vectors if convert_to_numpy else vectors.tolist()


def make_embedder(model_name: str) -> Embedder:
    """
    Factory function to create an embedder instance based on environment configuration.

    Args:
        model_name: Identifier for the model to load.

    Returns:
        An instance of an object implementing the Embedder protocol.
    """
    backend = os.getenv("RAG_EMBEDDING_BACKEND", "sentence_transformers").strip().lower()

    if backend in {"openvino", "ov"}:
        return OpenVINOEmbedder(model_name)

    if backend == "auto":
        model_dir = _model_cache_dir(model_name)
        # Check if optimized model already exists locally
        if model_dir.exists():
            try:
                return OpenVINOEmbedder(model_name)
            except Exception as exc:
                logger.warning("Failed to load OpenVINO backend, falling back: %s", exc)
        
        # Check if user explicitly asked for export
        if _bool_env("RAG_OPENVINO_EXPORT", False):
            return OpenVINOEmbedder(model_name)
            
        logger.info("OpenVINO model not available locally; using SentenceTransformers.")
        return SentenceTransformerEmbedder(model_name)

    # Default fallback
    return SentenceTransformerEmbedder(model_name)
