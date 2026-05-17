import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger("EmbeddingBackends")


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _model_cache_dir(model_name: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    return Path(os.getenv("RAG_OPENVINO_MODEL_DIR", f"data/models/openvino/{slug}"))


def _normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        allow_model_download = _bool_env("RAG_ALLOW_MODEL_DOWNLOAD", False)
        if not allow_model_download:
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

    def get_embedding_dimension(self) -> int:
        return int(self.model.get_embedding_dimension())

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        **_: Any,
    ) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
        )
        return np.asarray(vectors, dtype="float32")


class OpenVINOEmbedder:
    """OpenVINO feature-extraction embedder for Intel CPU.

    It uses mean pooling over the token embeddings. Export is explicit because it
    can take time and may require network/model cache access.
    """

    def __init__(self, model_name: str):
        try:
            from optimum.intel.openvino import OVModelForFeatureExtraction
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "OpenVINO embedding backend requires openvino and optimum-intel[openvino]."
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
                f"OpenVINO model is not exported yet: {self.model_dir}. "
                "Run with RAG_OPENVINO_EXPORT=true once, or use RAG_EMBEDDING_BACKEND=sentence_transformers."
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
            logger.info("Exported OpenVINO embedding model to %s", self.model_dir)
        self.max_length = int(os.getenv("RAG_EMBEDDING_MAX_LENGTH", "512"))
        self.dimension = self._infer_dimension()

    def _infer_dimension(self) -> int:
        vectors = self.encode(["dimension probe"], batch_size=1, show_progress_bar=False)
        return int(vectors.shape[1])

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        **_: Any,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32") if hasattr(self, "dimension") else np.zeros((0, 0), dtype="float32")
        batches = range(0, len(texts), batch_size)
        if show_progress_bar:
            try:
                from tqdm.auto import tqdm

                batches = tqdm(list(batches), desc="OpenVINO batches")
            except Exception:
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
            hidden = np.asarray(hidden, dtype="float32")
            mask = np.asarray(encoded["attention_mask"], dtype="float32")[..., None]
            pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-12, None)
            out.append(pooled.astype("float32"))
        vectors = np.vstack(out).astype("float32")
        if normalize_embeddings:
            vectors = _normalize_embeddings(vectors).astype("float32")
        return vectors if convert_to_numpy else vectors.tolist()


def make_embedder(model_name: str):
    backend = os.getenv("RAG_EMBEDDING_BACKEND", "sentence_transformers").strip().lower()
    if backend in {"openvino", "ov"}:
        return OpenVINOEmbedder(model_name)
    if backend == "auto":
        model_dir = _model_cache_dir(model_name)
        if not model_dir.exists() and not _bool_env("RAG_OPENVINO_EXPORT", False):
            logger.warning(
                "OpenVINO model is not exported yet; using SentenceTransformers. "
                "Set RAG_OPENVINO_EXPORT=true once to create %s.",
                model_dir,
            )
            return SentenceTransformerEmbedder(model_name)
        try:
            return OpenVINOEmbedder(model_name)
        except Exception as exc:
            logger.warning("OpenVINO embedding backend unavailable; falling back to SentenceTransformers: %s", exc)
    return SentenceTransformerEmbedder(model_name)
