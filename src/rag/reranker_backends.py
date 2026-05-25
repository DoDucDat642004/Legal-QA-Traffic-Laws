import logging
import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger("RerankerBackends")

DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _model_dir(model_name: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    model_dir = Path(os.getenv("RAG_OPENVINO_RERANKER_MODEL_DIR", f"data/models/openvino/{slug}")).expanduser()
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parents[2] / model_dir
    return model_dir


def _looks_like_lfs_pointer(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size > 2048:
            return False
        return path.read_text(encoding="utf-8", errors="ignore").startswith("version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _validate_model_dir(model_dir: Path) -> None:
    required = ["config.json", "openvino_model.xml", "openvino_model.bin", "tokenizer_config.json"]
    missing = [name for name in required if not (model_dir / name).exists()]
    pointer_files = [name for name in required if _looks_like_lfs_pointer(model_dir / name)]
    if missing or pointer_files:
        details = []
        if missing:
            details.append(f"missing files: {', '.join(missing)}")
        if pointer_files:
            details.append(f"Git LFS pointer files not downloaded: {', '.join(pointer_files)}")
        raise RuntimeError(f"Local OpenVINO reranker at {model_dir} is incomplete ({'; '.join(details)}).")


class OpenVINOReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        try:
            from optimum.intel.openvino import OVModelForSequenceClassification
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("OpenVINO reranker requires optimum-intel[openvino] and transformers.") from exc

        allow_download = _bool_env("RAG_ALLOW_RERANKER_DOWNLOAD", False)
        export_model = _bool_env("RAG_OPENVINO_RERANKER_EXPORT", False)
        self.model_dir = _model_dir(model_name)
        source = self.model_dir if self.model_dir.exists() else model_name

        if not self.model_dir.exists() and not export_model:
            raise RuntimeError(
                f"OpenVINO reranker model not found at {self.model_dir}. "
                "Ship the exported model or set RAG_OPENVINO_RERANKER_EXPORT=true during build."
            )
        if self.model_dir.exists():
            _validate_model_dir(self.model_dir)

        local_files_only = (isinstance(source, Path) and source.exists()) or not allow_download
        self.tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=local_files_only)
        self.model = OVModelForSequenceClassification.from_pretrained(
            str(source),
            export=bool(not self.model_dir.exists() and export_model),
            device=os.getenv("RAG_RERANKER_DEVICE", "CPU"),
            compile=True,
            local_files_only=local_files_only,
        )

        if export_model and not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(self.model_dir)
            self.tokenizer.save_pretrained(self.model_dir)
            logger.info("Exported OpenVINO reranker to %s", self.model_dir)

        self.max_length = max(64, min(int(os.getenv("RAG_RERANKER_MAX_LENGTH", "512")), 1024))

    def predict(self, sentence_pairs: Sequence[tuple[str, str]] | Sequence[list[str]], **_: object) -> np.ndarray:
        if not sentence_pairs:
            return np.asarray([], dtype="float32")
        queries = [pair[0] for pair in sentence_pairs]
        passages = [pair[1] for pair in sentence_pairs]
        encoded = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        outputs = self.model(**encoded)
        logits = np.asarray(outputs.logits, dtype="float32")
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        elif logits.ndim == 2:
            logits = logits[:, -1]
        return logits.astype("float32")


def make_reranker(model_name: str = DEFAULT_RERANKER_MODEL):
    backend = os.getenv("RAG_RERANKER_BACKEND", "openvino").strip().lower()
    if backend in {"openvino", "ov"}:
        return OpenVINOReranker(model_name)

    from sentence_transformers import CrossEncoder

    model_path = Path(model_name).expanduser()
    allow_download = _bool_env("RAG_ALLOW_RERANKER_DOWNLOAD", False)
    if not model_path.exists() and not allow_download:
        raise RuntimeError(f"Reranker model is not local and downloads are disabled: {model_name}")
    return CrossEncoder(
        str(model_path) if model_path.exists() else model_name,
        max_length=int(os.getenv("RAG_RERANKER_MAX_LENGTH", "512")),
        device=os.getenv("RAG_RERANKER_DEVICE") or None,
    )
