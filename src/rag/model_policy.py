"""Central model allowlist and fallback helpers for Google GenAI calls."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

ALLOWED_TEXT_MODELS: tuple[str, ...] = (
    "gemma-4-31b-it",
    "gemma-4-26b-a4b",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
)

ALLOWED_VISION_MODELS: tuple[str, ...] = tuple(
    model for model in ALLOWED_TEXT_MODELS if model.startswith("gemini-")
)


def _split_env_models(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _dedupe_allowed(models: Iterable[str], *, vision: bool = False) -> list[str]:
    allowed = set(ALLOWED_VISION_MODELS if vision else ALLOWED_TEXT_MODELS)
    out: list[str] = []
    for model in models:
        if model in allowed and model not in out:
            out.append(model)
    return out


def model_candidates(*env_names: str, vision: bool = False) -> list[str]:
    """Returns allowed models, honoring env preferences only when allowlisted."""
    defaults = list(ALLOWED_VISION_MODELS if vision else ALLOWED_TEXT_MODELS)
    preferred: list[str] = []
    for env_name in env_names:
        preferred.extend(_split_env_models(os.getenv(env_name, "")))
    return _dedupe_allowed([*preferred, *defaults], vision=vision)


def first_text_model(*env_names: str) -> str:
    return model_candidates(*env_names, vision=False)[0]


def first_vision_model(*env_names: str) -> str:
    return model_candidates(*env_names, vision=True)[0]


def generate_content_with_fallback(
    client: Any,
    *,
    contents: Any,
    config: Optional[Any] = None,
    env_names: tuple[str, ...] = (),
    models: Optional[Iterable[str]] = None,
    vision: bool = False,
    logger: Optional[Any] = None,
    label: str = "generation",
    require_text: bool = True,
) -> tuple[Any, str]:
    """Calls generate_content with allowed fallback models only."""
    last_exc: Optional[Exception] = None
    candidates = _dedupe_allowed(models or model_candidates(*env_names, vision=vision), vision=vision)
    for model in candidates:
        try:
            kwargs = {"model": model, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            response = client.models.generate_content(**kwargs)
            if require_text and not getattr(response, "text", None):
                raise ValueError("API returned empty text")
            return response, model
        except Exception as exc:
            last_exc = exc
            if logger is not None:
                logger.warning("%s failed with model %s: %s", label, model, exc)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No allowed model candidates configured.")


async def async_generate_content_with_fallback(
    client: Any,
    *,
    contents: Any,
    config: Optional[Any] = None,
    env_names: tuple[str, ...] = (),
    models: Optional[Iterable[str]] = None,
    vision: bool = False,
    logger: Optional[Any] = None,
    label: str = "generation",
    require_text: bool = True,
) -> tuple[Any, str]:
    """Async variant for google-genai aio clients."""
    last_exc: Optional[Exception] = None
    candidates = _dedupe_allowed(models or model_candidates(*env_names, vision=vision), vision=vision)
    for model in candidates:
        try:
            kwargs = {"model": model, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            response = await client.aio.models.generate_content(**kwargs)
            if require_text and not getattr(response, "text", None):
                raise ValueError("API returned empty text")
            return response, model
        except Exception as exc:
            last_exc = exc
            if logger is not None:
                logger.warning("%s failed with model %s: %s", label, model, exc)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No allowed model candidates configured.")
