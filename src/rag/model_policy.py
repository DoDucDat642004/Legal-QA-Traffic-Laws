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

FAST_TEXT_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemma-4-26b-a4b",
    "gemma-4-31b-it",
)

DEEP_TEXT_MODELS: tuple[str, ...] = (
    "gemma-4-31b-it",
    "gemma-4-26b-a4b",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
)

TASK_DEFAULT_MODELS: dict[str, tuple[str, ...]] = {
    "answer": DEEP_TEXT_MODELS,
    "continuation": DEEP_TEXT_MODELS,
    "extraction": DEEP_TEXT_MODELS,
    "qa_generation": DEEP_TEXT_MODELS,
    "planner": FAST_TEXT_MODELS,
    "condense": FAST_TEXT_MODELS,
    "sign_probe": FAST_TEXT_MODELS,
    "vision": ALLOWED_VISION_MODELS,
}


def _split_env_models(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _dedupe_allowed(models: Iterable[str], *, vision: bool = False) -> list[str]:
    allowed = set(ALLOWED_VISION_MODELS if vision else ALLOWED_TEXT_MODELS)
    out: list[str] = []
    for model in models:
        if model in allowed and model not in out:
            out.append(model)
    return out


def _task_defaults(task: Optional[str], *, vision: bool) -> list[str]:
    if vision:
        return list(TASK_DEFAULT_MODELS["vision"])
    if not task:
        return list(ALLOWED_TEXT_MODELS)
    key = str(task).strip().lower().replace("-", "_")
    return list(TASK_DEFAULT_MODELS.get(key, ALLOWED_TEXT_MODELS))


def model_candidates(*env_names: str, vision: bool = False, task: Optional[str] = None) -> list[str]:
    """Returns allowed models, honoring env preferences only when allowlisted."""
    defaults = _task_defaults(task, vision=vision)
    preferred: list[str] = []
    for env_name in env_names:
        preferred.extend(_split_env_models(os.getenv(env_name, "")))
    return _dedupe_allowed([*preferred, *defaults], vision=vision)


def first_text_model(*env_names: str, task: Optional[str] = None) -> str:
    return model_candidates(*env_names, vision=False, task=task)[0]


def first_vision_model(*env_names: str, task: Optional[str] = None) -> str:
    return model_candidates(*env_names, vision=True, task=task)[0]


def generate_content_with_fallback(
    client: Any,
    *,
    contents: Any,
    config: Optional[Any] = None,
    env_names: tuple[str, ...] = (),
    models: Optional[Iterable[str]] = None,
    vision: bool = False,
    task: Optional[str] = None,
    logger: Optional[Any] = None,
    label: str = "generation",
    require_text: bool = True,
) -> tuple[Any, str]:
    """Calls generate_content with allowed fallback models only."""
    last_exc: Optional[Exception] = None
    candidates = _dedupe_allowed(models or model_candidates(*env_names, vision=vision, task=task), vision=vision)
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
    task: Optional[str] = None,
    logger: Optional[Any] = None,
    label: str = "generation",
    require_text: bool = True,
) -> tuple[Any, str]:
    """Async variant for google-genai aio clients."""
    last_exc: Optional[Exception] = None
    candidates = _dedupe_allowed(models or model_candidates(*env_names, vision=vision, task=task), vision=vision)
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
