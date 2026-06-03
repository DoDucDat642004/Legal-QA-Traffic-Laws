from pathlib import Path
from typing import Optional

from src.rag.legal_utils import public_asset_path


def local_processed_asset(path: str, processed_dir: Path) -> Optional[Path]:
    """Return a local file path for a public /processed asset when it exists."""
    public_path = public_asset_path(path)
    if not public_path.startswith("/processed/"):
        return None

    relative = public_path[len("/processed/") :]
    root = processed_dir.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def image_source(path: str, *, api_url: str, processed_dir: Path) -> str:
    """Build a Streamlit-safe image source for local and Hugging Face runtimes."""
    public_path = public_asset_path(path)
    if not public_path:
        return ""
    if public_path.startswith(("http://", "https://")):
        return public_path

    local_path = local_processed_asset(public_path, processed_dir)
    if local_path:
        return str(local_path)

    if public_path.startswith("/"):
        return f"{api_url.rstrip('/')}{public_path}"
    return public_path
