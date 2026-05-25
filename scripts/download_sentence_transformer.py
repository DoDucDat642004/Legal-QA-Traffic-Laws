import argparse
import os
import re
from pathlib import Path


def _default_model_dir(model_name: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    return Path("data/models/sentence-transformers") / slug


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and save a SentenceTransformer model.")
    parser.add_argument("model_name")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_dir = Path(args.output_dir) if args.output_dir else _default_model_dir(args.model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model_name)
    model.save(str(output_dir))
    print(f"Saved {args.model_name} to {output_dir}")


if __name__ == "__main__":
    main()
