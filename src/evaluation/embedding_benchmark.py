import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

from src.rag.embedding_backends import make_embedder
from src.rag.record_expander import load_expanded_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding backend throughput on local CPU/OpenVINO.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model", default=os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    parser.add_argument("--backend", default=os.getenv("RAG_EMBEDDING_BACKEND", "sentence_transformers"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "64")))
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    os.environ["RAG_EMBEDDING_BACKEND"] = args.backend
    records = load_expanded_records(args.processed_dir)
    texts = [(record.get("rag_text") or "")[:4000] for record in records if record.get("rag_text")][: args.limit]
    embedder = make_embedder(args.model)

    timings = []
    total_vectors = 0
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        t0 = time.perf_counter()
        vectors = embedder.encode(batch, batch_size=args.batch_size, normalize_embeddings=True)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        total_vectors += len(vectors)

    total_time = sum(timings)
    def percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, math.ceil((len(ordered) - 1) * p)))
        return ordered[idx]

    result = {
        "model": args.model,
        "backend": args.backend,
        "dimension": embedder.get_embedding_dimension(),
        "batch_size": args.batch_size,
        "vectors": total_vectors,
        "total_seconds": round(total_time, 3),
        "vectors_per_second": round(total_vectors / total_time, 3) if total_time else 0.0,
        "batch_seconds_mean": round(statistics.mean(timings), 3) if timings else 0.0,
        "batch_seconds_p95": round(percentile(timings, 0.95), 3),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
