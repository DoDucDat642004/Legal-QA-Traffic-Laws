import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.data_pipeline.qa_generator import ALLOWED_DIFFICULTIES, ALLOWED_INTENTS, validate_qa_pair


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _base_from_qa(path: Path) -> str:
    name = path.name
    if name.endswith(".pdf.qa.json"):
        return name[: -len(".pdf.qa.json")]
    return name.replace(".qa.json", "")


def _norm_question(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def audit_qa_file(qa_path: Path, chunks_dir: Path, processed_dir: Path) -> dict:
    base = _base_from_qa(qa_path)
    qa_pairs = _load_json(qa_path)

    chunks_path = chunks_dir / f"{base}.chunks.jsonl"
    chunks = _load_jsonl(chunks_path) if chunks_path.exists() else []
    expected_chunk_ids = {c.get("source_chunk_id") for c in chunks if c.get("source_chunk_id")}

    processed_path = processed_dir / f"{base}.pdf.extracted.json"
    records = _load_json(processed_path) if processed_path.exists() else []
    source_by_chunk = {}
    for record in records:
        chunk_id = record.get("source_chunk_id")
        if chunk_id and chunk_id not in source_by_chunk:
            source_by_chunk[chunk_id] = (
                record.get("source_body_exact")
                or record.get("original_text")
                or record.get("content")
                or ""
            )

    covered_chunk_ids = {q.get("source_chunk_id") for q in qa_pairs if q.get("source_chunk_id")}
    invalid_intents = []
    invalid_difficulties = []
    invalid_quotes = []
    missing_citations = []
    no_search_queries = []

    questions = []
    for idx, qa in enumerate(qa_pairs):
        qa_id = f"{qa.get('source_chunk_id', 'unknown')}#{idx}"
        intent = str(qa.get("intent") or "").strip().upper()
        difficulty = str(qa.get("difficulty") or "").strip().upper()
        if intent not in ALLOWED_INTENTS:
            invalid_intents.append(qa_id)
        if difficulty not in ALLOWED_DIFFICULTIES:
            invalid_difficulties.append(qa_id)
        if not qa.get("citation"):
            missing_citations.append(qa_id)
        if not qa.get("search_queries"):
            no_search_queries.append(qa_id)

        source = source_by_chunk.get(qa.get("source_chunk_id"), "")
        if source and not validate_qa_pair(qa, source, qa.get("doc_name", "")):
            invalid_quotes.append(qa_id)
        questions.append(_norm_question(qa.get("question", "")))

    duplicate_count = len(questions) - len(set(questions))
    missing_chunks = sorted(expected_chunk_ids - covered_chunk_ids)

    return {
        "file": qa_path.name,
        "qa_count": len(qa_pairs),
        "expected_chunk_count": len(expected_chunk_ids),
        "covered_chunk_count": len(expected_chunk_ids & covered_chunk_ids),
        "missing_chunk_count": len(missing_chunks),
        "missing_chunk_examples": missing_chunks[:20],
        "intent_distribution": dict(Counter(str(q.get("intent") or "").strip().upper() for q in qa_pairs)),
        "difficulty_distribution": dict(Counter(str(q.get("difficulty") or "").strip().upper() for q in qa_pairs)),
        "adversarial_count": sum(1 for q in qa_pairs if q.get("is_adversarial")),
        "duplicate_question_count": duplicate_count,
        "invalid_intent_count": len(invalid_intents),
        "invalid_difficulty_count": len(invalid_difficulties),
        "invalid_quote_count": len(invalid_quotes),
        "missing_citation_count": len(missing_citations),
        "missing_search_query_count": len(no_search_queries),
        "invalid_intent_examples": invalid_intents[:10],
        "invalid_difficulty_examples": invalid_difficulties[:10],
        "invalid_quote_examples": invalid_quotes[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated QA coverage and quality.")
    parser.add_argument("--qa-dir", default="data/qa_pairs")
    parser.add_argument("--chunks-dir", default="data/chunks")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--file", help="Audit one QA file name or basename.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    qa_dir = Path(args.qa_dir)
    files = sorted(qa_dir.glob("*.qa.json"))
    if args.file:
        files = [p for p in files if args.file in p.name]

    reports = [
        audit_qa_file(path, Path(args.chunks_dir), Path(args.processed_dir))
        for path in files
    ]

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return

    for report in reports:
        print(f"\n{report['file']}")
        print(
            f"  qa={report['qa_count']} covered_chunks={report['covered_chunk_count']}/"
            f"{report['expected_chunk_count']} missing_chunks={report['missing_chunk_count']}"
        )
        print(
            f"  intents={report['intent_distribution']} difficulty={report['difficulty_distribution']} "
            f"adversarial={report['adversarial_count']} duplicates={report['duplicate_question_count']}"
        )
        print(
            f"  invalid_intents={report['invalid_intent_count']} "
            f"invalid_difficulty={report['invalid_difficulty_count']} "
            f"invalid_quotes={report['invalid_quote_count']} "
            f"missing_citations={report['missing_citation_count']} "
            f"missing_search_queries={report['missing_search_query_count']}"
        )


if __name__ == "__main__":
    main()
