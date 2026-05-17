import argparse
import json
import re
from pathlib import Path

from src.rag.legal_utils import (
    normalize_sign_code,
    penalty_summary,
    sign_group_from_code,
    source_text,
    looks_like_procedure,
)


CROSS_REF_RE = re.compile(
    r"(?:(điểm)\s+([a-zđ])\s+)?(?:(khoản)\s+(\d+)\s+)?điều\s+(\d+[a-z]?)",
    re.IGNORECASE,
)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _node_id(record: dict) -> str:
    return record.get("source_chunk_id") or record.get("id")


def _ref_key(document: str, article: str = "", clause: str = "", point: str = "") -> str:
    parts = [document or ""]
    if article:
        parts.append(f"D{article}")
    if clause:
        parts.append(f"K{clause}")
    if point:
        parts.append(f"P{point}")
    return "|".join(parts)


def _record_ref_key(record: dict) -> str:
    ref = record.get("legal_reference") or {}
    return _ref_key(
        ref.get("document") or record.get("doc_name") or "",
        str(ref.get("article") or ""),
        str(ref.get("clause") or ""),
        str(ref.get("point") or ""),
    )


def _safe_id(value: str) -> str:
    return re.sub(r"[^0-9A-Za-zÀ-ỹ_:.|-]+", "_", str(value or "")).strip("_")


def _ensure_legal_hierarchy(nodes: dict, edges: list, record: dict, chunk_id: str) -> None:
    ref = record.get("legal_reference") or {}
    document = ref.get("document") or record.get("doc_name") or ""
    article = str(ref.get("article") or "")
    clause = str(ref.get("clause") or "")
    point = str(ref.get("point") or "")
    if not document:
        return

    doc_id = f"doc::{_safe_id(document)}"
    nodes[doc_id] = {"id": doc_id, "type": "document", "name": document}
    parent_id = doc_id

    if article:
        article_id = f"{doc_id}::D{_safe_id(article)}"
        nodes[article_id] = {
            "id": article_id,
            "type": "article",
            "num": article,
            "doc_name": document,
        }
        edges.append({"source": doc_id, "target": article_id, "type": "HAS_ARTICLE"})
        parent_id = article_id

    if clause:
        clause_id = f"{parent_id}::K{_safe_id(clause)}"
        nodes[clause_id] = {
            "id": clause_id,
            "type": "clause",
            "num": clause,
            "doc_name": document,
            "article": article,
        }
        edges.append({"source": parent_id, "target": clause_id, "type": "HAS_CLAUSE"})
        parent_id = clause_id

    if point:
        point_id = f"{parent_id}::P{_safe_id(point)}"
        nodes[point_id] = {
            "id": point_id,
            "type": "point",
            "num": point,
            "doc_name": document,
            "article": article,
            "clause": clause,
        }
        edges.append({"source": parent_id, "target": point_id, "type": "HAS_POINT"})
        parent_id = point_id

    if parent_id != doc_id:
        edges.append({"source": parent_id, "target": chunk_id, "type": "HAS_CHUNK"})


def _add_penalty_node(nodes: dict, edges: list, record: dict, chunk_id: str) -> None:
    if not record.get("penalties"):
        return
    summary = penalty_summary(record)
    penalty_id = f"penalty::{_safe_id(chunk_id)}"
    nodes[penalty_id] = {
        "id": penalty_id,
        "type": "penalty",
        "doc_name": record.get("doc_name") or (record.get("legal_reference") or {}).get("document"),
        "source_chunk_id": chunk_id,
        **summary,
    }
    edges.append({"source": chunk_id, "target": penalty_id, "type": "HAS_PENALTY"})


def _add_procedure_node(nodes: dict, edges: list, record: dict, chunk_id: str) -> None:
    if not looks_like_procedure(record):
        return
    text = source_text(record)
    procedure_id = f"procedure::{_safe_id(chunk_id)}"
    metadata = record.get("metadata") or {}
    quantitative = record.get("quantitative_data") or {}
    nodes[procedure_id] = {
        "id": procedure_id,
        "type": "procedure",
        "doc_name": record.get("doc_name") or (record.get("legal_reference") or {}).get("document"),
        "source_chunk_id": chunk_id,
        "name": metadata.get("rule_type") or metadata.get("domain") or "",
        "target_audience": metadata.get("target_audience") or metadata.get("traffic_participant") or [],
        "submission_methods": [quantitative.get("submission_method")] if quantitative.get("submission_method") else [],
        "processing_time_days": quantitative.get("processing_time_days"),
        "raw_procedure_text": text[:2000],
    }
    edges.append({"source": chunk_id, "target": procedure_id, "type": "HAS_PROCEDURE"})


def _add_sign_node(nodes: dict, edges: list, record: dict, fig: dict, chunk_id: str, figure_id: str) -> None:
    code = fig.get("code")
    if not code:
        return
    normalized = normalize_sign_code(str(code))
    if not normalized:
        return
    sign_id = f"sign::{normalized}"
    existing = nodes.get(sign_id, {})
    image_paths = set(existing.get("image_paths") or [])
    if fig.get("image_path"):
        image_paths.add(fig["image_path"])
    linked_figure_ids = set(existing.get("linked_figure_ids") or [])
    linked_figure_ids.add(figure_id)
    nodes[sign_id] = {
        "id": sign_id,
        "type": "sign",
        "code": code,
        "normalized_code": normalized,
        "name": fig.get("name") or existing.get("name") or "",
        "sign_group": fig.get("sign_group") or sign_group_from_code(normalized),
        "meaning": record.get("meaning_and_usage") or record.get("qa_context") or "",
        "visual_description": fig.get("caption") or "",
        "doc_name": record.get("doc_name") or (record.get("legal_reference") or {}).get("document"),
        "source_chunk_id": chunk_id,
        "image_paths": sorted(image_paths),
        "linked_figure_ids": sorted(linked_figure_ids),
    }
    edges.append({"source": chunk_id, "target": sign_id, "type": "HAS_SIGN"})
    edges.append({"source": figure_id, "target": sign_id, "type": "REPRESENTS_SIGN"})


def export_graph(processed_files: list[Path], out_path: Path) -> dict:
    nodes = {}
    edges = []
    ref_index = {}

    for file_path in processed_files:
        records = _load_json(file_path)
        for record in records:
            rid = _node_id(record)
            if not rid:
                continue
            ref = record.get("legal_reference") or {}
            ref_key = _record_ref_key(record)
            ref_index.setdefault(ref_key, rid)
            nodes[rid] = {
                "id": rid,
                "type": "legal_chunk",
                "doc_name": record.get("doc_name") or ref.get("document"),
                "record_id": record.get("id"),
                "record_type": record.get("record_type"),
                "legal_reference": ref,
                "page_start": ref.get("page_start") or (record.get("chunk_meta") or {}).get("page_start"),
                "page_end": ref.get("page_end") or (record.get("chunk_meta") or {}).get("page_end"),
                "text_sha256": record.get("source_text_sha256"),
            }
            _ensure_legal_hierarchy(nodes, edges, record, rid)
            _add_penalty_node(nodes, edges, record, rid)
            _add_procedure_node(nodes, edges, record, rid)

            for parent in record.get("parent_hierarchy") or []:
                if not isinstance(parent, dict):
                    continue
                parent_id = f"{rid}::parent::{parent.get('kind')}::{parent.get('num')}"
                nodes[parent_id] = {
                    "id": parent_id,
                    "type": parent.get("kind") or "parent",
                    "num": parent.get("num"),
                    "title": parent.get("title"),
                    "doc_name": record.get("doc_name") or ref.get("document"),
                }
                edges.append({"source": parent_id, "target": rid, "type": "PARENT_OF"})

            for table in record.get("tables") or []:
                if not isinstance(table, dict) or not table.get("id"):
                    continue
                table_id = f"table::{table['id']}"
                nodes[table_id] = {
                    "id": table_id,
                    "type": "table",
                    "doc_name": record.get("doc_name") or ref.get("document"),
                    "source_chunk_id": rid,
                    "page": table.get("page"),
                    "image_path": table.get("image_path"),
                    "text": table.get("text"),
                    "bbox": table.get("bbox"),
                }
                edges.append({"source": rid, "target": table_id, "type": "HAS_TABLE"})

            for fig in record.get("figures") or []:
                if not isinstance(fig, dict) or not fig.get("id"):
                    continue
                fig_id = f"figure::{fig['id']}"
                nodes[fig_id] = {
                    "id": fig_id,
                    "type": "figure",
                    "code": fig.get("code"),
                    "name": fig.get("name"),
                    "caption": fig.get("caption"),
                    "doc_name": record.get("doc_name") or ref.get("document"),
                    "source_chunk_id": rid,
                    "page": fig.get("page"),
                    "image_path": fig.get("image_path"),
                }
                edges.append({"source": rid, "target": fig_id, "type": "HAS_FIGURE"})
                _add_sign_node(nodes, edges, record, fig, rid, fig_id)

            source_text = record.get("source_body_exact") or record.get("content") or ""
            doc_name = record.get("doc_name") or ref.get("document") or ""
            for match in CROSS_REF_RE.finditer(source_text):
                point = match.group(2) or ""
                clause = match.group(4) or ""
                article = match.group(5) or ""
                target_ref = _ref_key(doc_name, article, clause, point)
                edges.append({
                    "source": rid,
                    "target_ref": target_ref,
                    "target": ref_index.get(target_ref),
                    "type": "CITES",
                    "raw": match.group(0),
                })

    graph = {"nodes": list(nodes.values()), "edges": edges}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Export legal chunks, parent hierarchy, references, tables, and figures as graph JSON.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--file", help="Processed file fragment.")
    parser.add_argument("--out", default="data/graph/legal_graph.json")
    args = parser.parse_args()

    files = sorted(Path(args.processed_dir).glob("*.extracted.json"))
    if args.file:
        files = [p for p in files if args.file in p.name]
    graph = export_graph(files, Path(args.out))
    print(f"Exported graph: nodes={len(graph['nodes'])}, edges={len(graph['edges'])}, out={args.out}")


if __name__ == "__main__":
    main()
