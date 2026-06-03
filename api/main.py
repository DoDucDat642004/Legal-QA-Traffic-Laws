"""
Traffic Law AI API Server.

This module provides a FastAPI-based web server for legal traffic law Q&A.
It supports text-based queries, traffic sign identification, and image-based sign recognition.
"""

import asyncio
import io
import json
import logging
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from PIL import Image

from src.rag.legal_graph_rag import LegalGraphRAG
from src.rag.legal_utils import (
    ascii_lower,
    format_reference,
    normalize_sign_code,
    public_asset_path,
    record_image_paths,
    source_text,
)
from src.rag.model_policy import generate_content_with_fallback
from src.rag.query_preprocessor import PreparedQuery, prepare_chat_query

# --- Configuration & Initialization ---
load_dotenv(override=False)
logger = logging.getLogger("traffic_law_api")

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = default
    return max(minimum, min(value, maximum))

# --- FastAPI App Setup ---
app = FastAPI(
    title="Luật Giao Thông AI",
    description="Hệ thống hỏi đáp pháp luật giao thông đường bộ Việt Nam.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for document images and sign assets
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
processed_dir = os.path.join(project_root, "data", "processed")
if os.path.exists(processed_dir):
    app.mount("/processed", StaticFiles(directory=processed_dir), name="processed")

# --- Dependency Providers ---
@lru_cache(maxsize=1)
def get_rag() -> LegalGraphRAG:
    """Provides a singleton instance of the RAG engine."""
    enable_reranker = os.getenv("RAG_ENABLE_RERANKER", "false").lower() in {"1", "true", "yes", "on"}
    return LegalGraphRAG(
        "data/processed",
        graph_path="data/graph/legal_graph.json",
        use_reranker=enable_reranker,
    )

# --- Helper Functions ---
def _traffic_sign_query_hints(description: str, user_query: str = "") -> str:
    """Generates visual and legal hints for sign-related queries."""
    text = f"{description} {user_query}".lower()
    hints = ["QCVN 41:2024", "Thông tư 51/2024", "biển báo cấm", "Phụ lục B"]
    
    sign_hints = []
    if re.search(r"\bP\s*\.?\s*\d{2,3}[a-zđ]?\b", text, re.IGNORECASE):
        sign_hints.extend(re.findall(r"\bP\s*\.?\s*\d{2,3}[a-zđ]?\b", f"{description} {user_query}", re.IGNORECASE))
    
    mappings = {
        "ngược chiều": "P.102", "no entry": "P.102", "thanh ngang": "P.102",
        "vạch ngang trắng": "P.102", "vạch trắng": "P.102",
        "đường cấm": "P.101", "ô tô": "P.103a", "xe máy": "P.104",
        "xe tải": "P.106a", "người đi bộ": "P.112", "rẽ trái": "P.123a",
        "rẽ phải": "P.123b", "quay đầu": "P.124a", "vượt": "P.125",
        "tốc độ": "P.127", "dừng": "P.130", "đỗ": "P.131",
        "trẻ em": "W.225", "học sinh": "W.225", "tam giác": "biển báo nguy hiểm",
        "nền vàng": "biển báo nguy hiểm", "đèn tín hiệu": "W.209",
    }
    for phrase, code in mappings.items():
        if phrase in text: sign_hints.append(code)
        
    deduped = list(dict.fromkeys(x.strip() for x in [*hints, *sign_hints] if x and x.strip()))
    return ". ".join(deduped)

def _parse_vision_json(text: str) -> Dict[str, Any]:
    """Extracts structured JSON from Vision model output."""
    if not text: return {}
    cleaned = re.sub(r"```(?:json)?", "", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        # Fallback regex extraction
        codes = re.findall(r"\b(?:P|W|R|I|S|IE)\.?\d{2,3}[a-zđ]?\b", text, re.IGNORECASE)
        return {"candidate_codes": list(dict.fromkeys(codes)), "is_traffic_sign": bool(codes)}

def _vision_candidate_codes(vision: Dict[str, Any]) -> List[str]:
    raw_values: List[Any] = []
    for key in ["candidate_codes", "alternatives"]:
        value = vision.get(key)
        if isinstance(value, list):
            raw_values.extend(value)
    codes: List[str] = []
    for item in raw_values:
        if isinstance(item, dict):
            item = item.get("code") or item.get("sign_code") or item.get("id")
        code = normalize_sign_code(str(item or ""))
        if code:
            codes.append(code)
    return list(dict.fromkeys(codes))[:6]

def _trusted_vision_codes(rag: LegalGraphRAG, vision: Dict[str, Any]) -> List[str]:
    try:
        confidence = float(vision.get("confidence") or 0)
    except Exception:
        confidence = 0.0
    codes = [
        code
        for code in _vision_candidate_codes(vision)
        if rag.retriever.sign_catalog.lookup(code)
    ]
    if confidence >= 0.58:
        return codes[:4]
    if len(codes) == 1 and confidence >= 0.45:
        return codes[:1]
    return []

def _vision_text(vision: Dict[str, Any]) -> str:
    values: List[str] = []
    for key in ["raw_description", "shape", "dominant_colors", "symbol", "text", "sign_group"]:
        value = vision.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))
    return ". ".join(values)

def _sign_image_query(vision: Dict[str, Any], trusted_codes: List[str], user_query: str) -> str:
    visual_desc = _vision_text(vision)
    if trusted_codes:
        code_text = " ".join(trusted_codes)
        return (
            f"Biển báo {code_text}. {visual_desc}. "
            "Tra cứu ý nghĩa, hình dạng nhận dạng, phạm vi áp dụng, căn cứ hình ảnh gốc trong QCVN 41:2024 "
            "và nếu đi trái hiệu lệnh biển báo thì mức xử phạt liên quan theo Nghị định 168/2024/NĐ-CP. "
            f"Câu hỏi người dùng: {user_query}"
        )
    hints = _traffic_sign_query_hints(visual_desc, user_query)
    return (
        f"{hints}. {visual_desc}. "
        "Ảnh biển báo chưa đủ chắc chắn mã số; tra cứu nhóm biển, ý nghĩa, căn cứ gốc và nêu rõ nếu cần xác nhận thêm. "
        f"Câu hỏi người dùng: {user_query}"
    )

def _references(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Formats source records for API metadata."""
    def images_for_doc(doc: Dict[str, Any]) -> List[str]:
        return [public_asset_path(path) for path in record_image_paths(doc)]

    references = []
    for d in docs:
        images = images_for_doc(d)
        ref = d.get("legal_reference") or {}
        references.append({
            "source_chunk_id": d.get("source_chunk_id"),
            "reference_text": format_reference(d),
            "doc_name": d.get("doc_name") or ref.get("document"),
            "modality": d.get("rag_modality"),
            "legal_reference": ref,
            "page_start": ref.get("page_start") or d.get("page_start"),
            "page_end": ref.get("page_end") or d.get("page_end"),
            "image": images[0] if images else "",
            "images": images,
            "retrieval_reasons": d.get("retrieval_reasons", []),
            "retrieval_score": d.get("retrieval_score"),
            "retrieval_slot_id": d.get("retrieval_slot_id"),
            "retrieval_slot_facet": d.get("retrieval_slot_facet"),
            "excerpt": _snippet(source_text(d), 900),
            "penalties": d.get("penalties") or {},
        })
    return references

def _context_images(docs: List[Dict[str, Any]], *, limit: int = 80) -> List[str]:
    ranked: List[tuple[int, str]] = []
    seen = set()
    for idx, doc in enumerate(docs):
        modality = str(doc.get("rag_modality") or "")
        slot = str(doc.get("retrieval_slot_facet") or "")
        reason_text = " ".join(str(x) for x in (doc.get("retrieval_reasons") or []))
        priority = 50 + idx
        if modality == "sign":
            priority = 0 + idx
        elif slot == "source_image":
            priority = 10 + idx
        elif modality in {"figure", "table"}:
            priority = 20 + idx
        elif "legal_detail" in reason_text or "document_overview" in reason_text:
            priority = 35 + idx
        for path in record_image_paths(doc):
            public = public_asset_path(path)
            if public and public not in seen:
                seen.add(public)
                ranked.append((priority, public))
    return [path for _priority, path in sorted(ranked, key=lambda item: item[0])[:limit]]


def _api_image_limit() -> int:
    return _env_int("RAG_API_IMAGE_LIMIT", 16, minimum=0, maximum=80)


def _maybe_graph_trace(rag: LegalGraphRAG, docs: List[Dict[str, Any]], *, depth: int = 3, max_nodes: int = 70) -> Dict[str, Any] | None:
    if not _env_bool("RAG_INCLUDE_GRAPH_TRACE", False):
        return None
    return _graph_trace(rag, docs, depth=depth, max_nodes=max_nodes)


def _timeout_fallback_result(rag: LegalGraphRAG, query: str) -> Dict[str, Any]:
    plan, profile = rag._build_query_profile(query)
    budget = dict(getattr(profile, "retrieval_budget", None) or {})
    budget["top_k"] = min(int(budget.get("top_k") or 16), _env_int("RAG_TIMEOUT_FALLBACK_TOP_K", 16, minimum=6, maximum=40))
    budget["expand_depth"] = min(int(budget.get("expand_depth") or 1), _env_int("RAG_TIMEOUT_FALLBACK_EXPAND_DEPTH", 1, minimum=0, maximum=2))
    budget["max_contexts"] = min(int(budget.get("max_contexts") or 12), _env_int("RAG_TIMEOUT_FALLBACK_CONTEXTS", 12, minimum=4, maximum=32))
    profile.retrieval_budget = budget
    docs = rag._retrieve_direct(query, plan, profile)
    docs = docs[: _env_int("RAG_TIMEOUT_FALLBACK_CONTEXTS", 12, minimum=4, maximum=32)]
    deterministic = rag._deterministic_structured_answer(query, docs)
    answer = deterministic or rag._extractive_answer(query, docs)
    images = rag._context_images(docs, limit=_api_image_limit())
    return {
        "answer": answer,
        "contexts": docs,
        "references": rag.format_references(docs),
        "images": images,
        "query_analysis": rag._analysis_payload(plan, profile),
        "metadata": {
            "route": "timeout_fallback",
            "sequential": False,
            "reason": "full_query_exceeded_deadline",
        },
    }


def _prepare_query_for_chat(client: Any, query: str, chat_history: List[Dict[str, Any]]) -> PreparedQuery:
    return prepare_chat_query(client, query, chat_history)


def _attach_query_preprocessing(analysis: Dict[str, Any], prepared: PreparedQuery) -> Dict[str, Any]:
    payload = prepared.public_payload()
    analysis = dict(analysis or {})
    analysis["query_preprocessing"] = payload
    if prepared.missing_data_hints:
        existing = analysis.get("missing_data_hints") or []
        hints = []
        seen = set()
        for item in [*existing, *prepared.missing_data_hints]:
            text = str(item or "").strip()
            key = ascii_lower(text)
            if text and key not in seen:
                seen.add(key)
                hints.append(text)
        analysis["missing_data_hints"] = hints
    return analysis

def _snippet(text: str, limit: int = 800) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"

def _record_payload(record: Dict[str, Any], *, text_limit: int = 1200) -> Dict[str, Any]:
    ref = record.get("legal_reference") or {}
    images = [public_asset_path(path) for path in record_image_paths(record)]
    return {
        "source_chunk_id": record.get("source_chunk_id") or record.get("id"),
        "record_id": record.get("record_id") or record.get("id"),
        "reference_text": format_reference(record),
        "doc_name": record.get("doc_name") or ref.get("document"),
        "modality": record.get("rag_modality") or record.get("type") or "text",
        "legal_reference": ref,
        "page_start": ref.get("page_start") or record.get("page_start"),
        "page_end": ref.get("page_end") or record.get("page_end"),
        "retrieval_score": record.get("retrieval_score"),
        "retrieval_reasons": record.get("retrieval_reasons") or [],
        "retrieval_slot_id": record.get("retrieval_slot_id"),
        "retrieval_slot_facet": record.get("retrieval_slot_facet"),
        "images": images,
        "image": images[0] if images else "",
        "excerpt": _snippet(source_text(record), text_limit),
        "penalties": record.get("penalties") or {},
        "rag_metadata": record.get("rag_metadata") or {},
    }

def _node_label(node: Dict[str, Any]) -> str:
    node_type = node.get("type") or "node"
    if node_type == "document":
        return str(node.get("name") or node.get("id") or "document")
    if node_type in {"article", "clause", "point"}:
        bits = [node_type]
        if node.get("num"):
            bits.append(str(node["num"]))
        if node.get("doc_name"):
            bits.append(str(node["doc_name"]))
        return " ".join(bits)
    ref = node.get("legal_reference") or {}
    parts = []
    if ref.get("point"):
        parts.append(f"Điểm {ref.get('point')}")
    if ref.get("clause"):
        parts.append(f"Khoản {ref.get('clause')}")
    if ref.get("article"):
        parts.append(f"Điều {ref.get('article')}")
    if ref.get("document") or node.get("doc_name"):
        parts.append(str(ref.get("document") or node.get("doc_name")))
    if parts:
        return ", ".join(parts)
    return str(node.get("code") or node.get("normalized_code") or node.get("id") or node_type)

def _node_payload(node: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": node.get("id"),
        "type": node.get("type"),
        "label": _node_label(node),
        "doc_name": node.get("doc_name") or node.get("name"),
        "legal_reference": node.get("legal_reference") or {},
        "source_chunk_id": node.get("source_chunk_id") or node.get("id"),
        "page_start": node.get("page_start"),
        "page_end": node.get("page_end"),
        "graph_distance": node.get("graph_distance"),
        "graph_cost": node.get("graph_cost"),
        "graph_via": node.get("graph_via"),
    }

def _neo4j_edges_between(graph_store: Any, node_ids: set[str], *, limit: int = 140) -> List[Dict[str, Any]]:
    if not node_ids or not hasattr(graph_store, "driver"):
        return []
    try:
        with graph_store.driver.session(database=graph_store.config.neo4j_database) as session:
            rows = session.run(
                """
                MATCH (a:LegalNode)-[r]->(b:LegalNode)
                WHERE a.id IN $ids AND b.id IN $ids
                RETURN a.id AS source, b.id AS target, type(r) AS type, r.data_json AS data_json
                LIMIT $limit
                """,
                ids=list(node_ids),
                limit=limit,
            )
            edges = []
            for row in rows:
                data = {}
                try:
                    data = json.loads(row["data_json"] or "{}")
                except Exception:
                    data = {}
                edges.append({
                    "source": row["source"],
                    "target": row["target"],
                    "type": row["type"] or data.get("type") or "RELATED",
                    "raw": data.get("raw") or "",
                    "target_ref": data.get("target_ref") or "",
                })
            return edges
    except Exception as exc:
        logger.warning("Could not read Neo4j trace edges: %s", exc)
        return []

def _graph_store_stats(graph_store: Any) -> Dict[str, Any]:
    if hasattr(graph_store, "nodes") and hasattr(graph_store, "out_edges"):
        graph_nodes = getattr(graph_store, "nodes", {}) or {}
        graph_out_edges = getattr(graph_store, "out_edges", {}) or {}
        edge_types = Counter(
            str(edge.get("type") or "RELATED")
            for edges in graph_out_edges.values()
            for edge in edges
        )
        node_types = Counter(str(node.get("type") or "unknown") for node in graph_nodes.values())
        return {
            "node_count": len(graph_nodes),
            "edge_count": sum(len(edges) for edges in graph_out_edges.values()),
            "node_types": [{"name": name, "count": count} for name, count in node_types.most_common()],
            "edge_types": [{"name": name, "count": count} for name, count in edge_types.most_common()],
        }

    if hasattr(graph_store, "driver"):
        try:
            with graph_store.driver.session(database=graph_store.config.neo4j_database) as session:
                node_total = session.run("MATCH (n:LegalNode) RETURN count(n) AS count").single()["count"]
                edge_total = session.run("MATCH (:LegalNode)-[r]->(:LegalNode) RETURN count(r) AS count").single()["count"]
                node_type_rows = session.run(
                    """
                    MATCH (n:LegalNode)
                    RETURN coalesce(n.type, 'unknown') AS name, count(n) AS count
                    ORDER BY count DESC
                    """
                )
                edge_type_rows = session.run(
                    """
                    MATCH (:LegalNode)-[r]->(:LegalNode)
                    RETURN type(r) AS name, count(r) AS count
                    ORDER BY count DESC
                    """
                )
                return {
                    "node_count": int(node_total or 0),
                    "edge_count": int(edge_total or 0),
                    "node_types": [{"name": row["name"], "count": row["count"]} for row in node_type_rows],
                    "edge_types": [{"name": row["name"], "count": row["count"]} for row in edge_type_rows],
                }
        except Exception as exc:
            logger.warning("Could not read graph stats: %s", exc)

    return {"node_count": 0, "edge_count": 0, "node_types": [], "edge_types": []}

def _graph_trace(rag: LegalGraphRAG, docs: List[Dict[str, Any]], *, depth: int = 3, max_nodes: int = 70) -> Dict[str, Any]:
    graph_store = rag.graph_store
    if not docs or not getattr(graph_store, "loaded", False):
        return {"seed_node_ids": [], "nodes": [], "edges": [], "relation_counts": {}, "node_type_counts": {}}

    seed_ids = graph_store.lookup_record_nodes(docs)
    if not seed_ids:
        seed_ids = [
            str(doc.get("source_chunk_id") or doc.get("id"))
            for doc in docs
            if doc.get("source_chunk_id") or doc.get("id")
        ]
    seed_ids = list(dict.fromkeys(seed_ids))[:12]
    expanded = graph_store.expand(seed_ids, depth=max(1, min(int(depth), 5)), max_nodes=max_nodes)
    if hasattr(graph_store, "same_ref_context"):
        expanded.extend(graph_store.same_ref_context(seed_ids, max_nodes=max_nodes // 2, per_seed=16))

    by_id: Dict[str, Dict[str, Any]] = {}
    for node in expanded:
        node_id = node.get("id")
        if node_id and node_id not in by_id:
            by_id[node_id] = node
    node_ids = set(by_id)

    edges: List[Dict[str, Any]] = []
    seen_edges = set()
    if hasattr(graph_store, "out_edges"):
        for source_id in node_ids:
            for edge in graph_store.out_edges.get(source_id, []):
                targets: List[str] = []
                target = edge.get("target")
                if target in node_ids:
                    targets.append(target)
                elif edge.get("target_ref") and hasattr(graph_store, "_resolve_target_ref"):
                    targets.extend(
                        target_id
                        for target_id in graph_store._resolve_target_ref(str(edge["target_ref"]), limit=6)
                        if target_id in node_ids
                    )
                for target_id in targets:
                    key = (source_id, target_id, edge.get("type"), edge.get("raw"))
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    edges.append({
                        "source": source_id,
                        "target": target_id,
                        "type": edge.get("type") or "RELATED",
                        "raw": edge.get("raw") or "",
                        "target_ref": edge.get("target_ref") or "",
                    })
    elif hasattr(graph_store, "driver"):
        edges.extend(_neo4j_edges_between(graph_store, node_ids, limit=140))

    node_type_counts = Counter(str(node.get("type") or "unknown") for node in by_id.values())
    relation_counts = Counter(str(edge.get("type") or "RELATED") for edge in edges)
    return {
        "seed_node_ids": seed_ids,
        "nodes": [_node_payload(node) for node in by_id.values()],
        "edges": edges[:140],
        "relation_counts": dict(relation_counts),
        "node_type_counts": dict(node_type_counts),
        "backend": type(graph_store).__name__,
    }

def _matches_source_filters(
    record: Dict[str, Any],
    *,
    document: str = "",
    article: str = "",
    clause: str = "",
    point: str = "",
    modality: str = "",
    has_penalty: bool = False,
    has_sign: bool = False,
    has_table: bool = False,
    has_procedure: bool = False,
) -> bool:
    ref = record.get("legal_reference") or {}
    meta = record.get("rag_metadata") or {}
    doc = str(record.get("doc_name") or ref.get("document") or meta.get("doc") or "")
    if document and doc != document:
        return False
    if article and ascii_lower(str(ref.get("article") or "")) != ascii_lower(article):
        return False
    if clause and ascii_lower(str(ref.get("clause") or "")) != ascii_lower(clause):
        return False
    if point and ascii_lower(str(ref.get("point") or "")) != ascii_lower(point):
        return False
    if modality and str(record.get("rag_modality") or "text") != modality:
        return False
    if has_penalty and not (meta.get("has_penalty") or record.get("penalties")):
        return False
    if has_sign and not meta.get("has_sign"):
        return False
    if has_table and not meta.get("has_table"):
        return False
    if has_procedure and not meta.get("has_procedure"):
        return False
    return True

def _dedupe_records(records: List[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        key = record.get("source_chunk_id") or record.get("id") or json.dumps(record.get("legal_reference") or {}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
        if len(deduped) >= limit:
            break
    return deduped

def _claim_tokens(text: str) -> set[str]:
    stopwords = {
        "theo", "cua", "va", "hoac", "neu", "thi", "la", "co", "duoc", "khong",
        "trong", "voi", "cho", "nguoi", "dieu", "khien", "phuong", "tien",
    }
    return {tok for tok in re.findall(r"[a-z0-9đ]+", ascii_lower(text)) if len(tok) >= 3 and tok not in stopwords}

def _extract_claims(answer: str, *, limit: int = 24) -> List[str]:
    claims: List[str] = []
    for raw_line in (answer or "").splitlines():
        line = re.sub(r"^\s*(?:[-*+]|\d+[\.)])\s*", "", raw_line).strip()
        line = re.sub(r"^\|?[-:\s|]+$", "", line).strip()
        if not line or len(line) < 28:
            continue
        if line.startswith("#") or set(line) <= {"|", "-", " "}:
            continue
        if "|" in line and len(line.split("|")) >= 3:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            for cell in cells:
                if len(cell) >= 28:
                    claims.append(cell)
        else:
            claims.append(line)
        if len(claims) >= limit:
            break
    return list(dict.fromkeys(claims))[:limit]

def _verify_claims(answer: str, records: List[Dict[str, Any]], *, claim_limit: int = 24) -> Dict[str, Any]:
    claims = _extract_claims(answer, limit=claim_limit)
    prepared = [(record, _claim_tokens(source_text(record))) for record in records]
    out = []
    for claim in claims:
        tokens = _claim_tokens(claim)
        ranked = []
        for record, source_tokens in prepared:
            if not tokens or not source_tokens:
                continue
            overlap = tokens & source_tokens
            score = len(overlap) / max(1, len(tokens))
            if score <= 0:
                continue
            ranked.append((score, len(overlap), record))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score = ranked[0][0] if ranked else 0.0
        if best_score >= 0.24:
            status = "supported"
        elif best_score >= 0.12:
            status = "weak"
        else:
            status = "needs_review"
        out.append({
            "claim": claim,
            "status": status,
            "score": round(best_score, 3),
            "supports": [_record_payload(record, text_limit=700) for _score, _overlap, record in ranked[:3]],
        })
    return {
        "claim_count": len(out),
        "supported_count": sum(1 for item in out if item["status"] == "supported"),
        "weak_count": sum(1 for item in out if item["status"] == "weak"),
        "needs_review_count": sum(1 for item in out if item["status"] == "needs_review"),
        "claims": out,
    }

def _auto_claim_verification(answer: str, records: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not _env_bool("RAG_AUTO_VERIFY_CLAIMS", True):
        return None
    if not answer or not records:
        return None
    claim_limit = _env_int("RAG_AUTO_VERIFY_MAX_CLAIMS", 18, minimum=4, maximum=40)
    try:
        return _verify_claims(answer, records, claim_limit=claim_limit)
    except Exception as exc:
        logger.warning("Automatic claim verification failed: %s", exc)
        return None

def _answer_trace(
    *,
    query: str,
    result: Dict[str, Any],
    analysis: Dict[str, Any] | None,
    records: List[Dict[str, Any]],
    verification: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    if not _env_bool("RAG_INCLUDE_ANSWER_TRACE", True):
        return None

    analysis = analysis or {}
    metadata = result.get("metadata") or {}
    slot_results = metadata.get("slot_results") or result.get("sequential_results") or []
    slots = result.get("slots") or metadata.get("slots") or analysis.get("evidence_slots") or (analysis.get("plan") or {}).get("subquestions") or []
    facets = analysis.get("facets") or []

    coverage: List[Dict[str, Any]] = []
    if slot_results:
        for item in slot_results[:18]:
            slot = item.get("slot") or {}
            coverage.append({
                "slot_id": slot.get("id") or "",
                "facet": slot.get("facet") or "general",
                "query": slot.get("query") or "",
                "reason": slot.get("reason") or "",
                "status": item.get("status") or "unknown",
                "record_count": int(item.get("record_count") or 0),
                "image_count": len(item.get("images") or []),
                "error": item.get("error") or "",
            })
    else:
        by_facet = Counter(str(record.get("retrieval_slot_facet") or record.get("rag_modality") or "general") for record in records)
        for idx, slot in enumerate(slots[:18], start=1):
            facet = str(slot.get("facet") or "general")
            count = int(by_facet.get(facet, 0))
            if count:
                status = "hit"
            elif records:
                count = len(records)
                status = "retrieved_direct"
            else:
                status = "not_traced"
            coverage.append({
                "slot_id": slot.get("id") or f"slot_{idx}",
                "facet": facet,
                "query": slot.get("query") or "",
                "reason": slot.get("reason") or "",
                "status": status,
                "record_count": count,
                "image_count": 0,
                "error": "",
            })

    references = [format_reference(record) for record in records[:30]]
    references = list(dict.fromkeys(ref for ref in references if ref))
    image_count = len(_context_images(records, limit=_api_image_limit()))
    verification_summary = None
    if verification:
        verification_summary = {
            "claim_count": verification.get("claim_count", 0),
            "supported_count": verification.get("supported_count", 0),
            "weak_count": verification.get("weak_count", 0),
            "needs_review_count": verification.get("needs_review_count", 0),
        }

    miss_count = sum(1 for item in coverage if item.get("status") in {"miss", "error", "not_traced"})
    steps = [
        {
            "name": "Phân tích câu hỏi",
            "summary": (
                f"Nhận diện {len(facets) or 1} nhóm vấn đề: {', '.join(facets) if facets else analysis.get('intent', 'general')}."
            ),
        },
        {
            "name": "Tách nhánh truy vấn",
            "summary": f"Tạo {len(slots) or len(coverage) or 1} câu hỏi con bắt buộc/ưu tiên để truy xuất riêng.",
        },
        {
            "name": "Truy xuất căn cứ",
            "summary": f"Thu được {len(records)} nguồn, {len(references)} căn cứ pháp lý duy nhất và {image_count} ảnh nguồn.",
        },
        {
            "name": "Tổng hợp và kiểm chứng",
            "summary": (
                f"Kiểm chứng lexical {verification_summary['claim_count']} kết luận; "
                f"{verification_summary['supported_count']} mạnh, {verification_summary['weak_count']} yếu, "
                f"{verification_summary['needs_review_count']} cần rà soát."
                if verification_summary else "Đã tổng hợp câu trả lời từ các nguồn đã retrieve."
            ),
        },
    ]
    return {
        "query": query,
        "route": metadata.get("route") or ("sequential" if metadata.get("sequential") else "direct"),
        "sequential": bool(metadata.get("sequential")),
        "plan_source": metadata.get("plan_source") or (analysis.get("plan") or {}).get("plan_source") or analysis.get("decomposition_source"),
        "difficulty": analysis.get("difficulty"),
        "difficulty_label": analysis.get("difficulty_label"),
        "facets": facets,
        "slot_count": len(slots) or len(coverage),
        "retrieved_context_count": len(records),
        "reference_count": len(references),
        "image_count": image_count,
        "coverage": coverage,
        "missing_or_weak_branch_count": miss_count,
        "verification": verification_summary,
        "verification_details": (verification or {}).get("claims", [])[:8] if verification else [],
        "steps": steps,
    }

# --- API Endpoints ---
@app.get("/health")
async def health():
    return {"status": "ok", "rag_loaded": get_rag.cache_info().currsize > 0}

@app.get("/system/status")
async def system_status():
    try:
        rag = get_rag()
        vector_store = rag.vector_store
        embedder = getattr(vector_store, "embedder", None)
        records = getattr(vector_store, "records", []) or []
        graph_store = rag.graph_store
        docs = Counter(
            str(record.get("doc_name") or (record.get("legal_reference") or {}).get("document") or "Không rõ")
            for record in records
        )
        modalities = Counter(str(record.get("rag_modality") or "text") for record in records)
        graph_stats = _graph_store_stats(graph_store)
        graph_path = Path(getattr(graph_store, "graph_path", "data/graph/legal_graph.json"))
        return {
            "status": "ok",
            "api_version": app.version,
            "rag_loaded": get_rag.cache_info().currsize > 0,
            "configured_vector_backend": getattr(getattr(rag, "config", None), "vector_backend", ""),
            "vector_backend": type(vector_store).__name__,
            "using_qdrant": type(vector_store).__name__ == "QdrantLegalVectorStore",
            "graph_backend": type(graph_store).__name__,
            "embedding_backend": os.getenv(
                "RAG_EMBEDDING_BACKEND",
                getattr(getattr(rag, "config", None), "embedding_backend", "openvino"),
            ),
            "embedding_runtime": type(embedder).__name__ if embedder is not None else "",
            "using_openvino": type(embedder).__name__ == "OpenVINOEmbedder",
            "embedding_model": getattr(
                vector_store,
                "embedding_model_name",
                getattr(getattr(rag, "config", None), "embedding_model", ""),
            ),
            "configured_embedding_dimension": getattr(getattr(rag, "config", None), "embedding_dimension", 768),
            "embedding_dimension": getattr(vector_store, "dimension", None),
            "openvino_device": os.getenv(
                "RAG_OPENVINO_DEVICE",
                getattr(getattr(rag, "config", None), "openvino_device", "CPU"),
            ),
            "openvino_model_dir": str(
                getattr(
                    embedder,
                    "model_dir",
                    getattr(getattr(rag, "config", None), "openvino_model_dir", ""),
                )
            ),
            "vector_record_count": len(records),
            "documents": [{"name": name, "count": count} for name, count in docs.most_common()],
            "modalities": [{"name": name, "count": count} for name, count in modalities.most_common()],
            "graph": {
                "loaded": bool(getattr(graph_store, "loaded", False)),
                "path": str(graph_path),
                "modified_time": graph_path.stat().st_mtime if graph_path.exists() else None,
                **graph_stats,
            },
            "qdrant_collection": getattr(getattr(rag, "config", None), "qdrant_collection", ""),
        }
    except Exception as e:
        logger.exception("Error in /system/status")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources/search")
async def sources_search(
    q: str = Query("", max_length=800),
    document: str = Query(""),
    article: str = Query(""),
    clause: str = Query(""),
    point: str = Query(""),
    modality: str = Query(""),
    has_penalty: bool = Query(False),
    has_sign: bool = Query(False),
    has_table: bool = Query(False),
    has_procedure: bool = Query(False),
    limit: int = Query(30, ge=1, le=100),
):
    try:
        rag = get_rag()
        filters = {
            "documents": [document] if document else [],
            "modalities": [modality] if modality else [],
            "has_penalty": has_penalty,
            "has_sign": has_sign,
            "has_table": has_table,
            "has_procedure": has_procedure,
        }
        if q.strip():
            candidates = rag.vector_store.search(q.strip(), top_k=max(limit * 4, 80), filters=filters)
        else:
            candidates = [dict(record) for record in getattr(rag.vector_store, "records", []) or []]
        filtered = [
            record
            for record in candidates
            if _matches_source_filters(
                record,
                document=document,
                article=article,
                clause=clause,
                point=point,
                modality=modality,
                has_penalty=has_penalty,
                has_sign=has_sign,
                has_table=has_table,
                has_procedure=has_procedure,
            )
        ]
        records = _dedupe_records(filtered, limit=limit)
        return {
            "query": q,
            "count": len(records),
            "results": [_record_payload(record) for record in records],
        }
    except Exception as e:
        logger.exception("Error in /sources/search")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/trace")
async def graph_trace(
    query: str = Query("", max_length=800),
    source_chunk_ids: str = Query(""),
    depth: int = Query(3, ge=1, le=5),
    limit: int = Query(70, ge=10, le=180),
):
    try:
        rag = get_rag()
        ids = [item.strip() for item in re.split(r"[,;\n]+", source_chunk_ids or "") if item.strip()]
        if ids:
            records = rag.vector_store.by_source_chunk_ids(ids)
            known_ids = {record.get("source_chunk_id") or record.get("id") for record in records}
            records.extend({"source_chunk_id": source_id, "id": source_id} for source_id in ids if source_id not in known_ids)
        elif query.strip():
            records = rag.retrieve(query.strip(), top_k=8, expand_depth=1)
        else:
            records = []
        return {
            "query": query,
            "source_chunk_ids": ids,
            "seeds": [_record_payload(record, text_limit=500) for record in records[:12]],
            "trace": _graph_trace(rag, records, depth=depth, max_nodes=limit),
        }
    except Exception as e:
        logger.exception("Error in /graph/trace")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/verify")
async def chat_verify(answer: str = Form(...), source_chunk_ids: str = Form("")):
    try:
        rag = get_rag()
        ids = []
        try:
            parsed = json.loads(source_chunk_ids or "[]")
            if isinstance(parsed, list):
                ids = [str(item) for item in parsed if item]
        except Exception:
            ids = [item.strip() for item in re.split(r"[,;\n]+", source_chunk_ids or "") if item.strip()]
        records = rag.vector_store.by_source_chunk_ids(ids) if ids else []
        return _verify_claims(answer, records)
    except Exception as e:
        logger.exception("Error in /chat/verify")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/analyze")
async def chat_analyze(query: str = Form(...), history: str = Form("[]")):
    """Analyzes query complexity and identified slots."""
    try:
        rag = get_rag()
        try:
            chat_history = json.loads(history)
        except Exception:
            chat_history = []

        prepared = _prepare_query_for_chat(rag.client, query, chat_history)
        search_query = prepared.effective_query
        analysis = rag.analyze_query(search_query)
        analysis = _attach_query_preprocessing(analysis, prepared)
        return {
            "query": query,
            "condensed_query": search_query if search_query != query else None,
            "query_preprocessing": prepared.public_payload(),
            "analysis": analysis
        }
    except Exception as e:
        logger.exception("Error in /chat/analyze")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat/text')
async def chat_text(query: str = Form(...), history: str = Form("[]")):
    try:
        rag = get_rag()
        try:
            chat_history = json.loads(history)
        except Exception:
            chat_history = []

        prepared = _prepare_query_for_chat(rag.client, query, chat_history)
        search_query = prepared.effective_query
        if prepared.was_preprocessed:
            logger.info("Prepared Search Query: %s", search_query)

        deadline = _env_int("RAG_CHAT_TEXT_DEADLINE_SECONDS", 120, minimum=20, maximum=300)
        try:
            result = await asyncio.wait_for(asyncio.to_thread(rag.query_adaptive, search_query), timeout=deadline)
        except asyncio.TimeoutError:
            logger.warning("/chat/text exceeded %ss; returning direct extractive fallback", deadline)
            result = await asyncio.to_thread(_timeout_fallback_result, rag, search_query)
        ans, docs = result["answer"], result["contexts"]
        analysis = result.get("query_analysis") or rag.analyze_query(search_query)
        analysis = _attach_query_preprocessing(analysis, prepared)
        extra = result.get("metadata") or {}
        extra["query_preprocessing"] = prepared.public_payload()

        images = _context_images(docs, limit=_api_image_limit())
        verification = _auto_claim_verification(ans, docs)
        answer_trace = _answer_trace(
            query=search_query,
            result=result,
            analysis=analysis,
            records=docs,
            verification=verification,
        )
        return {
            "answer": ans,
            "condensed_query": search_query if search_query != query else None,
            "query_preprocessing": prepared.public_payload(),
            "query_analysis": analysis,
            "images": images,
            "reference_images": images,
            "references": _references(docs),
            "answer_trace": answer_trace,
            "claim_verification": verification,
            "graph_trace": _maybe_graph_trace(rag, docs),
            "metadata": extra
        }
    except Exception as e:
        logger.exception("Error in /chat/text")
        raise HTTPException(status_code=500, detail=str(e))

def _condense_query(client: Any, current_query: str, chat_history: List[Dict]) -> str:
    """Rewrite query based on history."""
    if not chat_history: return current_query
    history_text = "\n".join([f"{m.get('role', 'user').upper()}: {m.get('content', '')[:300]}" for m in chat_history[-3:]])
    prompt = (
        "Bạn là hệ thống tóm tắt ngữ cảnh cho Luật Giao Thông.\n"
        "Viết lại câu hỏi mới nhất thành một câu hỏi ĐỘC LẬP, ĐẦY ĐỦ NGỮ CẢNH.\n"
        f"Lịch sử:\n{history_text}\n"
        f"Câu hỏi mới: {current_query}\n"
        "Viết lại:"
    )
    try:
        res, _model = generate_content_with_fallback(
            client,
            contents=[prompt],
            env_names=("RAG_CONDENSE_MODEL", "RAG_PLANNER_MODEL"),
            task="condense",
            logger=logger,
            label="Condense query",
        )
        return (res.text or current_query).strip()
    except Exception: return current_query

@app.post('/chat/sign')
async def chat_sign(query: str = Form(...)):
    try:
        rag = get_rag()
        docs = rag.retrieve_sign(query, top_k=8)
        ans = rag.generate_answer(query, docs)
        images = _context_images(docs, limit=_api_image_limit())
        analysis = rag.analyze_query(query)
        result = {
            "slots": (analysis.get("evidence_slots") or []),
            "metadata": {"route": "sign", "sequential": False, "plan_source": (analysis.get("plan") or {}).get("plan_source")},
        }
        verification = _auto_claim_verification(ans, docs)
        return {
            "answer": ans,
            "query_analysis": analysis,
            "reference_images": images,
            "references": _references(docs),
            "answer_trace": _answer_trace(query=query, result=result, analysis=analysis, records=docs, verification=verification),
            "claim_verification": verification,
            "graph_trace": _maybe_graph_trace(rag, docs),
        }
    except Exception as e:
        logger.exception("Error in /chat/sign")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat/table')
async def chat_table(query: str = Form(...)):
    try:
        rag = get_rag()
        docs = rag.retrieve_table(query, top_k=8)
        ans = rag.generate_answer(query, docs)
        images = _context_images(docs, limit=_api_image_limit())
        analysis = rag.analyze_query(query)
        result = {
            "slots": (analysis.get("evidence_slots") or []),
            "metadata": {"route": "table", "sequential": False, "plan_source": (analysis.get("plan") or {}).get("plan_source")},
        }
        verification = _auto_claim_verification(ans, docs)
        return {
            "answer": ans,
            "query_analysis": analysis,
            "table_images": images,
            "references": _references(docs),
            "answer_trace": _answer_trace(query=query, result=result, analysis=analysis, records=docs, verification=verification),
            "claim_verification": verification,
            "graph_trace": _maybe_graph_trace(rag, docs),
        }
    except Exception as e:
        logger.exception("Error in /chat/table")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat/image')
async def chat_image(image: UploadFile = File(...), query: str = Form("")):
    """Multimodal sign identification."""
    try:
        if not client: raise HTTPException(status_code=500, detail="Gemini client not initialized.")
        image_bytes = await image.read()
        user_img = Image.open(io.BytesIO(image_bytes))
        
        vision_prompt = (
            "Bạn là bộ nhận diện biển báo giao thông Việt Nam theo QCVN 41:2024.\n"
            "Hãy quan sát ảnh thật kỹ: hình dạng, màu nền, viền, ký hiệu ở giữa, chữ/số, mũi tên, phương tiện, người, công trường, trẻ em, đèn tín hiệu.\n"
            "Không được đoán mã nếu ký hiệu không rõ. Nếu không chắc, để candidate_codes rỗng và mô tả đặc điểm nhìn thấy.\n"
            "Chỉ trả về JSON object, không markdown, schema:\n"
            "{"
            "\"is_traffic_sign\": true,"
            "\"candidate_codes\": [\"P.102\"],"
            "\"alternatives\": [{\"code\":\"W.225\",\"reason\":\"...\"}],"
            "\"confidence\": 0.0,"
            "\"shape\": \"tròn/tam giác/chữ nhật/bát giác/khác\","
            "\"dominant_colors\": [\"đỏ\",\"vàng\"],"
            "\"symbol\": \"mô tả ký hiệu chính\","
            "\"text\": \"chữ/số nhìn thấy nếu có\","
            "\"sign_group\": \"cấm|nguy hiểm/cảnh báo|hiệu lệnh|chỉ dẫn|phụ|không rõ\","
            "\"raw_description\": \"mô tả ngắn, trung tính theo ảnh\""
            "}."
        )
        
        res, _model = generate_content_with_fallback(
            client,
            contents=[vision_prompt, user_img],
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=2048),
            env_names=("RAG_VISION_MODEL",),
            vision=True,
            task="vision",
            logger=logger,
            label="Vision sign recognition",
        )
        vision = _parse_vision_json(res.text or "")
        rag = get_rag()
        trusted_codes = _trusted_vision_codes(rag, vision)

        if not vision.get("is_traffic_sign", True) and float(vision.get("confidence", 0)) < 0.4:
            return {"answer": "Không nhận diện được biển báo giao thông trong ảnh.", "references": []}

        final_query = _sign_image_query(vision, trusted_codes, query)
        result = rag.query_adaptive(final_query)
        docs = result["contexts"]
        if trusted_codes:
            exact_sign_docs = rag.retriever.sign_catalog.records_for_codes(trusted_codes, per_code=8)
            merged_docs: List[Dict[str, Any]] = []
            seen = set()
            for doc in [*exact_sign_docs, *docs]:
                key = doc.get("source_chunk_id") or doc.get("id") or json.dumps(doc.get("legal_reference") or {}, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                merged_docs.append(doc)
            docs = merged_docs[:80]
            ans = rag.generate_answer(final_query, docs)
        else:
            ans = result["answer"]
        images = _context_images(docs, limit=_api_image_limit())
        analysis = result.get("query_analysis") or {}
        verification = _auto_claim_verification(ans, docs)
        return {
            "answer": ans,
            "vision": {**vision, "trusted_codes": trusted_codes},
            "query_analysis": analysis,
            "metadata": result.get("metadata"),
            "reference_images": images,
            "references": _references(docs),
            "answer_trace": _answer_trace(query=final_query, result=result, analysis=analysis, records=docs, verification=verification),
            "claim_verification": verification,
            "graph_trace": _maybe_graph_trace(rag, docs),
        }
    except Exception as e:
        logger.exception("Error in /chat/image")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
