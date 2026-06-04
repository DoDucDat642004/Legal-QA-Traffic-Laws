"""Canonical source-document metadata used by API and frontend displays."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def _ascii_lower(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", without_marks.lower()).strip()


DOCUMENT_SOURCES: list[dict[str, Any]] = [
    {
        "group": "Nhóm văn bản Luật",
        "name": "Luật Trật tự, an toàn giao thông đường bộ 2024",
        "document_number": "36/2024/QH15",
        "aliases": ["Luật Trật tự ATGT 2024", "Luật Trật tự ATGT 2024 (Tiếp)"],
        "raw_files": ["36-2024-qh15.pdf", "36-2024-qh15_tiep.pdf"],
        "processed_files": ["36-2024-qh15.pdf.extracted.json", "36-2024-qh15_tiep.pdf.extracted.json"],
        "source_url": "https://vanban.chinhphu.vn/?pageid=27160&docid=211194",
    },
    {
        "group": "Nhóm văn bản Luật",
        "name": "Luật Đường bộ 2024",
        "document_number": "35/2024/QH15",
        "aliases": ["Luật Đường bộ 2024"],
        "raw_files": ["35-2024-qh15.pdf"],
        "processed_files": ["35-2024-qh15.pdf.extracted.json"],
        "source_url": "https://vanban.chinhphu.vn/?pageid=27160&docid=211193&classid=1&typegroupid=3",
    },
    {
        "group": "Nhóm Nghị định xử phạt",
        "name": "Nghị định 168/2024/NĐ-CP",
        "document_number": "168/2024/NĐ-CP",
        "aliases": ["Nghị định 168/2024/NĐ-CP"],
        "raw_files": ["168-nd-cp.signed.pdf"],
        "processed_files": ["168-nd-cp.signed.pdf.extracted.json"],
        "source_url": "https://vanban.chinhphu.vn/?pageid=27160&docid=212167",
        "note": "Xử phạt hành chính và trừ điểm giấy phép lái xe.",
    },
    {
        "group": "Nhóm Nghị định xử phạt",
        "name": "Nghị định 336/2025/NĐ-CP",
        "document_number": "336/2025/NĐ-CP",
        "aliases": ["Nghị định 336/2025/NĐ-CP"],
        "raw_files": ["336-2025-nd-cp-22122025-signed-17665482569851736009102.pdf"],
        "processed_files": ["336-2025-nd-cp-22122025-signed-17665482569851736009102.pdf.extracted.json"],
        "source_url": "https://vanban.chinhphu.vn/?pageid=27160&docid=216275",
        "note": "Xử phạt hoạt động đường bộ, hiệu lực từ 01/03/2026.",
    },
    {
        "group": "Nhóm Quy chuẩn và Thông tư",
        "name": "QCVN 41:2024 (Thông tư 51/2024)",
        "document_number": "QCVN 41:2024/BGTVT",
        "aliases": ["QCVN 41:2024 (Thông tư 51/2024)", "QCVN 41:2019/BGTVT"],
        "raw_files": ["51-bgtvt-kem.pdf"],
        "processed_files": ["51-bgtvt-kem.pdf.extracted.json"],
        "source_url": "https://cdn.thuvienphapluat.vn/phap-luat/2022-2/CTNN/quy-chuan-ky-thuat-qcvn-41-2019-bgtvt-bao-hieu-duong-bo.pdf",
        "note": "Corpus nội bộ đang gắn nhãn QCVN 41:2024/Thông tư 51; URL tham chiếu do người dùng cung cấp là QCVN 41:2019.",
    },
    {
        "group": "Nhóm Quy chuẩn và Thông tư",
        "name": "Thông tư 35/2024/TT-BGTVT",
        "document_number": "35/2024/TT-BGTVT",
        "aliases": ["Thông tư 35/2024/TT-BGTVT"],
        "raw_files": ["35-bgtvt.pdf"],
        "processed_files": ["35-bgtvt.pdf.extracted.json"],
        "source_url": "https://vanban.chinhphu.vn/?pageid=27160&docid=211984&classid=1&typegroupid=6",
        "note": "Đào tạo, sát hạch, cấp giấy phép lái xe.",
    },
]


_SOURCE_BY_ALIAS: dict[str, dict[str, Any]] = {}
for item in DOCUMENT_SOURCES:
    for alias in [item["name"], item.get("document_number", ""), *(item.get("aliases") or [])]:
        key = _ascii_lower(alias)
        if key:
            _SOURCE_BY_ALIAS[key] = item


def source_info_for_document(document: str | None) -> dict[str, Any] | None:
    """Return source metadata for a document name from a legal reference."""
    key = _ascii_lower(document or "")
    if not key:
        return None
    if key in _SOURCE_BY_ALIAS:
        return dict(_SOURCE_BY_ALIAS[key])
    for alias_key, item in _SOURCE_BY_ALIAS.items():
        if alias_key and (alias_key in key or key in alias_key):
            return dict(item)
    return None


def source_catalog_payload() -> list[dict[str, Any]]:
    return [dict(item) for item in DOCUMENT_SOURCES]
