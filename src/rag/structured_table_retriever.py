import re
import unicodedata
from typing import Any


def _ascii_lower(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", without_marks.lower()).strip()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", _ascii_lower(text)))


def _norm(text: str) -> str:
    return _ascii_lower(str(text or ""))


def _flat_headers(headers: list[Any]) -> list[str]:
    out = []
    for header in headers or []:
        if isinstance(header, list):
            out.append(" ".join(str(cell or "").strip() for cell in header if str(cell or "").strip()))
        else:
            out.append(str(header or "").strip())
    return out


def _headers_and_rows(table: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    rows = [row for row in (table.get("rows") or []) if isinstance(row, list)]
    headers = _flat_headers(table.get("headers") or table.get("columns") or [])
    if headers or len(rows) < 2:
        return headers, rows
    first = [str(cell or "").strip() for cell in rows[0]]
    alpha_cells = sum(1 for cell in first if re.search(r"[A-Za-zÀ-ỹ]", cell))
    if alpha_cells >= max(1, len([c for c in first if c]) // 2):
        return first, rows[1:]
    return [], rows


def _table_text(record: dict[str, Any]) -> str:
    table = record.get("table") if isinstance(record.get("table"), dict) else {}
    headers, rows = _headers_and_rows(table)
    header_text = " | ".join(str(cell or "") for cell in headers if not isinstance(cell, list))
    row_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in rows if isinstance(row, list))
    return "\n".join(
        x
        for x in [
            table.get("caption") or "",
            header_text,
            table.get("text") or "",
            row_text,
            record.get("rag_text") or "",
            record.get("source_body_exact") or "",
        ]
        if x
    )


class StructuredTableRetriever:
    def __init__(self, records: list[dict[str, Any]]):
        self.records = [record for record in records if record.get("rag_modality") == "table" or record.get("table")]
        self.row_items = self._build_row_items()

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        q_tokens = _tokens(query)
        explicit_terms = self._explicit_terms(query)
        scored = self._score_rows(query, q_tokens, explicit_terms)
        for record in self.records:
            text = _table_text(record)
            if not text:
                continue
            text_lower = _norm(text)
            doc_norm = _norm(record.get("doc_name") or (record.get("legal_reference") or {}).get("document") or "")
            score = 0.0
            overlap = q_tokens & _tokens(text)
            score += min(len(overlap), 12) * 0.08
            for term in explicit_terms:
                if _norm(term) in text_lower:
                    score += 0.65
            if "bảng" in (query or "").lower():
                score += 0.25
            if "giay phep lai xe" in text_lower and any(t in _norm(query) for t in ["hang", "a1", "b1", "b2", "giay phep"]):
                score += 0.8
            if "toc do" in text_lower and "toc do" in _norm(query):
                score += 0.6
            if "toc do" in _norm(query) and "toc do" not in text_lower:
                score -= 3.5
            if "cao toc" in _norm(query) and "cao toc" not in text_lower:
                score -= 0.8
            if "toc do" in _norm(query) and ("qcvn" in doc_norm or "thong tu 51" in doc_norm):
                score += 0.5
            if "kich thuoc" in _norm(query) and "bien" in _norm(query):
                if "qcvn" in doc_norm or "thong tu 51" in doc_norm:
                    score += 1.2
                else:
                    score -= 0.5
            if score <= 0:
                continue
            item = dict(record)
            item["rag_modality"] = "table"
            item["retrieval_score"] = max(float(item.get("retrieval_score") or 0), score)
            item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["structured_table_summary"]))
            matched_rows = self._matched_rows(query, item.get("table") or {})
            if matched_rows:
                item["matched_table_rows"] = matched_rows[:8]
                item["rag_text"] = "\n".join(
                    [
                        item.get("rag_text") or "",
                        "Các dòng bảng khớp trực tiếp:",
                        *[" | ".join(str(cell or "") for cell in row) for row in matched_rows[:8]],
                    ]
                ).strip()
            scored.append(item)
        return sorted(scored, key=lambda r: float(r.get("retrieval_score") or 0), reverse=True)[:top_k]

    def _build_row_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for record in self.records:
            table = record.get("table") if isinstance(record.get("table"), dict) else {}
            headers, rows = _headers_and_rows(table)
            caption = " ".join(x for x in [table.get("caption") or "", table.get("text") or ""] if x).strip()
            for idx, row in enumerate(rows):
                if not isinstance(row, list):
                    continue
                row_text = self._row_text(headers, row)
                if not row_text.strip():
                    continue
                items.append(
                    {
                        "record": record,
                        "row_index": idx,
                        "row": row,
                        "headers": headers,
                        "caption": caption,
                        "text": "\n".join(x for x in [caption, row_text] if x),
                    }
                )
        return items

    def _row_text(self, headers: list[Any], row: list[Any]) -> str:
        flat_headers = _flat_headers(headers)
        parts = []
        for idx, cell in enumerate(row):
            cell_text = str(cell or "").strip()
            if not cell_text:
                continue
            header = flat_headers[idx] if idx < len(flat_headers) else ""
            parts.append(f"{header}: {cell_text}" if header else cell_text)
        return " | ".join(parts)

    def _score_rows(self, query: str, q_tokens: set[str], explicit_terms: list[str]) -> list[dict[str, Any]]:
        q_lower = _norm(query)
        query_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", q_lower))
        scored: list[dict[str, Any]] = []
        for item in self.row_items:
            text = item["text"]
            text_lower = _norm(text)
            row_lower = _norm(" ".join(str(cell or "") for cell in item["row"]))
            overlap = q_tokens & _tokens(text)
            score = min(len(overlap), 18) * 0.11
            exact_hits = [term for term in explicit_terms if _norm(term) in text_lower]
            score += len(exact_hits) * 1.15
            if q_lower and q_lower in text_lower:
                score += 2.0
            if any(_norm(term) in row_lower for term in exact_hits):
                score += 0.65
            if "kich thuoc" in q_lower and any(k in text_lower for k in ["kich thuoc", "duong kinh", "chieu cao", "chieu rong"]):
                score += 1.2 # Boosted for dimension queries
            if "hang" in q_lower and any(k in text_lower for k in ["a1", "a2", "b1", "b2", "c1", "d1", "giay phep"]):
                score += 0.85
            if "toc do" in q_lower and "toc do" in text_lower:
                score += 0.75
            if "toc do" in q_lower and "toc do" not in text_lower:
                score -= 4.0
            if "cao toc" in q_lower and "cao toc" not in text_lower:
                score -= 1.0
            row_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", row_lower))
            number_hits = query_numbers & row_numbers
            if number_hits:
                score += min(len(number_hits), 4) * 0.35
            
            # Context preservation logic:
            base = dict(item["record"])
            doc_norm = _norm(base.get("doc_name") or (base.get("legal_reference") or {}).get("document") or "")
            if "kich thuoc" in q_lower and "bien" in q_lower:
                if "qcvn" in doc_norm or "thong tu 51" in doc_norm:
                    score += 1.5
                else:
                    score -= 0.5
            if "toc do" in q_lower and ("qcvn" in doc_norm or "thong tu 51" in doc_norm):
                score += 0.7
            
            if score <= 0.15:
                continue
            
            table = dict(base.get("table") or {})
            base["rag_modality"] = "table"
            base["table"] = table
            base["table_row_index"] = item["row_index"]
            base["matched_table_rows"] = [item["row"]]
            
            # Ensure headers and captions are always present in rag_text
            header_text = " | ".join(str(h) for h in item["headers"] if h)
            base["rag_text"] = "\n".join(
                [
                    f"Bảng: {item['caption']}",
                    f"Tiêu đề cột: {header_text}",
                    f"Dòng khớp trực tiếp #{item['row_index'] + 1}:",
                    self._row_text(item["headers"], item["row"]),
                    f"Ghi chú/Nội dung liên quan: {base.get('rag_text', '')}",
                ]
            ).strip()
            
            base["retrieval_score"] = max(float(base.get("retrieval_score") or 0), score + 0.75)
            base["retrieval_reasons"] = sorted(set(base.get("retrieval_reasons", []) + ["structured_table_row_with_context"]))
            scored.append(base)
        return scored

    def _explicit_terms(self, query: str) -> list[str]:
        terms = re.findall(r"\b(?:DP|IE|P|W|R|I|S|E)?\.?\d{1,3}[a-zđ]?\b", query or "", flags=re.IGNORECASE)
        terms.extend(re.findall(r"\b[A-Z]{1,3}\d{1,2}[a-z]?\b", query or "", flags=re.IGNORECASE))
        terms.extend(re.findall(r"\bHình\s+[A-Z]\.\d+[a-z]?\b", query or "", flags=re.IGNORECASE))
        terms.extend(re.findall(r"\bBảng\s+[A-ZĐ]?\s*\.?\s*\d+[a-z]?\b", query or "", flags=re.IGNORECASE))
        terms.extend(re.findall(r"\bPhụ\s+lục\s+[A-ZĐ]\b", query or "", flags=re.IGNORECASE))
        terms.extend(re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:km/h|m|cm|%)\b", query or "", flags=re.IGNORECASE))
        quoted = re.findall(r"[\"“]([^\"”]+)[\"”]", query or "")
        terms.extend(quoted)
        query_norm = _norm(query)
        for phrase in ["kích thước", "tốc độ", "giấy phép lái xe", "nền đường yếu", "đường cao tốc"]:
            if _norm(phrase) in query_norm:
                terms.append(phrase)
        return list(dict.fromkeys(term.strip() for term in terms if term.strip()))

    def _matched_rows(self, query: str, table: dict[str, Any]) -> list[list[Any]]:
        rows = table.get("rows") or []
        if not rows:
            return []
        q = _norm(query)
        q_tokens = _tokens(query)
        out = []
        for row in rows:
            if not isinstance(row, list):
                continue
            row_text = _norm(" ".join(str(cell or "") for cell in row))
            explicit_hit = any(_norm(term) in row_text for term in self._explicit_terms(query))
            token_hit = len(q_tokens & _tokens(row_text)) >= 2
            if explicit_hit or token_hit:
                out.append(row)
        if not out and any(x in q for x in ["hang", "a1", "b1", "b2", "giay phep"]):
            return rows[:6]
        return out
