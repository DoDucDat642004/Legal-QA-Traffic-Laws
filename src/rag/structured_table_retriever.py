import re
from typing import Any


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", (text or "").lower()))


def _table_text(record: dict[str, Any]) -> str:
    table = record.get("table") if isinstance(record.get("table"), dict) else {}
    rows = table.get("rows") or []
    row_text = "\n".join(" | ".join(str(cell or "") for cell in row) for row in rows if isinstance(row, list))
    return "\n".join(
        x
        for x in [
            table.get("caption") or "",
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

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        q_tokens = _tokens(query)
        explicit_terms = self._explicit_terms(query)
        scored = []
        for record in self.records:
            text = _table_text(record)
            if not text:
                continue
            text_lower = text.lower()
            score = 0.0
            overlap = q_tokens & _tokens(text)
            score += min(len(overlap), 12) * 0.08
            for term in explicit_terms:
                if term.lower() in text_lower:
                    score += 0.65
            if "bảng" in (query or "").lower():
                score += 0.25
            if "giấy phép lái xe" in text_lower and any(t in (query or "").lower() for t in ["hạng", "a1", "b1", "b2", "giấy phép"]):
                score += 0.8
            if "tốc độ" in text_lower and "tốc độ" in (query or "").lower():
                score += 0.6
            if score <= 0:
                continue
            item = dict(record)
            item["rag_modality"] = "table"
            item["retrieval_score"] = max(float(item.get("retrieval_score") or 0), score)
            item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["structured_table"]))
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

    def _explicit_terms(self, query: str) -> list[str]:
        terms = re.findall(r"\b[A-ZĐ]{0,2}\.?\d{1,3}[a-zđ]?\b", query or "", flags=re.IGNORECASE)
        terms.extend(re.findall(r"\b[A-Z]\d\b", query or "", flags=re.IGNORECASE))
        quoted = re.findall(r"[\"“]([^\"”]+)[\"”]", query or "")
        terms.extend(quoted)
        return list(dict.fromkeys(term.strip() for term in terms if term.strip()))

    def _matched_rows(self, query: str, table: dict[str, Any]) -> list[list[Any]]:
        rows = table.get("rows") or []
        if not rows:
            return []
        q = (query or "").lower()
        q_tokens = _tokens(query)
        out = []
        for row in rows:
            if not isinstance(row, list):
                continue
            row_text = " ".join(str(cell or "") for cell in row).lower()
            explicit_hit = any(term.lower() in row_text for term in self._explicit_terms(query))
            token_hit = len(q_tokens & _tokens(row_text)) >= 2
            if explicit_hit or token_hit:
                out.append(row)
        if not out and any(x in q for x in ["hạng", "a1", "b1", "b2", "giấy phép"]):
            return rows[:6]
        return out
