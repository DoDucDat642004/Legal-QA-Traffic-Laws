import hashlib
import re
import logging

logger = logging.getLogger("CoverageValidator")

class CoverageValidator:
    def __init__(self):
        # Precise patterns for hierarchy detection
        self.article_re = re.compile(r'^Điều\s+(\d+[a-z]*)', re.IGNORECASE | re.MULTILINE)
        # Clauses are numbered 1., 2., 3. at start of line
        self.clause_re = re.compile(r'^\s*(\d+)\.\s+[A-ZÀ-Ỹ]', re.MULTILINE)
        # Points are a), b), c) at start of line
        self.point_re = re.compile(r'^\s*([a-zđ])\)\s+[A-ZÀ-Ỹ]', re.MULTILINE)

    def _get_expected_structure(self, text: str) -> dict:
        """Heuristic audit of raw text to estimate expected units."""
        return {
            "articles": set(self.article_re.findall(text)),
            "clauses": set(self.clause_re.findall(text)),
            "points": set(self.point_re.findall(text))
        }

    def _get_extracted_structure(self, records: list) -> dict:
        extracted = {"articles": set(), "clauses": set(), "points": set()}
        garbage_terms = ["không xác định", "n/a", "none", "unknown", "khh", "không biết"]
        
        for rec in records:
            ref = rec.get("legal_reference", {})
            # Quality check: only count as extracted if not garbage
            def is_clean(v):
                v_str = str(v).lower()
                return v and not any(g in v_str for g in garbage_terms)

            if is_clean(ref.get("article")): extracted["articles"].add(str(ref["article"]).upper())
            if is_clean(ref.get("clause")): extracted["clauses"].add(str(ref["clause"]))
            if is_clean(ref.get("point")): extracted["points"].add(str(ref["point"]).lower())
        return extracted

    def _clean_ref_value(self, value) -> str:
        if value is None:
            return ""
        value = str(value).strip()
        garbage_terms = ["không xác định", "n/a", "none", "unknown", "khh", "không biết", "null"]
        if not value or any(g in value.lower() for g in garbage_terms):
            return ""
        return value

    def _coord_key(self, ref: dict) -> str:
        article = self._clean_ref_value(ref.get("article"))
        clause = self._clean_ref_value(ref.get("clause"))
        point = self._clean_ref_value(ref.get("point"))
        if not any([article, clause, point]):
            return ""
        return f"D{article}|K{clause}|P{point}"

    def _get_expected_from_chunks(self, source_chunks: list[dict]) -> dict:
        chunk_ids = set()
        coords = set()
        by_kind = {"article": set(), "clause": set(), "point": set(), "technical": set()}

        for chunk in source_chunks:
            chunk_id = self._clean_ref_value(chunk.get("source_chunk_id"))
            if chunk_id:
                chunk_ids.add(chunk_id)

            coord = self._coord_key({
                "article": chunk.get("article_num"),
                "clause": chunk.get("clause_num"),
                "point": chunk.get("point_key"),
            })
            if coord:
                coords.add(coord)
                kind = chunk.get("kind")
                if kind in by_kind:
                    by_kind[kind].add(coord)

        return {"chunk_ids": chunk_ids, "coords": coords, "by_kind": by_kind}

    def _get_extracted_from_records(self, records: list[dict]) -> dict:
        chunk_ids = set()
        coords = set()
        source_only_ids = set()
        exact_text_issues = []

        for rec in records:
            chunk_id = self._clean_ref_value(rec.get("source_chunk_id"))
            if chunk_id:
                chunk_ids.add(chunk_id)
                if rec.get("record_type") == "source_legal_unit":
                    source_only_ids.add(chunk_id)

            coord = self._coord_key(rec.get("legal_reference") or {})
            if coord:
                coords.add(coord)

            body = rec.get("source_body_exact")
            expected_hash = rec.get("source_text_sha256")
            if body and expected_hash:
                actual_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
                if actual_hash != expected_hash:
                    exact_text_issues.append(rec.get("id") or chunk_id or "unknown")

        return {
            "chunk_ids": chunk_ids,
            "coords": coords,
            "source_only_ids": source_only_ids,
            "exact_text_issues": exact_text_issues,
        }

    def validate(self, extracted_records: list, raw_text: str, source_chunks: list[dict] | None = None) -> dict:
        expected = self._get_expected_structure(raw_text)
        extracted = self._get_extracted_structure(extracted_records)
        
        # Calculate quality signals
        total_records = len(extracted_records)
        garbage_records = 0
        garbage_terms = ["không xác định", "n/a", "none", "unknown", "khh"]
        
        for rec in extracted_records:
            ref_str = str(rec.get("legal_reference", "")).lower()
            content_str = str(rec.get("violation_content", "")).lower()
            if any(g in ref_str for g in garbage_terms) or any(g in content_str for g in garbage_terms):
                garbage_records += 1
        
        garbage_ratio = garbage_records / total_records if total_records > 0 else 0
        
        results = {}
        for level in ["articles", "clauses", "points"]:
            exp = expected[level]
            ext = extracted[level]
            missing = sorted(list(exp - ext))
            score = (len(exp) - len(missing)) / len(exp) if exp else 1.0
            results[level] = {
                "expected_count": len(exp),
                "extracted_count": len(ext),
                "missing": missing,
                "score": score
            }
        
        chunk_report = None
        if source_chunks is not None:
            expected_chunks = self._get_expected_from_chunks(source_chunks)
            extracted_chunks = self._get_extracted_from_records(extracted_records)

            missing_chunk_ids = sorted(expected_chunks["chunk_ids"] - extracted_chunks["chunk_ids"])
            missing_coords = sorted(expected_chunks["coords"] - extracted_chunks["coords"], key=self._natural_sort_key)
            chunk_total = len(expected_chunks["chunk_ids"])
            coord_total = len(expected_chunks["coords"])

            chunk_report = {
                "expected_chunk_count": chunk_total,
                "covered_chunk_count": len(expected_chunks["chunk_ids"] & extracted_chunks["chunk_ids"]),
                "missing_chunk_ids": missing_chunk_ids,
                "chunk_coverage_score": (chunk_total - len(missing_chunk_ids)) / chunk_total if chunk_total else 1.0,
                "expected_coordinate_count": coord_total,
                "covered_coordinate_count": len(expected_chunks["coords"] & extracted_chunks["coords"]),
                "missing_coordinates": missing_coords,
                "coordinate_coverage_score": (coord_total - len(missing_coords)) / coord_total if coord_total else 1.0,
                "source_only_chunk_count": len(extracted_chunks["source_only_ids"] & expected_chunks["chunk_ids"]),
                "source_only_chunk_ids": sorted(extracted_chunks["source_only_ids"] & expected_chunks["chunk_ids"])[:100],
                "exact_text_hash_issue_count": len(extracted_chunks["exact_text_issues"]),
                "exact_text_hash_issue_ids": extracted_chunks["exact_text_issues"][:100],
                "expected_by_kind": {k: len(v) for k, v in expected_chunks["by_kind"].items()},
            }

        # Overall score penalizes garbage and, when available, missing source chunks/coords.
        base_coverage = min(r["score"] for r in results.values())
        if chunk_report:
            base_coverage = min(
                base_coverage,
                chunk_report["chunk_coverage_score"],
                chunk_report["coordinate_coverage_score"],
            )
        overall_score = base_coverage * (1.0 - garbage_ratio)
        
        status = "ENTERPRISE_READY" if overall_score > 0.95 else "INCOMPLETE"
        if garbage_ratio > 0.1: status = "DIRTY_DATA"
        
        report = {
            "levels": results,
            "source_chunk_coverage": chunk_report,
            "overall_score": overall_score,
            "garbage_ratio": garbage_ratio,
            "status": status,
            "article_coverage": results["articles"]["score"]
        }
        return report

    def _natural_sort_key(self, s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]
