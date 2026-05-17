import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.rag.legal_utils import SIGN_CODE_RE, format_reference, normalize_sign_code, sign_group_from_code, source_text


SIGN_NAME_RE = re.compile(
    r"Biển\s+số\s+"
    r"(?P<code>(?:DP|IE|P|W|R|I|S|E)\s*\.?\s*\d{2,3}[a-zđ]?)"
    r"(?:\s*\([^)]*\))?"
    r"\s*(?::|-)?\s*[\"“]?"
    r"(?P<name>[^;#\n\"”]+)",
    re.IGNORECASE,
)


COMMON_SIGN_HINTS: dict[str, dict[str, str]] = {
    "P101": {
        "name": "Đường cấm",
        "visual_features": "Biển tròn, viền đỏ, nền trắng, không có ký hiệu ở giữa.",
    },
    "P102": {
        "name": "Cấm đi ngược chiều",
        "visual_features": "Biển tròn nền đỏ, có vạch ngang màu trắng ở giữa.",
    },
    "P103A": {"name": "Cấm xe ô tô", "visual_features": "Biển tròn viền đỏ, nền trắng, có hình ô tô màu đen."},
    "P104": {"name": "Cấm xe máy", "visual_features": "Biển tròn viền đỏ, nền trắng, có hình xe máy màu đen."},
    "P106A": {"name": "Cấm xe ô tô tải", "visual_features": "Biển tròn viền đỏ, nền trắng, có hình xe tải."},
    "P112": {"name": "Cấm người đi bộ", "visual_features": "Biển tròn viền đỏ, nền trắng, có hình người đi bộ."},
    "P123A": {"name": "Cấm rẽ trái", "visual_features": "Biển tròn viền đỏ, nền trắng, có mũi tên rẽ trái bị cấm."},
    "P123B": {"name": "Cấm rẽ phải", "visual_features": "Biển tròn viền đỏ, nền trắng, có mũi tên rẽ phải bị cấm."},
    "P124A": {"name": "Cấm quay đầu xe", "visual_features": "Biển tròn viền đỏ, nền trắng, có mũi tên quay đầu bị cấm."},
    "P125": {"name": "Cấm vượt", "visual_features": "Biển tròn viền đỏ, nền trắng, có ký hiệu hai xe."},
    "P127": {"name": "Tốc độ tối đa cho phép", "visual_features": "Biển tròn viền đỏ, nền trắng, có số tốc độ ở giữa."},
    "P130": {"name": "Cấm dừng xe và đỗ xe", "visual_features": "Biển tròn nền xanh, viền đỏ, có hai vạch chéo đỏ."},
    "P131": {"name": "Cấm đỗ xe", "visual_features": "Biển tròn nền xanh, viền đỏ, có một vạch chéo đỏ."},
}


@dataclass
class TrafficSignEntry:
    code: str
    normalized_code: str
    name: str = ""
    meaning: str = ""
    visual_features: str = ""
    group: str = "unknown"
    source_chunk_id: str = ""
    legal_reference: dict[str, Any] = field(default_factory=dict)
    image_path: str = ""
    crop_image_path: str = ""
    source_record: dict[str, Any] = field(default_factory=dict)
    supporting_records: list[dict[str, Any]] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        display_code = self.code if "." in self.code else self.normalized_code
        text_parts = [
            f"Mã biển báo: {display_code}",
            f"Tên/ý nghĩa: {self.name or self.meaning}".strip(),
            f"Nhóm biển: {self.group}",
            f"Đặc điểm nhận dạng: {self.visual_features}" if self.visual_features else "",
            f"Căn cứ liên quan: {format_reference(self.source_record)}" if self.source_record else "",
            source_text(self.source_record)[:1800] if self.source_record else "",
        ]
        record = dict(self.source_record)
        record.update(
            {
                "rag_modality": "sign",
                "rag_text": "\n".join(x for x in text_parts if x).strip(),
                "source_chunk_id": self.source_chunk_id or self.normalized_code,
                "legal_reference": self.legal_reference or self.source_record.get("legal_reference") or {},
                "image_path": self.crop_image_path or self.image_path,
                "figure": {
                    "id": f"catalog_{self.normalized_code}",
                    "code": display_code,
                    "name": self.name or self.meaning,
                    "caption": self.visual_features,
                    "image_path": self.crop_image_path or self.image_path,
                    "source": "traffic_sign_catalog",
                },
                "rag_metadata": {
                    **(record.get("rag_metadata") or {}),
                    "modality": "sign",
                    "has_sign": True,
                    "sign_codes": [self.normalized_code],
                    "image_paths": [x for x in [self.crop_image_path or self.image_path] if x],
                },
                "retrieval_score": max(float(record.get("retrieval_score") or 0), 4.0),
                "retrieval_reasons": sorted(set(record.get("retrieval_reasons", []) + ["traffic_sign_catalog"])),
            }
        )
        return record


class TrafficSignCatalog:
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        self.asset_map = self._load_sign_assets()
        self.entries: dict[str, TrafficSignEntry] = {}
        self._build()

    def lookup(self, code: str) -> TrafficSignEntry | None:
        return self.entries.get(normalize_sign_code(code))

    def find_codes(self, query: str) -> list[str]:
        q = (query or "").lower()
        codes = [normalize_sign_code(match.group(0)) for match in SIGN_CODE_RE.finditer(query or "")]
        for normalized, entry in self.entries.items():
            haystack = " ".join([entry.code, entry.name, entry.meaning, entry.visual_features]).lower()
            if entry.name and len(entry.name.strip()) >= 4 and entry.name.lower() in q:
                codes.append(normalized)
            elif any(token and token in q for token in self._important_tokens(haystack)):
                if normalized in COMMON_SIGN_HINTS:
                    codes.append(normalized)
        return list(dict.fromkeys(codes))

    def records_for_codes(self, codes: list[str], per_code: int = 4) -> list[dict[str, Any]]:
        out = []
        for code in codes:
            entry = self.lookup(code)
            if not entry:
                continue
            out.append(entry.to_record())
            for record in entry.supporting_records[:per_code]:
                item = dict(record)
                item["retrieval_score"] = max(float(item.get("retrieval_score") or 0), 2.4)
                item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["traffic_sign_support"]))
                out.append(item)
        return out

    def _build(self) -> None:
        parsed_names = self._parse_names_from_corpus()
        for record in self.records:
            figure = record.get("figure") if isinstance(record.get("figure"), dict) else None
            code = figure.get("code") if figure else ""
            if not code:
                continue
            normalized = normalize_sign_code(code)
            if not normalized:
                continue
            existing = self.entries.get(normalized)
            name = (figure.get("name") or parsed_names.get(normalized) or COMMON_SIGN_HINTS.get(normalized, {}).get("name") or "").strip()
            visual = COMMON_SIGN_HINTS.get(normalized, {}).get("visual_features") or self._infer_visual_features(normalized, name)
            candidate = TrafficSignEntry(
                code=code,
                normalized_code=normalized,
                name=name,
                meaning=name,
                visual_features=visual,
                group=sign_group_from_code(code),
                source_chunk_id=record.get("source_chunk_id") or record.get("id") or normalized,
                legal_reference=record.get("legal_reference") or {},
                image_path=record.get("image_path") or "",
                crop_image_path=self._asset_crop_path(normalized) or self._crop_path(record.get("image_path") or ""),
                source_record=record,
            )
            if existing is None or self._record_quality(candidate) > self._record_quality(existing):
                if existing:
                    candidate.supporting_records = existing.supporting_records
                self.entries[normalized] = candidate
            self.entries.setdefault(normalized, candidate).supporting_records.append(record)

    def _parse_names_from_corpus(self) -> dict[str, str]:
        names = {}
        for record in self.records:
            if not self._is_qcvn_record(record):
                continue
            text = record.get("rag_text") or source_text(record)
            for match in SIGN_NAME_RE.finditer(text or ""):
                code = normalize_sign_code(match.group("code"))
                name = re.sub(r"\s+", " ", (match.group("name") or "")).strip(" .:-")
                if code and name and len(name) <= 140:
                    names.setdefault(code, name)
        return names

    def _record_quality(self, entry: TrafficSignEntry) -> float:
        text = source_text(entry.source_record).lower()
        score = 0.0
        if entry.name:
            score += 2.0
        if entry.crop_image_path and "sign_assets" in entry.crop_image_path:
            score += 2.0
        if "phụ lục" in text or "hình" in text:
            score += 1.0
        if "điều 22" in text and "khoảng cách mép ngoài" in text:
            score -= 1.0
        if self._is_qcvn_record(entry.source_record):
            score += 0.5
        return score

    def _crop_path(self, image_path: str) -> str:
        return image_path if "sign_assets/" in (image_path or "").replace("\\", "/") else ""

    def _asset_crop_path(self, normalized_code: str) -> str:
        asset = self.asset_map.get(normalized_code) or {}
        return asset.get("image_path") or ""

    def _load_sign_assets(self) -> dict[str, dict[str, Any]]:
        out = {}
        for meta_path in Path("data/processed/sign_assets").glob("*_meta.json"):
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    assets = json.load(f)
            except Exception:
                continue
            if not isinstance(assets, list):
                continue
            for asset in assets:
                if not isinstance(asset, dict) or not asset.get("code"):
                    continue
                normalized = normalize_sign_code(asset.get("code") or "")
                current = out.get(normalized)
                if current is None or self._asset_quality(asset) > self._asset_quality(current):
                    out[normalized] = asset
        return out

    def _asset_quality(self, asset: dict[str, Any]) -> float:
        score = 0.0
        if asset.get("image_path"):
            score += 2.0
        if asset.get("name"):
            score += 1.0
        bbox = asset.get("bbox") or []
        if len(bbox) == 4:
            width = abs(float(bbox[2]) - float(bbox[0]))
            height = abs(float(bbox[3]) - float(bbox[1]))
            if 40 <= width <= 500 and 40 <= height <= 500:
                score += 1.0
            score += 2.0 * min(height / 120.0, 1.0)
        return score

    def _is_qcvn_record(self, record: dict[str, Any]) -> bool:
        doc = (record.get("doc_name") or (record.get("legal_reference") or {}).get("document") or "").lower()
        return "qcvn" in doc or "thông tư 51" in doc

    def _infer_visual_features(self, normalized: str, name: str) -> str:
        group = sign_group_from_code(normalized)
        if group == "prohibition":
            return "Nhóm biển báo cấm, thường có dạng tròn với viền đỏ hoặc nền đỏ/xanh tùy loại cấm."
        if group == "warning":
            return "Nhóm biển cảnh báo nguy hiểm, thường có dạng tam giác viền đỏ, nền vàng."
        if group == "mandatory":
            return "Nhóm biển hiệu lệnh, thường có dạng tròn nền xanh với ký hiệu màu trắng."
        if group == "guide":
            return "Nhóm biển chỉ dẫn, thường có nền xanh hoặc nền phù hợp loại chỉ dẫn."
        return f"Đặc điểm nhận dạng lấy theo hình/caption trong QCVN cho biển {normalized} {name}".strip()

    def _important_tokens(self, text: str) -> list[str]:
        stop = {"biển", "số", "cấm", "cho", "phép", "hình", "tròn", "viền", "nền"}
        return [token for token in re.findall(r"\w+", text) if len(token) >= 5 and token not in stop][:8]
