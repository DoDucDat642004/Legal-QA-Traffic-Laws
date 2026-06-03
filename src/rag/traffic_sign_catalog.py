import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.legal_utils import SIGN_CODE_RE, ascii_lower, format_reference, normalize_sign_code, sign_group_from_code, source_text
from src.rag.model_policy import generate_content_with_fallback

# --- Logging Configuration ---
logger = logging.getLogger("TrafficSignCatalog")

# --- Constants & Patterns ---
SIGN_NAME_RE = re.compile(
    r"Biển\s+số\s+"
    r"(?P<code>(?:DP|IE|P|W|R|I|S|E)\s*\.?\s*\d{2,3}[a-zđ]?)"
    r"(?:\s*\([^)]*\))?"
    r"\s*(?::|-)?\s*[\"“]?"
    r"(?P<name>[^;#\n\"”]+)",
    re.IGNORECASE,
)
NORMALIZED_SIGN_CODE_RE = re.compile(r"^(?:DP|IE|[PWRISE])\d{2,3}[A-ZĐ]?$", re.IGNORECASE)

# High-frequency visual feature hints for deterministic mapping
COMMON_SIGN_HINTS: Dict[str, Dict[str, str]] = {
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
    "P122": {"name": "Dừng lại (STOP)", "visual_features": "Biển hình bát giác đều, nền đỏ, chữ STOP màu trắng."},
    "P123A": {"name": "Cấm rẽ trái", "visual_features": "Biển tròn viền đỏ, nền trắng, có mũi tên rẽ trái bị cấm."},
    "P123B": {"name": "Cấm rẽ phải", "visual_features": "Biển tròn viền đỏ, nền trắng, có mũi tên rẽ phải bị cấm."},
    "P124A": {"name": "Cấm quay đầu xe", "visual_features": "Biển tròn viền đỏ, nền trắng, có mũi tên quay đầu bị cấm."},
    "P125": {"name": "Cấm vượt", "visual_features": "Biển tròn viền đỏ, nền trắng, có ký hiệu hai xe."},
    "P127": {"name": "Tốc độ tối đa cho phép", "visual_features": "Biển tròn viền đỏ, nền trắng, có số tốc độ ở giữa."},
    "P130": {"name": "Cấm dừng xe và đỗ xe", "visual_features": "Biển tròn nền xanh, viền đỏ, có hai vạch chéo đỏ."},
    "P131": {"name": "Cấm đỗ xe", "visual_features": "Biển tròn nền xanh, viền đỏ, có một vạch chéo đỏ."},
    "R420": {"name": "Bắt đầu khu đông dân cư", "visual_features": "Biển hình chữ nhật nền xanh, có hình vẽ nhà cửa màu trắng."},
    "W201": {"name": "Chỗ ngoặt nguy hiểm", "visual_features": "Biển cảnh báo hình tam giác, viền đỏ, nền vàng, có ký hiệu đường cong."},
    "W203": {"name": "Đường bị thu hẹp", "visual_features": "Biển cảnh báo hình tam giác, viền đỏ, nền vàng, có ký hiệu đường bị thu hẹp."},
    "W204": {"name": "Đường hai chiều", "visual_features": "Biển cảnh báo hình tam giác, viền đỏ, nền vàng, có hai mũi tên ngược chiều."},
    "W205": {"name": "Đường giao nhau", "visual_features": "Biển cảnh báo hình tam giác, viền đỏ, nền vàng, có ký hiệu giao nhau."},
    "W207": {"name": "Giao nhau với đường không ưu tiên", "visual_features": "Biển cảnh báo hình tam giác, viền đỏ, nền vàng, có sơ đồ nhánh giao nhau."},
    "W208": {"name": "Giao nhau với đường ưu tiên", "visual_features": "Biển cảnh báo hình tam giác, viền đỏ, nền vàng, có ký hiệu giao với đường ưu tiên."},
    "W209": {"name": "Giao nhau có tín hiệu đèn", "visual_features": "Biển cảnh báo hình tam giác, viền đỏ, nền vàng, có hình đèn tín hiệu giao thông."},
    "W224": {"name": "Đường người đi bộ cắt ngang", "visual_features": "Biển cảnh báo hình tam giác, viền đỏ, nền vàng, có hình người đi bộ."},
    "W225": {"name": "Trẻ em", "visual_features": "Biển cảnh báo hình tam giác, viền đỏ, nền vàng, có hình trẻ em."},
    "W227": {"name": "Công trường", "visual_features": "Biển cảnh báo hình tam giác, viền đỏ, nền vàng, có hình người đang thi công."},
}


@dataclass
class TrafficSignEntry:
    """Unified data model for a traffic sign entry."""
    code: str
    normalized_code: str
    name: str = ""
    meaning: str = ""
    visual_features: str = ""
    group: str = "unknown"
    source_chunk_id: str = ""
    legal_reference: Dict[str, Any] = field(default_factory=dict)
    image_path: str = ""
    crop_image_path: str = ""
    source_record: Dict[str, Any] = field(default_factory=dict)
    supporting_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_record(self) -> Dict[str, Any]:
        """Converts the entry back into a RAG-compatible context record."""
        display_code = self.code if "." in self.code else self.normalized_code
        text_parts = [
            f"Mã biển báo: {display_code}",
            f"Tên/ý nghĩa: {self.name or self.meaning}".strip(),
            f"Nhóm biển: {self.group}",
            f"Đặc điểm nhận dạng: {self.visual_features}" if self.visual_features else "",
            f"Ý nghĩa sử dụng: {self.source_record.get('meaning_and_usage', '')}",
            f"Căn cứ: {format_reference(self.source_record)}" if self.source_record else "",
            source_text(self.source_record)[:1500] if self.source_record else "",
        ]
        
        record = dict(self.source_record)
        sign_text = "\n".join(x for x in text_parts if x).strip()
        record.update({
            "rag_modality": "sign",
            "rag_text": sign_text,
            "source_body_exact": sign_text,
            "source_chunk_id": self.source_chunk_id or record.get("source_chunk_id") or record.get("id") or self.normalized_code,
            "legal_reference": self.legal_reference or record.get("legal_reference") or {},
            "image_path": self.crop_image_path or self.image_path,
            "figure": {
                "id": f"catalog_{self.normalized_code}",
                "code": display_code,
                "name": self.name or self.meaning,
                "caption": self.visual_features,
                "image_path": self.crop_image_path or self.image_path,
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
        })
        return record


class TrafficSignCatalog:
    """In-memory knowledge base for traffic signs, indexing images and metadata."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records
        self.asset_map = self._load_sign_assets()
        self.entries: Dict[str, TrafficSignEntry] = {}
        self._build()

    def lookup(self, code: str) -> Optional[TrafficSignEntry]:
        """Direct lookup by sign code (e.g., 'P.102')."""
        normalized = normalize_sign_code(code)
        if not NORMALIZED_SIGN_CODE_RE.fullmatch(normalized):
            return None
        return self.entries.get(normalized)

    def ai_semantic_probe(self, query: str, client: Any) -> List[str]:
        """Leverages LLM to translate natural language descriptions into formal sign codes."""
        if not client: return []
            
        prompt = (
            "Bạn là chuyên gia về Hệ thống báo hiệu đường bộ Việt Nam (QCVN 41:2024).\n"
            "Nhiệm vụ: Phân tích mô tả của người dùng và trả về danh sách các MÃ BIỂN BÁO phù hợp nhất.\n"
            "Chỉ trả về JSON danh sách mã (ví dụ: [\"P.102\", \"P.103a\"]).\n"
            f"Mô tả của người dùng: \"{query}\"\n"
            "JSON mã biển báo:"
        )
        
        try:
            from google.genai import types
            res, _model = generate_content_with_fallback(
                client,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0.0),
                env_names=("RAG_SIGN_PROBE_MODEL",),
                task="sign_probe",
                logger=logger,
                label="Traffic sign semantic probe",
            )
            match = re.search(r"\[.*\]", (res.text or "").replace("\n", ""))
            if match:
                codes = json.loads(match.group(0))
                return [normalize_sign_code(str(c)) for c in codes if c]
        except Exception as e:
            logger.warning("AI Semantic Probe failed: %s", e)
        return []

    def find_codes(self, query: str) -> List[str]:
        """Heuristic search for sign codes based on keywords and visual features."""
        qa = ascii_lower(query)
        codes = [
            code
            for code in [normalize_sign_code(m.group(0)) for m in SIGN_CODE_RE.finditer(query)]
            if NORMALIZED_SIGN_CODE_RE.fullmatch(code)
        ]
        
        feature_maps = {
            "tron vien do": (["P"], 1.0),
            "tron nen do": (["P102"], 1.5),
            "tron nen xanh": (["R"], 1.0),
            "tam giac": (["W"], 0.8),
            "nen vang": (["W"], 0.6),
            "vien do": (["P", "W"], 0.3),
            "canh bao": (["W"], 1.0),
            "tre em": (["W225"], 3.0),
            "hoc sinh": (["W225"], 2.4),
            "nguoi di bo cat ngang": (["W224"], 2.6),
            "nguoi di bo": (["P112", "W224"], 1.0),
            "o to": (["P103A"], 2.6),
            "xe may": (["P104"], 2.6),
            "mo to": (["P104"], 2.2),
            "xe tai": (["P106A"], 2.6),
            "re trai": (["P123A"], 2.4),
            "re phai": (["P123B"], 2.4),
            "quay dau": (["P124A"], 2.4),
            "den tin hieu": (["W209"], 2.2),
            "duong hai chieu": (["W204"], 2.4),
            "duong bi thu hep": (["W203"], 2.4),
            "cong truong": (["W227"], 2.4),
            "nguoc chieu": (["P102"], 2.0),
            "vach ngang": (["P102"], 1.0),
            "bat giac": (["P122"], 2.4),
            "stop": (["P122"], 3.0),
        }

        scores: Dict[str, float] = {}
        for feature, (targets, weight) in feature_maps.items():
            if feature in qa:
                for target in targets:
                    if len(target) == 1:
                        for normalized in self.entries:
                            if normalized.startswith(target):
                                scores[normalized] = scores.get(normalized, 0.0) + weight
                    else:
                        scores[target] = scores.get(target, 0.0) + weight

        for normalized, entry in self.entries.items():
            haystack = ascii_lower(" ".join([entry.code, entry.name, entry.meaning, entry.visual_features]))
            name = ascii_lower(entry.name or entry.meaning or "")
            if name and name in qa:
                scores[normalized] = scores.get(normalized, 0.0) + 3.0
            for phrase in self._important_phrases(haystack):
                if phrase in qa:
                    scores[normalized] = scores.get(normalized, 0.0) + (1.2 if " " in phrase else 0.45)
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        found = [c for c, s in sorted_candidates if s >= 1.2 and c in self.entries and NORMALIZED_SIGN_CODE_RE.fullmatch(c)]
        
        return list(dict.fromkeys(codes + found))[:15]

    def records_for_codes(self, codes: List[str], per_code: int = 4) -> List[Dict[str, Any]]:
        """Fetches detailed records and supporting evidence for a list of codes."""
        out = []
        for code in codes:
            if entry := self.lookup(code):
                out.append(entry.to_record())
                for r in entry.supporting_records[:per_code]:
                    item = dict(r)
                    item["retrieval_score"] = max(float(item.get("retrieval_score") or 0), 2.5)
                    item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["traffic_sign_support"]))
                    out.append(item)
        return out

    def _build(self) -> None:
        """Parses the global record set to populate the catalog."""
        parsed_names = self._parse_names_from_corpus()
        for record in self.records:
            figures = record.get("figures") or ([record.get("figure")] if record.get("figure") else [])
            for fig in figures:
                if not (fig and fig.get("code")): continue
                code = fig.get("code")
                norm = normalize_sign_code(code)
                if not NORMALIZED_SIGN_CODE_RE.fullmatch(norm):
                    continue
                asset = self.asset_map.get(norm) or {}
                common = COMMON_SIGN_HINTS.get(norm) or {}
                raw_name = (fig.get("name") or asset.get("name") or parsed_names.get(norm) or "").strip()
                raw_visual = (fig.get("visual_features") or fig.get("caption") or "").strip()
                
                entry = TrafficSignEntry(
                    code=code,
                    normalized_code=norm,
                    name=self._best_name(raw_name, common.get("name") or ""),
                    meaning=(fig.get("meaning") or fig.get("caption") or parsed_names.get(norm) or common.get("name") or "").strip(),
                    visual_features=self._best_visual(raw_visual, common.get("visual_features") or ""),
                    group=sign_group_from_code(code),
                    source_chunk_id=record.get("source_chunk_id") or record.get("id") or norm,
                    legal_reference=record.get("legal_reference") or {},
                    image_path=fig.get("image_path") or record.get("image_path", ""),
                    crop_image_path=asset.get("image_path", ""),
                    source_record=record
                )
                existing = self.entries.get(norm)
                if existing is None or self._entry_quality(entry) > self._entry_quality(existing):
                    if existing:
                        entry.supporting_records = existing.supporting_records
                    self.entries[norm] = entry
                self.entries[norm].supporting_records.append(record)

    def _parse_names_from_corpus(self) -> Dict[str, str]:
        names: Dict[str, str] = {}
        for record in self.records:
            text = record.get("rag_text") or source_text(record)
            for match in SIGN_NAME_RE.finditer(text or ""):
                code = normalize_sign_code(match.group("code"))
                name = re.sub(r"\s+", " ", (match.group("name") or "")).strip(" .:-")
                if code and name and len(name) <= 140:
                    names.setdefault(code, name)
        return names

    def _entry_quality(self, entry: TrafficSignEntry) -> float:
        score = 0.0
        if entry.name:
            score += 1.0
        if entry.visual_features:
            score += 0.7
        if entry.crop_image_path:
            score += 2.0
        if "sign_assets" in (entry.crop_image_path or ""):
            score += 1.0
        if "qcvn" in ascii_lower(entry.source_record.get("doc_name") or ""):
            score += 0.5
        return score

    def _best_name(self, raw_name: str, common_name: str) -> str:
        raw = re.sub(r"\s+", " ", raw_name or "").strip(" .:-")
        qa = ascii_lower(raw)
        if common_name and (
            not raw
            or len(raw) > 90
            or any(term in qa for term in ["nhu quy dinh", "muc ", "phu luc", "bien so", "ma bien", "co hieu luc"])
            or any(qa.startswith(prefix) for prefix in ["tai ", "khi ", "truong hop ", "neu ", "duoc ", "dung de "])
        ):
            return common_name
        return raw or common_name

    def _best_visual(self, raw_visual: str, common_visual: str) -> str:
        raw = re.sub(r"\s+", " ", raw_visual or "").strip()
        qa = ascii_lower(raw)
        if common_visual and (
            not raw
            or len(raw) > 180
            or any(term in qa for term in ["ma bien/vach", "duoc nhac trong doan", "dung anh trang goc"])
        ):
            return common_visual
        return raw or common_visual

    def _important_phrases(self, text: str) -> List[str]:
        stop = {"bien", "bao", "cam", "cho", "phep", "hinh", "tron", "vien", "nen", "mau", "duong", "canh", "bao"}
        tokens = [token for token in re.findall(r"\w+", text) if len(token) >= 3 and token not in stop]
        phrases: List[str] = []
        for n in (3, 2):
            for idx in range(0, max(0, len(tokens) - n + 1)):
                phrase = " ".join(tokens[idx: idx + n])
                if 5 <= len(phrase) <= 60:
                    phrases.append(phrase)
        phrases.extend(token for token in tokens if len(token) >= 4)
        return list(dict.fromkeys(phrases))[:24]

    def _load_sign_assets(self) -> Dict[str, Dict[str, Any]]:
        """Loads pre-cropped image metadata."""
        assets = {}
        for path in Path("data/processed/sign_assets").glob("*_meta.json"):
            try:
                with path.open("r") as f:
                    data = json.load(f)
                    for a in data:
                        if c := a.get("code"):
                            normalized = normalize_sign_code(c)
                            if NORMALIZED_SIGN_CODE_RE.fullmatch(normalized):
                                assets[normalized] = a
            except Exception: continue
        return assets
