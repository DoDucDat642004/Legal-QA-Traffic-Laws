from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Optional, Literal
import re


EMPTY_NUMERIC_VALUES = {"", "n/a", "na", "none", "null", "không", "khong", "không có", "khong co", "-", "—"}


def optional_int_from_llm(v: Any) -> Any:
    """Coerce common LLM placeholders/units for optional integer fields."""
    if v is None or isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if v.is_integer() else None
    if isinstance(v, str):
        value = v.strip()
        if value.lower() in EMPTY_NUMERIC_VALUES:
            return None
        numbers = re.findall(r'\d+(?:[.,]\d{3})*', value)
        if len(numbers) == 1:
            return int(numbers[0].replace(".", "").replace(",", ""))
        return None
    return v


class BaseLegalReference(BaseModel):
    document: str = Field(..., description="Tên tài liệu pháp lý. VD: 'Nghị định 168/2024/NĐ-CP'")
    document_id: Optional[str] = Field(None, description="ID ổn định của tài liệu trong kho dữ liệu")
    document_type: Optional[Literal["law", "decree", "circular", "qcvn", "other"]] = None
    document_number: Optional[str] = Field(None, description="Số/ký hiệu văn bản. VD: 168/2024/NĐ-CP")
    issue_date: Optional[str] = Field(None, description="Ngày ban hành dạng ISO YYYY-MM-DD nếu tách được")
    effective_date: Optional[str] = Field(None, description="Ngày hiệu lực dạng ISO YYYY-MM-DD nếu tách được")
    part: Optional[str] = Field(None, description="Phần (nếu có)")
    chapter: Optional[str] = Field(None, description="Chương (nếu có)")
    section: Optional[str] = Field(None, description="Mục (nếu có)")
    article: Optional[str] = Field(None, description="Số hiệu Điều. VD: '6', '1a'")
    clause: Optional[str] = Field(None, description="Số hiệu Khoản. VD: '3', '2a'")
    point: Optional[str] = Field(None, description="Chữ cái của Điểm. VD: 'a'")
    appendix: Optional[str] = Field(None, description="Phụ lục (nếu có)")
    item: Optional[str] = Field(None, description="Mục trong Phụ lục")
    table: Optional[str] = Field(None, description="ID/số bảng nếu record gắn với bảng")
    figure: Optional[str] = Field(None, description="ID/số hình/biển báo nếu record gắn với hình")
    page_start: Optional[int] = Field(None, description="Trang bắt đầu trong PDF, zero-based")
    page_end: Optional[int] = Field(None, description="Trang kết thúc trong PDF, zero-based")

    @field_validator('page_start', 'page_end', mode='before')
    @classmethod
    def empty_string_to_none(cls, v: Any) -> Any:
        return optional_int_from_llm(v)


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class ExtractionMeta(BaseModel):
    source_file: Optional[str] = None
    extraction_engine: Optional[str] = Field(None, description="fitz, llamaparse, pdfplumber, manual, gemma-4...")
    engine_version: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source_chunk_id: Optional[str] = None
    source_text_sha256: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    ocr_quality_score: Optional[float] = Field(None, ge=0, le=1)
    layout_confidence: Optional[float] = Field(None, ge=0, le=1)
    table_confidence: Optional[float] = Field(None, ge=0, le=1)
    enrichment_status: Optional[Literal["not_started", "success", "partial", "pending_or_failed", "failed"]] = None
    warnings: List[str] = Field(default_factory=list)


class TableAsset(BaseModel):
    id: str
    page: Optional[int] = None
    bbox: Optional[List[float]] = Field(None, description="[x0, y0, x1, y1] in PDF page coordinates")
    rows: List[List[Optional[str]]] = Field(default_factory=list, description="Dữ liệu bảng dạng mảng 2 chiều để LLM truy vấn")
    text: Optional[str] = None
    image_path: Optional[str] = Field(None, description="Đường dẫn đến ảnh crop của riêng bảng này")
    caption: Optional[str] = None
    linked_reference: Optional[BaseLegalReference] = None
    extraction_meta: Optional[ExtractionMeta] = None

class FigureAsset(BaseModel):
    id: str
    code: Optional[str] = Field(None, description="Mã hình/biển. VD: P.101, Hình 3")
    name: Optional[str] = None
    caption: Optional[str] = None
    asset_type: Optional[Literal["traffic_sign", "figure", "table_image", "symbol", "unknown"]] = "unknown"
    sign_group: Optional[Literal["prohibition", "warning", "mandatory", "guide", "supplementary", "highway", "road_marking", "unknown"]] = "unknown"
    doc_name: Optional[str] = None
    page: Optional[int] = None
    bbox: Optional[List[float]] = Field(None, description="[x0, y0, x1, y1] for cropped image")
    caption_bbox: Optional[List[float]] = None
    image_path: Optional[str] = None
    extract_method: Optional[str] = None
    linked_article: Optional[str] = Field(None, description="Số hiệu Điều liên quan đến biển báo này")
    linked_chunk_id: Optional[str] = Field(None, description="ID của chunk chứa mô tả biển báo này")
    linked_reference: Optional[BaseLegalReference] = None
    extraction_meta: Optional[ExtractionMeta] = None


class ChunkMeta(BaseModel):
    chapter_num: Optional[str] = None
    article_num: Optional[str] = None
    clause_num: Optional[str] = None
    point_key: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    kind: Optional[str] = None

class SourceLineageMixin(BaseModel):
    record_type: Optional[str] = None
    doc_name: Optional[str] = None
    source_chunk_id: Optional[str] = None
    content: Optional[str] = None
    source_body_exact: Optional[str] = None
    source_text_sha256: Optional[str] = None
    image_path: Optional[str] = None
    tables: List[TableAsset] = Field(default_factory=list)
    table_refs: List[str] = Field(default_factory=list)
    figures: List[FigureAsset] = Field(default_factory=list)
    figure_refs: List[str] = Field(default_factory=list)
    extraction_meta: Optional[ExtractionMeta] = None
    chunk_meta: Optional[ChunkMeta] = None


class RecordMetadata(BaseModel):
    enrichment_status: Optional[str] = None
    chunk_kind: Optional[str] = None
    domain: Optional[str] = None
    confidence: Optional[float] = None

class ParentUnit(BaseModel):
    kind: str
    num: str
    title: Optional[str] = None

class LegalSourceRecord(SourceLineageMixin):
    id: str
    legal_reference: BaseLegalReference
    parent_hierarchy: List[ParentUnit] = Field(default_factory=list, description="Semantic parent graph (Article -> Chapter...)")
    metadata: Optional[RecordMetadata] = None
    original_text: Optional[str] = None
    violation_content: Optional[str] = None
    meaning_and_usage: Optional[str] = None
    qa_context: Optional[str] = None


class Metadata168(BaseModel):
    subject: Optional[str] = Field(None, description="Đối tượng vi phạm (VD: Người điều khiển xe ô tô)")
    vehicle_type: List[str] = Field(default_factory=list, description="Loại phương tiện áp dụng")
    keyword_tags: List[str] = Field(default_factory=list, description="3-5 từ khóa chính về lỗi")


class MainPenalty168(BaseModel):
    type: Optional[str] = Field(None, description="Loại phạt: 'Phạt tiền', 'Cảnh cáo'...")
    currency: Optional[str] = Field("VND", description="Đơn vị tiền tệ")
    fine_basis: Optional[str] = Field(None, description="Căn cứ áp dụng mức phạt: cá nhân, tổ chức, phương tiện...")
    min_amount_vnd: Optional[int] = Field(None, description="Mức phạt tối thiểu (số nguyên)")
    max_amount_vnd: Optional[int] = Field(None, description="Mức phạt tối đa (số nguyên)")
    description: Optional[str] = Field(None, description="Mô tả nguyên văn mức phạt")
    raw_penalty_text: Optional[str] = Field(None, description="Đoạn nguồn chứa hình phạt")

    @field_validator('min_amount_vnd', 'max_amount_vnd', mode='before')
    @classmethod
    def empty_string_to_none(cls, v: Any) -> Any:
        return optional_int_from_llm(v)


class Penalties168(BaseModel):
    main_penalty: Optional[MainPenalty168] = None
    additional_penalties: List[str] = Field(default_factory=list, description="Hình thức phạt bổ sung")
    point_deduction: Optional[int] = Field(None, description="Số điểm GPLX bị trừ (số nguyên). Nếu không có thì để null")
    license_suspension: Optional[str] = None
    vehicle_impoundment: Optional[str] = None
    remedial_measures: List[str] = Field(default_factory=list)

    @field_validator('point_deduction', mode='before')
    @classmethod
    def empty_string_to_none(cls, v: Any) -> Any:
        return optional_int_from_llm(v)


class DecreeViolationRule168(SourceLineageMixin):
    id: str = Field(..., description="ID duy nhất. VD: 'D6_K3_a'")
    legal_reference: BaseLegalReference
    metadata: Optional[Metadata168] = None
    violation_content: str = Field(..., description="Nội dung hành vi vi phạm")
    penalties: Optional[Penalties168] = None
    qa_context: Optional[str] = Field(
        None,
        description="Câu văn tự nhiên bắt đầu bằng 'Theo Điểm/Khoản...' + đầy đủ hành vi + phạt + trừ điểm"
    )


class Metadata336(BaseModel):
    domain: Optional[str] = Field(None, description="Lĩnh vực vi phạm")
    subject_target: List[str] = Field(default_factory=list, description="Đối tượng bị phạt: Cá nhân, Tổ chức")
    keyword_tags: List[str] = Field(default_factory=list, description="Từ khóa vi phạm")


class MainPenalty336(BaseModel):
    type: Optional[str] = Field(None, description="Loại phạt")
    currency: Optional[str] = Field("VND", description="Đơn vị tiền tệ")
    fine_basis: Optional[str] = Field(None, description="Căn cứ áp dụng mức phạt")
    individual_min_vnd: Optional[int] = Field(None, description="Mức phạt tiền tối thiểu cho CÁ NHÂN (số nguyên).")
    individual_max_vnd: Optional[int] = Field(None, description="Mức phạt tiền tối đa cho CÁ NHÂN (số nguyên).")
    organization_min_vnd: Optional[int] = Field(None, description="Mức phạt tiền tối thiểu cho TỔ CHỨC (số nguyên, thường gấp đôi cá nhân).")
    organization_max_vnd: Optional[int] = Field(None, description="Mức phạt tiền tối đa cho TỔ CHỨC (số nguyên).")
    raw_penalty_text: Optional[str] = Field(None, description="Đoạn nguồn chứa hình phạt")

    @field_validator('individual_min_vnd', 'individual_max_vnd', 'organization_min_vnd', 'organization_max_vnd', mode='before')
    @classmethod
    def empty_string_to_none(cls, v: Any) -> Any:
        if v == "":
            return None
        return v


class Penalties336(BaseModel):
    main_penalty: Optional[MainPenalty336] = None
    additional_penalties: List[str] = Field(default_factory=list, description="Phạt bổ sung")
    remedial_measures: List[str] = Field(default_factory=list, description="Biện pháp khắc phục hậu quả")
    license_suspension: Optional[str] = None
    vehicle_impoundment: Optional[str] = None


class DecreeViolationRule336(SourceLineageMixin):
    id: str
    legal_reference: BaseLegalReference
    metadata: Optional[Metadata336] = None
    violation_content: str
    penalties: Optional[Penalties336] = None
    qa_context: Optional[str] = Field(None, description="Tóm tắt hành vi, mức phạt, khắc phục hậu quả")


class LawMetadata(BaseModel):
    domain: Optional[str] = Field(None, description="Chủ đề chính")
    rule_type: Optional[str] = Field(None, description="Phân loại: Định nghĩa, Hành vi bị nghiêm cấm, Quy tắc bắt buộc, Thẩm quyền...")
    traffic_participant: List[str] = Field(default_factory=list, description="Đối tượng tham gia giao thông áp dụng")
    keyword_tags: List[str] = Field(default_factory=list)


class LawRule(SourceLineageMixin):
    id: str
    legal_reference: BaseLegalReference
    metadata: Optional[LawMetadata] = None
    original_text: str = Field(..., description="Trích xuất nguyên văn, chính xác từng dấu câu")
    qa_context: Optional[str] = Field(None, description="Diễn giải nội dung thành câu tự nhiên bắt đầu bằng 'Theo Khoản... Điều... Luật...'")


class CircularMetadata(BaseModel):
    domain: Optional[str] = None
    rule_type: Optional[str] = Field(None, description="Quy định hồ sơ, Quy trình thủ tục, Thông số định lượng...")
    target_audience: List[str] = Field(default_factory=list, description="Đối tượng áp dụng (VD: Cơ sở đào tạo, Học viên)")
    keyword_tags: List[str] = Field(default_factory=list)


class QuantitativeData(BaseModel):
    total_training_hours: Optional[int] = None
    theory_hours: Optional[int] = None
    practice_hours: Optional[int] = None
    required_distance_km: Optional[int] = None
    dossier_quantity: Optional[str] = None
    submission_method: Optional[str] = None
    processing_time_days: Optional[int] = None
    other_metrics: List[str] = Field(default_factory=list, description="Các thông số định lượng khác (vd: 'Lưu trữ 5 năm', 'Tuổi tối đa 55')")

    @field_validator('total_training_hours', 'theory_hours', 'practice_hours', 'required_distance_km', 'processing_time_days', mode='before')
    @classmethod
    def empty_string_to_none(cls, v: Any) -> Any:
        return optional_int_from_llm(v)


class CircularRule(SourceLineageMixin):
    id: str
    legal_reference: BaseLegalReference
    metadata: Optional[CircularMetadata] = None
    original_text: str
    quantitative_data: Optional[QuantitativeData] = None
    qa_context: Optional[str] = None


class SignInfo(BaseModel):
    sign_code: Optional[str] = None
    sign_name: Optional[str] = None
    sign_type: Optional[str] = None
    sign_group: Optional[str] = None
    asset_ids: List[str] = Field(default_factory=list)


class TechnicalSpecs(BaseModel):
    shape: Optional[str] = None
    colors: Optional[str] = None
    placement_rules: Optional[str] = None
    dimensions: Optional[str] = None
    applicability: Optional[str] = None
    exceptions: List[str] = Field(default_factory=list)


class QCVNMetadata(BaseModel):
    keyword_tags: List[str] = Field(default_factory=list)


class QCVNRule(SourceLineageMixin):
    id: str
    legal_reference: BaseLegalReference
    sign_info: Optional[SignInfo] = None
    metadata: Optional[QCVNMetadata] = None
    meaning_and_usage: str = Field(..., description="Trích xuất nguyên văn ý nghĩa sử dụng")
    technical_specs: Optional[TechnicalSpecs] = None
    caption: Optional[str] = None
    asset_ids: List[str] = Field(default_factory=list)
    qa_context: Optional[str] = None


class Decree168List(BaseModel):
    thought_process: str = Field(..., description="Phân tích logic của AI về các điều khoản này trước khi trích xuất.")
    rules: List[DecreeViolationRule168]


class Decree336List(BaseModel):
    thought_process: str = Field(..., description="Phân tích logic của AI về các điều khoản này trước khi trích xuất.")
    rules: List[DecreeViolationRule336]


class LawRuleList(BaseModel):
    thought_process: str = Field(..., description="Phân tích logic của AI về các điều khoản này trước khi trích xuất.")
    rules: List[LawRule]


class CircularRuleList(BaseModel):
    thought_process: str = Field(..., description="Phân tích logic của AI về các điều khoản này trước khi trích xuất.")
    rules: List[CircularRule]


class QCVNRuleList(BaseModel):
    thought_process: str = Field(..., description="Phân tích logic của AI về các điều khoản này trước khi trích xuất.")
    rules: List[QCVNRule]


class RAGVectorMetadata(BaseModel):
    """Metadata bắt buộc đi kèm mỗi vector document để filter/rerank chính xác."""
    source_chunk_id: str
    doc: str
    document_id: Optional[str] = None
    document_type: Optional[str] = None
    article: Optional[str] = ""
    clause: Optional[str] = ""
    point: Optional[str] = ""
    chapter: Optional[str] = ""
    section: Optional[str] = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    record_id: Optional[str] = None
    record_type: Optional[str] = None
    modality: Literal["text", "table", "figure", "sign", "penalty", "procedure"] = "text"
    has_table: bool = False
    has_figure: bool = False
    has_sign: bool = False
    has_penalty: bool = False
    has_procedure: bool = False
    table_ids: List[str] = Field(default_factory=list)
    figure_ids: List[str] = Field(default_factory=list)
    sign_codes: List[str] = Field(default_factory=list)
    image_paths: List[str] = Field(default_factory=list)
    fine_min_vnd: Optional[int] = None
    fine_max_vnd: Optional[int] = None
    point_deduction: Optional[int] = None
    source_text_sha256: Optional[str] = None


class GraphNodeBase(BaseModel):
    id: str
    type: str
    doc_name: Optional[str] = None
    source_chunk_id: Optional[str] = None


class DocumentNode(GraphNodeBase):
    type: Literal["document"] = "document"
    name: str
    document_type: Optional[str] = None
    document_number: Optional[str] = None


class ChapterNode(GraphNodeBase):
    type: Literal["chapter"] = "chapter"
    num: str
    title: Optional[str] = None


class ArticleNode(GraphNodeBase):
    type: Literal["article"] = "article"
    num: str
    title: Optional[str] = None


class ClauseNode(GraphNodeBase):
    type: Literal["clause"] = "clause"
    num: str
    article: Optional[str] = None


class PointNode(GraphNodeBase):
    type: Literal["point"] = "point"
    num: str
    article: Optional[str] = None
    clause: Optional[str] = None


class TableNode(GraphNodeBase):
    type: Literal["table"] = "table"
    page: Optional[int] = None
    image_path: Optional[str] = None
    text: Optional[str] = None
    bbox: Optional[List[float]] = None


class FigureNode(GraphNodeBase):
    type: Literal["figure"] = "figure"
    code: Optional[str] = None
    name: Optional[str] = None
    page: Optional[int] = None
    image_path: Optional[str] = None
    caption: Optional[str] = None


class SignNode(GraphNodeBase):
    type: Literal["sign"] = "sign"
    code: str
    normalized_code: str
    name: Optional[str] = None
    sign_group: Optional[str] = None
    meaning: Optional[str] = None
    visual_description: Optional[str] = None
    image_paths: List[str] = Field(default_factory=list)
    linked_figure_ids: List[str] = Field(default_factory=list)


class PenaltyNode(GraphNodeBase):
    type: Literal["penalty"] = "penalty"
    penalty_type: Optional[str] = None
    fine_min_vnd: Optional[int] = None
    fine_max_vnd: Optional[int] = None
    individual_min_vnd: Optional[int] = None
    individual_max_vnd: Optional[int] = None
    organization_min_vnd: Optional[int] = None
    organization_max_vnd: Optional[int] = None
    point_deduction: Optional[int] = None
    license_suspension: Optional[str] = None
    vehicle_impoundment: Optional[str] = None
    additional_penalties: List[str] = Field(default_factory=list)
    remedial_measures: List[str] = Field(default_factory=list)
    raw_penalty_text: Optional[str] = None


class ProcedureNode(GraphNodeBase):
    type: Literal["procedure"] = "procedure"
    name: Optional[str] = None
    target_audience: List[str] = Field(default_factory=list)
    dossier_requirements: List[str] = Field(default_factory=list)
    submission_methods: List[str] = Field(default_factory=list)
    processing_time_days: Optional[int] = None
    deadlines: List[str] = Field(default_factory=list)
    raw_procedure_text: Optional[str] = None


class GraphEdge(BaseModel):
    source: str
    target: Optional[str] = None
    target_ref: Optional[str] = None
    type: Literal[
        "HAS_CHAPTER",
        "HAS_ARTICLE",
        "HAS_CLAUSE",
        "HAS_POINT",
        "HAS_CHUNK",
        "PARENT_OF",
        "CITES",
        "HAS_TABLE",
        "HAS_FIGURE",
        "HAS_SIGN",
        "REPRESENTS_SIGN",
        "HAS_PENALTY",
        "HAS_PROCEDURE",
    ]
    raw: Optional[str] = None
