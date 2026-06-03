import asyncio
import glob
import json
import os
import logging
import re
import difflib
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.data_pipeline.text_normalizer import TextNormalizer
from src.rag.model_policy import async_generate_content_with_fallback, model_candidates

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExpertQA")

# Shared across one generation run to avoid retrying models already known to be exhausted.
EXHAUSTED_MODELS = set()
ALLOWED_INTENTS = {"DEFINITION", "PENALTY", "SCENARIO", "PROCEDURE", "EXCEPTION", "SIGN_MEANING", "TECHNICAL_SPEC"}
ALLOWED_DIFFICULTIES = {"EASY", "MEDIUM", "HARD"}
QA_PIPELINE_VERSION = "qa_v4_strict_verbatim_quotes_no_default_fuzzy"


def canonicalize_qa_pair(pair: dict, record: dict) -> dict:
    intent = str(pair.get("intent") or "").strip().upper()
    difficulty = str(pair.get("difficulty") or "").strip().upper()

    if intent not in ALLOWED_INTENTS:
        source = (
            record.get("source_body_exact")
            or record.get("original_text")
            or record.get("content")
            or ""
        ).lower()
        if any(k in source for k in ["phạt", "tiền", "trừ điểm", "tước"]):
            intent = "PENALTY"
        elif any(k in source for k in ["biển số", "biển báo", "vạch", "đèn tín hiệu"]):
            intent = "SIGN_MEANING"
        elif any(k in source for k in ["hồ sơ", "thủ tục", "trình tự", "thời hạn"]):
            intent = "PROCEDURE"
        else:
            intent = "DEFINITION"

    if difficulty in {"EZY", "BASIC"}:
        difficulty = "EASY"
    if difficulty not in ALLOWED_DIFFICULTIES:
        difficulty = "MEDIUM"

    pair["intent"] = intent
    pair["difficulty"] = difficulty
    pair["is_adversarial"] = bool(pair.get("is_adversarial", False))
    return pair


def record_priority(record: dict) -> int:
    if record.get("record_type") == "enriched_rule":
        return 3
    if record.get("record_type") == "source_legal_unit":
        return 1
    return 2


def select_best_records_by_chunk(records: list[dict]) -> list[dict]:
    best = {}
    for record in records:
        chunk_id = record.get("source_chunk_id")
        if not chunk_id:
            continue
        current = best.get(chunk_id)
        if current is None or record_priority(record) > record_priority(current):
            best[chunk_id] = record
    return list(best.values())


def _qa_meta_path(output_json_path: str) -> str:
    return output_json_path + ".meta.json"


def _write_qa_meta(output_json_path: str, input_json_path: str, qa_count: int, record_count: int) -> None:
    meta = {
        "qa_pipeline_version": QA_PIPELINE_VERSION,
        "input_json_path": input_json_path,
        "qa_count": qa_count,
        "record_count": record_count,
    }
    with open(_qa_meta_path(output_json_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _ensure_qa_output_version(output_json_path: str) -> None:
    if not os.path.exists(output_json_path):
        return
    try:
        with open(_qa_meta_path(output_json_path), "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("qa_pipeline_version") == QA_PIPELINE_VERSION:
            return
    except Exception:
        pass

    logger.warning(
        "QA output metadata is missing or stale; keeping %s in place for resume "
        "and validating existing QA against the current processed records.",
        output_json_path,
    )


def _qa_checkpoint_paths(output_json_path: str) -> list[str]:
    paths = []
    if os.path.exists(output_json_path):
        paths.append(output_json_path)
    backups = sorted(
        glob.glob(output_json_path + ".bak*"),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    paths.extend(p for p in backups if p not in paths)
    return paths


def _load_qa_checkpoint(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Could not load QA checkpoint %s: %s", path, exc)
        return []


def _table_source_text(table: dict) -> str:
    parts = [table.get("caption") or "", table.get("text") or ""]
    headers = table.get("headers") or []
    if headers:
        parts.append(" | ".join(str(h) for h in headers if h is not None))
    for row in table.get("rows") or []:
        if isinstance(row, list):
            parts.append(" | ".join(str(cell) for cell in row if cell is not None))
    return "\n".join(p for p in parts if p)


def _figure_source_text(figure: dict) -> str:
    return " ".join(str(figure.get(k) or "") for k in ("code", "name", "caption")).strip()


def _source_text(record: dict) -> str:
    parts = [
        record.get("source_body_exact")
        or record.get("original_text")
        or record.get("content")
        or "",
        record.get("meaning_and_usage") or "",
        record.get("violation_content") or "",
        record.get("qa_context") or "",
    ]
    for table in record.get("tables") or []:
        if isinstance(table, dict):
            parts.append(_table_source_text(table))
    for figure in record.get("figures") or []:
        if isinstance(figure, dict):
            parts.append(_figure_source_text(figure))
    return "\n".join(p for p in parts if p)


def _short_exact_quote(source_text: str) -> str:
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in (source_text or "").splitlines()
        if re.sub(r"\s+", " ", line).strip()
    ]
    candidates = [line for line in lines if 20 <= len(line) <= 260]
    if not candidates:
        text = re.sub(r"\s+", " ", source_text or "").strip()
        return text[:220]
    legal_priority = [
        line for line in candidates
        if any(k in line.lower() for k in ["phạt", "được", "không được", "phải", "biển", "bảng", "thủ tục", "thời hạn"])
    ]
    return (legal_priority or candidates)[0]


def _reference_from_record(record: dict) -> str:
    ref = record.get("legal_reference") or {}
    parts = []
    if ref.get("point"):
        parts.append(f"Điểm {ref.get('point')}")
    if ref.get("clause"):
        parts.append(f"Khoản {ref.get('clause')}")
    if ref.get("article"):
        parts.append(f"Điều {ref.get('article')}")
    if ref.get("chapter"):
        parts.append(f"Chương {ref.get('chapter')}")
    parts.append(record.get("doc_name") or ref.get("document") or "văn bản pháp luật")
    return ", ".join(str(p) for p in parts if p)


def deterministic_fallback_qa(record: dict) -> list[dict]:
    source = _source_text(record)
    quote = _short_exact_quote(source)
    if len(quote) < 8:
        return []

    citation = _reference_from_record(record)
    lower = source.lower()
    if record.get("tables"):
        intent = "TECHNICAL_SPEC"
        question = "Bảng trong quy định này nói về nội dung gì và cần hiểu thế nào?"
    elif record.get("figures") or record.get("figure_refs") or "biển" in lower or "vạch" in lower:
        intent = "SIGN_MEANING"
        question = "Biển báo hoặc hình minh họa trong phần này có ý nghĩa gì?"
    elif any(k in lower for k in ["phạt", "tiền", "trừ điểm", "tước"]):
        intent = "PENALTY"
        question = "Trường hợp này bị xử lý hoặc áp dụng mức phạt như thế nào?"
    elif any(k in lower for k in ["hồ sơ", "thủ tục", "trình tự", "thời hạn"]):
        intent = "PROCEDURE"
        question = "Thủ tục hoặc thời hạn trong quy định này được nêu ra sao?"
    else:
        intent = "DEFINITION"
        question = "Quy định này nói gì theo cách dễ hiểu?"

    answer = (
        f"Theo {citation}, nội dung cần căn cứ là: \"{quote}\". "
        "Khi áp dụng, cần bám đúng đoạn trích này và đối chiếu toàn bộ điều/khoản liên quan trong văn bản nguồn."
    )
    qa = {
        "intent": intent,
        "difficulty": "EASY",
        "question": question,
        "answer": answer,
        "quote": quote,
        "citation": citation,
        "search_queries": [question, citation],
        "is_adversarial": False,
        "source_chunk_id": record.get("source_chunk_id"),
        "doc_name": record.get("doc_name"),
        "source_reference": citation,
        "thought_process": "Tóm tắt ngắn: QA fallback được tạo từ trích dẫn nguyên văn để bảo toàn căn cứ và tránh bịa đặt.",
        "validated": True,
        "generation_mode": "deterministic_fallback",
    }
    return [qa] if validate_qa_pair(qa, source, record.get("doc_name", "")) else []

class SingleQA(BaseModel):
    intent: str = Field(description="Loại câu hỏi: DEFINITION, PENALTY, SCENARIO, PROCEDURE, EXCEPTION.")
    difficulty: str = Field(description="Độ khó: EASY, MEDIUM, HARD.")
    question: str = Field(description="Câu hỏi bằng tiếng Việt theo phong cách người dân.")
    answer: str = Field(description="Câu trả lời đầy đủ, chuyên nghiệp, giải thích rõ ràng và trích dẫn căn cứ.")
    quote: str = Field(description="Trích dẫn nguyên văn (Verbatim) từ văn bản nguồn để làm căn cứ cho câu trả lời.")
    citation: str = Field(description="Căn cứ pháp lý đầy đủ. VD: 'Khoản 2 Điều 6 Nghị định 100/2019/NĐ-CP'")
    search_queries: list[str] = Field(description="Danh sách các từ khóa tìm kiếm mà người dùng có thể gõ vào thanh tìm kiếm liên quan đến câu hỏi này.")
    is_adversarial: bool = Field(description="Đánh dấu câu hỏi lắt léo để test RAG.")

class ExpertQASet(BaseModel):
    thought_process: str = Field(description="Phân tích ngầm về hành vi, đối tượng và điều kiện trong đoạn text này.")
    qa_pairs: list[SingleQA]


def canonical_quote_text(text: str) -> str:
    text = TextNormalizer.normalize_vietnamese(text or "").lower()
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def word_canonical_quote_text(text: str) -> str:
    return re.sub(r'[^\w]+', ' ', canonical_quote_text(text)).strip()


def validate_qa_pair(pair: dict, source_text: str, doc_name: str, *, log_failure: bool = False) -> bool:
    """Validate QA pairs while tolerating minor model formatting changes and OCR noise."""
    quote = pair.get("quote", "").strip()
    answer = pair.get("answer", "").strip()
    citation = pair.get("citation", "").strip()
    
    if not quote or len(quote) < 8: return False
    if not answer or len(answer) < 15: return False
    
    clean_quote = canonical_quote_text(quote)
    clean_source = canonical_quote_text(source_text)
    
    if clean_quote in clean_source:
        return True

    word_quote = word_canonical_quote_text(quote)
    word_source = word_canonical_quote_text(source_text)
    if word_quote and word_quote in word_source:
        return True
        
    relaxed_validation = os.getenv("QA_RELAXED_QUOTE_VALIDATION", "").lower() in {"1", "true", "yes"}
    fuzzy_validation = os.getenv("QA_ENABLE_FUZZY_QUOTE_VALIDATION", "").lower() in {"1", "true", "yes"}
    fuzzy_threshold = float(os.getenv("QA_FUZZY_QUOTE_MIN_RATIO", "0.94"))

    # Keep fuzzy matching opt-in because QA citations must remain auditable.
    if fuzzy_validation and len(word_quote) > 40:
        source_words = word_source.split()
        quote_words = word_quote.split()
        window_size = len(quote_words) + 15
        
        best_ratio = 0
        for i in range(len(source_words) - len(quote_words) + 1):
            segment = " ".join(source_words[i:i+window_size])
            ratio = difflib.SequenceMatcher(None, word_quote, segment).ratio()
            if ratio > best_ratio: best_ratio = ratio
            if ratio >= fuzzy_threshold:
                logger.info(f" - [FUZZY PASS] Quote matches segment with {ratio:.2f} similarity")
                return True
                
    if relaxed_validation:
        # Last resort for very messy OCR. This is deliberately opt-in because it is not
        # strict enough for citation-grade quote verification.
        stop_words = {"và", "của", "cho", "các", "những", "là", "theo", "tại", "với", "trong", "được", "phải", "này"}
        quote_keywords = [w for w in re.findall(r'\w{3,}', quote.lower()) if w not in stop_words]
        source_words_set = set(re.findall(r'\w{3,}', source_text.lower()))
        matches = [w for w in quote_keywords if w in source_words_set]
        if len(set(matches)) >= max(6, int(len(set(quote_keywords)) * 0.8)):
            logger.info(f" - [RELAXED KEYWORD PASS] Found {len(set(matches))} key terms from quote in source.")
            return True
    
    modes = ["strict"]
    if fuzzy_validation:
        modes.append(f"fuzzy>={fuzzy_threshold:.2f}")
    if relaxed_validation:
        modes.append("relaxed-keyword")
    should_log_failure = log_failure or os.getenv("QA_LOG_INVALID_QUOTES", "").lower() in {"1", "true", "yes", "on"}
    if should_log_failure:
        logger.warning(
            "QA Validation Failed: Verbatim quote '%s...' not found in source text (%s validation).",
            quote[:40],
            "+".join(modes),
        )
    return False

async def generate_expert_qa(client: genai.Client, record: dict, semaphore: asyncio.Semaphore) -> list[dict]:
    source_body = _source_text(record)
    doc_name = record.get('doc_name', 'Văn bản pháp luật')
    chunk_id = record.get('source_chunk_id', 'unknown')
    
    ref_obj = record.get('legal_reference', {})
    ref_parts = []
    
    article = str(ref_obj.get('article', ''))
    if not article or any(x in article.lower() for x in ["không xác định", "n/a", "none"]):
        match = re.search(r'(?:^|\n)điều\s+(\d+[a-z]?)', source_body.lower())
        if match: 
            article = match.group(1).upper()
        else:
            parts = chunk_id.split("_")
            candidate_articles = [p for p in parts if p.isdigit() and int(p) < 500]
            if candidate_articles:
                article = candidate_articles[0]

    clause = str(ref_obj.get('clause', ''))
    if not clause or any(x in clause.lower() for x in ["không xác định", "n/a", "none"]):
        match = re.search(r'(?:^|\n)(\d+)\.', source_body)
        if match: 
            clause = match.group(1)
        else:
            match = re.search(r'khoản\s+(\d+)', source_body.lower())
            if match: clause = match.group(1)

    point = str(ref_obj.get('point', ''))
    if not point or any(x in point.lower() for x in ["không xác định", "n/a", "none"]):
        match = re.search(r'(?:^|\n)([a-z])\)', source_body.lower())
        if match: point = match.group(1)

    def is_valid_article(v): 
        v = str(v).strip().upper()
        return v and not any(x in v.lower() for x in ["không xác định", "n/a", "none", "unknown"])
    
    def is_valid_clause(v):
        v = str(v).strip()
        return v.isdigit()
        
    def is_valid_point(v):
        v = str(v).strip().lower()
        return len(v) == 1 and v.isalpha() # Point is usually a, b, c...

    if is_valid_point(point): ref_parts.append(f"Điểm {point}")
    if is_valid_clause(clause): ref_parts.append(f"Khoản {clause}")
    if is_valid_article(article): ref_parts.append(f"Điều {article}")
    if is_valid_article(ref_obj.get('chapter')): ref_parts.append(f"Chương {ref_obj.get('chapter')}")
    
    reference = ", ".join(ref_parts) if ref_parts else "Một số quy định"
    reference = reference.strip(", ")

    if len(source_body.strip()) < 15:
        return deterministic_fallback_qa(record)

    has_tables = bool(record.get("tables"))
    has_figures = bool(record.get("figures") or record.get("figure_refs") or record.get("asset_ids"))
    prompt_mode = os.getenv("QA_PROMPT_MODE", "few_shot_cot").lower()
    examples = ""
    if "few" in prompt_mode:
        examples = """
# FEW-SHOT STYLE EXAMPLES
Example 1:
TEXT: "Người điều khiển xe không chấp hành hiệu lệnh của đèn tín hiệu giao thông..."
QA: intent=SCENARIO, question="Nếu tôi vượt đèn đỏ thì bị xử lý theo căn cứ nào?", answer="...", quote="Người điều khiển xe không chấp hành hiệu lệnh của đèn tín hiệu giao thông"

Example 2:
TEXT: "Biển số P.102 'Cấm đi ngược chiều'..."
QA: intent=SIGN_MEANING, question="Biển P.102 có ý nghĩa gì và gặp biển này có được đi vào không?", answer="...", quote="Biển số P.102"
"""
    reasoning_instruction = (
        "Write a brief Vietnamese analysis_summary in thought_process before qa_pairs. "
        "Do not invent facts outside [TEXT]."
        if "cot" in prompt_mode
        else "Use zero-shot extraction from [TEXT] only. Keep thought_process as a one-sentence summary."
    )

    async with semaphore:
        prompt = f"""
# SYSTEM ROLE
You are a Legal Knowledge Engineer for Vietnamese traffic-law RAG datasets.

# DATA INPUT
- Document: {doc_name}
- Location: {reference}
- Has tables: {has_tables}
- Has traffic-sign/image assets: {has_figures}
- [TEXT]: 
{source_body}

# METHOD
{reasoning_instruction}
{examples}

# TASK
Generate 2-4 detailed QA pairs in Vietnamese based ONLY on the provided [TEXT].
The QA set must be natural, diverse, and useful for RAG retrieval.

# COVERAGE REQUIREMENTS
- Cover the main legal rule in [TEXT].
- If [TEXT] contains a fine, point deduction, suspension, or remedial measure, include a PENALTY QA.
- If [TEXT] contains procedure/time/dossier/process wording, include a PROCEDURE QA.
- If [TEXT] contains exceptions/conditions, include an EXCEPTION or HARD scenario QA.
- If [TEXT] contains a traffic sign, road marking, table, or image asset, include SIGN_MEANING or TECHNICAL_SPEC QA and mention the sign/table identifier.
- Questions must sound like real Vietnamese users, not copied headings.
- Answers must cite {reference} and must not add facts absent from [TEXT].
- quote must be a short verbatim substring copied exactly from [TEXT].

Return ONLY a JSON object following this schema:
{{
  "thought_process": "Phân tích nội dung",
  "qa_pairs": [
    {{
      "intent": "DEFINITION | PENALTY | SCENARIO | PROCEDURE | EXCEPTION | SIGN_MEANING | TECHNICAL_SPEC",
      "question": "Câu hỏi tự nhiên",
      "answer": "Trả lời chuyên nghiệp",
      "quote": "Trích dẫn nguyên văn từ [TEXT]",
      "citation": "Căn cứ pháp lý",
      "search_queries": ["query1"],
      "is_adversarial": false,
      "difficulty": "MEDIUM"
    }}
  ]
}}
"""
        model_fail_counts = {}
        for attempt in range(6):
            try:
                models_to_try = model_candidates("QA_PRIMARY_MODEL", "QA_MODEL", task="qa_generation")
                available_models = [m for m in models_to_try if m not in EXHAUSTED_MODELS]
                if not available_models:
                    logger.critical(f"ALL MODELS EXHAUSTED! Cannot process chunk {chunk_id}")
                    return deterministic_fallback_qa(record)
                
                model_name = available_models[attempt % len(available_models)]
                
                # If a model failed with 500 too many times in this specific chunk call, skip it
                if model_fail_counts.get(model_name, 0) >= 2:
                    logger.warning(f" - Skipping unstable model {model_name} for this chunk.")
                    continue

                await asyncio.sleep(2 + attempt) 
                
                config = types.GenerateContentConfig(
                    temperature=0.1, 
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                    response_schema=ExpertQASet,
                )
                
                res, model_name = await async_generate_content_with_fallback(
                    client,
                    contents=prompt,
                    config=config,
                    env_names=("QA_PRIMARY_MODEL", "QA_MODEL"),
                    models=available_models,
                    task="qa_generation",
                    logger=logger,
                    label="QA generation",
                )
                if res.text:
                    json_text = res.text.strip()
                    # Robust JSON cleanup
                    if not json_text.startswith("{") and "thought_process" in json_text:
                        json_text = "{" + json_text
                    
                    try:
                        parsed_data = ExpertQASet.model_validate_json(json_text)
                    except Exception:
                        match = re.search(r'\{.*\}', json_text, re.DOTALL)
                        if match:
                            parsed_data = ExpertQASet.model_validate_json(match.group(0))
                        else:
                            raise ValueError(f"Model {model_name} returned invalid JSON: {json_text[:200]}")
                    
                    final_qa = []
                    rejected_quotes = 0
                    for pair in parsed_data.qa_pairs:
                        data = canonicalize_qa_pair(pair.model_dump(), record)
                        if validate_qa_pair(data, source_body, doc_name, log_failure=False):
                            data.update({
                                "source_chunk_id": chunk_id, "doc_name": doc_name,
                                "source_reference": reference, "thought_process": parsed_data.thought_process,
                                "validated": True
                            })
                            final_qa.append(data)
                        else:
                            rejected_quotes += 1
                    
                    if final_qa: 
                        return final_qa
                    else:
                        logger.warning(
                            " - [VALIDATION FAIL] %s produced %s QA pair(s), all rejected by strict quote validation for %s.",
                            model_name,
                            rejected_quotes,
                            chunk_id,
                        )
                
            except Exception as e:
                err_str = str(e)
                if "500" in err_str or "INTERNAL" in err_str:
                    model_fail_counts[model_name] = model_fail_counts.get(model_name, 0) + 1
                    logger.warning(f" - [500 INTERNAL] Model {model_name} failed. Attempt {attempt+1}")
                elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if "PerDay" in err_str:
                        EXHAUSTED_MODELS.add(model_name)
                    await asyncio.sleep(10)
                else:
                    logger.error(f"[QA FAIL] {chunk_id} using {model_name} | {e}")
                
                await asyncio.sleep(5)
        return deterministic_fallback_qa(record)

async def process_generate_qa(input_json_path: str, output_json_path: str):
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else None
    
    if not os.path.exists(input_json_path):
        logger.error(f"Input file not found: {input_json_path}")
        return
    _ensure_qa_output_version(output_json_path)

    with open(input_json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    records = select_best_records_by_chunk(records)
        
    logger.info(f"🚀 Started QA Generation for {len(records)} law chunks...")
    
    semaphore = asyncio.Semaphore(3)
    qa_dataset = []
    valid_counts_by_chunk = {}
    record_by_chunk = {r.get("source_chunk_id"): r for r in records if r.get("source_chunk_id")}
    
    checkpoint_paths = _qa_checkpoint_paths(output_json_path)
    seen_questions = set()
    for checkpoint_path in checkpoint_paths:
        for qa in _load_qa_checkpoint(checkpoint_path):
            if not isinstance(qa, dict):
                continue
            chunk_id = qa.get("source_chunk_id")
            record = record_by_chunk.get(chunk_id)
            if not record:
                continue
            qa = canonicalize_qa_pair(qa, record)
            question_key = re.sub(r"\s+", " ", (qa.get("question") or "").strip().lower())
            dedupe_key = (chunk_id, question_key)
            if not question_key or dedupe_key in seen_questions:
                continue
            source_text = (
                record.get("source_body_exact")
                or record.get("original_text")
                or record.get("content")
                or ""
            )
            if validate_qa_pair(qa, source_text, record.get("doc_name", "")):
                qa["validated"] = True
                qa_dataset.append(qa)
                seen_questions.add(dedupe_key)
                valid_counts_by_chunk[chunk_id] = valid_counts_by_chunk.get(chunk_id, 0) + 1

    if checkpoint_paths:
        logger.info(
            " - Loaded %s valid QA pairs from %s checkpoint file(s), covering %s/%s chunks.",
            len(qa_dataset),
            len(checkpoint_paths),
            len(valid_counts_by_chunk),
            len(records),
        )
        tmp_path = output_json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, output_json_path)

    min_qa_per_chunk = int(os.getenv("QA_MIN_PAIRS_PER_CHUNK", "2"))
    pending = [
        r for r in records
        if valid_counts_by_chunk.get(r.get('source_chunk_id'), 0) < min_qa_per_chunk
    ]
    logger.info(f"Pending: {len(pending)} / Total: {len(records)}")

    batch_size = 2
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i+batch_size]
        if client is None:
            results = [deterministic_fallback_qa(r) for r in batch]
        else:
            tasks = [generate_expert_qa(client, r, semaphore) for r in batch]
            results = await asyncio.gather(*tasks)
        
        for q_list in results:
            if not q_list:
                continue
            for qa in q_list:
                chunk_id = qa.get("source_chunk_id")
                question_key = re.sub(r"\s+", " ", (qa.get("question") or "").strip().lower())
                dedupe_key = (chunk_id, question_key)
                if not chunk_id or not question_key or dedupe_key in seen_questions:
                    continue
                qa_dataset.append(qa)
                seen_questions.add(dedupe_key)
                valid_counts_by_chunk[chunk_id] = valid_counts_by_chunk.get(chunk_id, 0) + 1
            
        tmp_path = output_json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, output_json_path)

        logger.info(f" - Progress: {min(i+batch_size, len(pending))}/{len(pending)} | Dataset size: {len(qa_dataset)} | Saved to {output_json_path}")

        await asyncio.sleep(5)

    _write_qa_meta(output_json_path, input_json_path, len(qa_dataset), len(records))
