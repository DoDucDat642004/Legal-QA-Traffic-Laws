import re
import unicodedata
import hashlib

class TextNormalizer:
    # Suffixes common in Vietnamese syllables that get split by font artifacts
    VN_SPLIT_SUFFIXES = r'(?:ều|ển|ời|ành|ặt|ường|gược|ại|ồng|ắt|ừng|ược|ổi|ẫn|ộng|ến|ỉnh|ọng)'
    PDF_CONTROL_SPACE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]+')
    INTERNAL_MARKER_RE = re.compile(r'^\[INTERNAL_PAGE_MARKER_\d+\]$')
    PAGE_NUMBER_RE = re.compile(r'^(?:[-–—]\s*)?\d{1,4}(?:\s*[-–—])?$')
    TECHNICAL_METADATA_RE = re.compile(
        r'^(?:'
        r'người\s+ký|email|e-mail|cơ\s+quan|thời\s+gian\s+ký|ký\s+bởi|'
        r'digitally\s+signed|signature\s+valid|serial\s+number|signed\s+by'
        r')\s*[:：]',
        re.IGNORECASE,
    )
    FOOTER_OR_WATERMARK_RE = re.compile(
        r'(?:'
        r'\bvgp\b|'
        r'cổng\s+thông\s+tin\s+điện\s+tử\s+chính\s+phủ|'
        r'cổng\s+thông\s+điện\s+tử\s+chính\s+phủ|'
        r'c[oô]ng\s+th[oô]ng\s+tin\s+điện\s+tử\s+ch[ií]nh\s+ph[uủ]|'
        r'vanban\.chinhphu\.vn|chinhphu\.vn|'
        r'vbpl\.vn|thuvienphapluat|'
        r'http[s]?://|www\.'
        r')',
        re.IGNORECASE,
    )

    @staticmethod
    def restore_pdf_control_spaces(text: str) -> str:
        """Turn PDF control separators into real spaces before text cleanup.

        Some QCVN pages use control characters such as ``\x05`` between words.
        Dropping them creates glued captions like "Lànđườngdànhcho".
        """
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return TextNormalizer.PDF_CONTROL_SPACE_RE.sub(" ", text)

    @staticmethod
    def _preserve_initial_case(match: re.Match, replacement: str) -> str:
        value = match.group(0)
        if value[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    @staticmethod
    def fix_traffic_caption_spacing(text: str) -> str:
        """Repair common glued traffic-sign caption phrases from QCVN PDFs."""
        if not text:
            return ""

        phrase_fixes = [
            (r'\blànđườngdànhcho\s*', 'làn đường dành cho '),
            (r'\bhếtlànđườngdànhcho\s*', 'hết làn đường dành cho '),
            (r'\bbiểngộplànđườngtheophươngtiện\b', 'biển gộp làn đường theo phương tiện'),
            (r'\bkếtthúclànđườngtheophươngtiện\b', 'kết thúc làn đường theo phương tiện'),
            (r'\bhướngđitrênmỗilànđường\b', 'hướng đi trên mỗi làn đường'),
            (r'\bmỗilànđường\b', 'mỗi làn đường'),
            (r'\blànđường\b', 'làn đường'),
            (r'\bđườngtheophươngtiện\b', 'đường theo phương tiện'),
            (r'\bphảitheo\b', 'phải theo'),
            (r'vàxe', 'và xe'),
            (r'\bxeôtôkhách\b', 'xe ô tô khách'),
            (r'\bxeôtôcon\b', 'xe ô tô con'),
            (r'\bxeôtôtải\b', 'xe ô tô tải'),
            (r'\bxeôtô\b', 'xe ô tô'),
            (r'\bôtôkhách\b', 'ô tô khách'),
            (r'\bôtôcon\b', 'ô tô con'),
            (r'\bôtôtải\b', 'ô tô tải'),
            (r'\bôtô\b', 'ô tô'),
            (r'\bxemáy\s*', 'xe máy '),
            (r'\bxeđạp\s*', 'xe đạp '),
            (r'\bxebuýt\s*', 'xe buýt '),
            (r'\bbu\s*ýt\b', 'buýt'),
        ]
        for pattern, replacement in phrase_fixes:
            text = re.sub(
                pattern,
                lambda m, repl=replacement: TextNormalizer._preserve_initial_case(m, repl),
                text,
                flags=re.IGNORECASE,
            )
        return text

    @staticmethod
    def fix_word_splits(text: str) -> str:
        """Ghép lại các âm tiết tiếng Việt bị tách bởi artifact khoảng cách PDF."""
        # Pattern: prefix (1-4 chars) + space + known suffix + word boundary
        pattern = rf'([A-Za-zÀ-ỹĐđ]{{1,4}})\s({TextNormalizer.VN_SPLIT_SUFFIXES})\b'
        prev = None
        while prev != text:
            prev = text
            text = re.sub(pattern, r'\1\2', text)
        return text

    @staticmethod
    def recover_legal_keywords(text: str) -> str:
        """Fixes common OCR typos in Vietnamese legal keywords."""
        if not text: return ""

        # Keep this context-aware. A previous broad IGNORECASE replacement turned
        # normal body text into artifacts such as "người Điều khiển" and "Mục đích".
        heading_fixes = [
            (r'(?m)^(\s*)[ĐD]Iều(?=\s+\d+[a-zA-Z]?\b)', r'\1Điều'),
            (r'(?m)^(\s*)[ĐD]iêu(?=\s+\d+[a-zA-Z]?\b)', r'\1Điều'),
            (r'(?m)^(\s*)Kho[âa]n(?=\s+\d+\b)', r'\1Khoản'),
            (r'(?m)^(\s*)Chuong(?=\s+[IVXLCDM\d]+\b)', r'\1Chương'),
            (r'(?m)^(\s*)M[uụ]c(?=\s+[IVXLCDM\d]+\b)', r'\1Mục'),
            (r'(?m)^(\s*)Ph[uụ]\s*l[uụ]c(?=\s+[IVXLCDM\d]+\b)', r'\1Phụ lục'),
        ]
        for pattern, replacement in heading_fixes:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        citation_fixes = [
            (r'\b[ĐD]Iều(?=\s+\d+[a-zA-Z]?\b)', 'Điều'),
            (r'\b[ĐD]iêu(?=\s+\d+[a-zA-Z]?\b)', 'Điều'),
            (r'\b[DĐ]i[êe]m(?=\s+[a-zđ]\b)', 'điểm'),
            (r'\bKho[âa]n(?=\s+\d+\b)', 'khoản'),
        ]
        for pattern, replacement in citation_fixes:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    @classmethod
    def is_noise_line(cls, line: str) -> bool:
        """Return True for extraction artifacts that are not legal content.

        This intentionally stays conservative: it removes page numbers, PDF signature
        metadata and source-site watermarks, but keeps all legal headings/body lines.
        """
        clean = line.strip()
        if not clean:
            return False
        if cls.INTERNAL_MARKER_RE.match(clean):
            return False

        lower = clean.lower()
        if cls.PAGE_NUMBER_RE.match(clean):
            return True
        if re.match(r'^[ivxlcdm]{1,4}$', lower):
            return True
        if len(clean) <= 5 and re.match(r'^[A-ZĐ\s.]+$', clean) and clean not in {"MỤC", "PHẦN"}:
            return True
        if re.match(r'^trang\s+\d{1,4}(?:\s*/\s*\d{1,4})?$', lower):
            return True
        if cls.TECHNICAL_METADATA_RE.match(clean):
            return True
        if cls.FOOTER_OR_WATERMARK_RE.search(clean):
            return True
        if re.match(r'^(?:ngày|ngay)\s*:\s*\d{6,}[\s.]*$', lower):
            return True
        return False

    @classmethod
    def strip_extraction_noise(cls, text: str) -> str:
        if not text:
            return ""
        kept = []
        for line in text.splitlines():
            if cls.is_noise_line(line):
                continue
            kept.append(line.rstrip())
        return "\n".join(kept)

    @staticmethod
    def normalize_vietnamese(text: str) -> str:
        if not text:
            return ""
        
        text = unicodedata.normalize('NFC', text)
        text = TextNormalizer.restore_pdf_control_spaces(text)
        
        text = TextNormalizer.fix_word_splits(text)
        text = TextNormalizer.fix_traffic_caption_spacing(text)
        
        text = TextNormalizer.recover_legal_keywords(text)

        text = TextNormalizer.strip_extraction_noise(text)
        
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\t')
        
        text = re.sub(r'\s+([.,:;!?])', r'\1', text)
        text = re.sub(r'([.,:;!?])(?=[^\s\d])', r'\1 ', text)
        
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    @staticmethod
    def get_hash(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    @classmethod
    def process(cls, text: str) -> dict:
        raw_hash = cls.get_hash(text)
        normalized = cls.normalize_vietnamese(text)
        norm_hash = cls.get_hash(normalized)
        
        return {
            "original": text,
            "normalized": normalized,
            "raw_sha256": raw_hash,
            "norm_sha256": norm_hash
        }
