import os
import json
import logging
import fitz
import re
import hashlib
from src.data_pipeline.text_normalizer import TextNormalizer

logger = logging.getLogger("FigureExtractor")

class FigureExtractor:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.asset_dir = os.path.join(project_root, "data/processed/sign_assets")
        os.makedirs(self.asset_dir, exist_ok=True)
        
    def _get_graphical_regions(self, page) -> list[fitz.Rect]:
        """Identifies all bboxes containing images or complex drawings."""
        regions = []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] == 1:
                regions.append(fitz.Rect(b["bbox"]))
        
        paths = page.get_drawings()
        if len(paths) < 500:
            for p in paths:
                if p["rect"].width > 5 and p["rect"].height > 5:
                    regions.append(p["rect"])
        else:
            logger.debug(f" - [SKIP DRAWINGS] Too many vector paths ({len(paths)}) on page {page.number}")
                
        return regions

    def _extract_sign_page_by_caption_bands(self, page, page_rect) -> list[dict]:
        """Specialized logic for appendix pages: crop regions directly related to captions."""
        caption_re = re.compile(
            r'(?:Hình\s+[A-Z]\.\d+\s*[-–]\s*)?'
            r'Biển\s*số\s*([A-ZĐ]{0,3}\.?\d{2,3}[a-zđ]?)\s*[:\-]?\s*[\"“]?([^;#\"”]*)',
            re.IGNORECASE
        )
        captions = []
        page_dict = page.get_text("dict")
        
        for block in page_dict["blocks"]:
            if block["type"] != 0: continue
            for line in block["lines"]:
                text_raw = "".join(s["text"] for s in line["spans"]).strip()
                text = TextNormalizer.normalize_vietnamese(text_raw)
                for m in caption_re.finditer(text):
                    captions.append({
                        "code": m.group(1),
                        "name": m.group(2).strip(),
                        "bbox": line["bbox"],
                        "y": line["bbox"][1]
                    })
        
        assets = []
        # Sort by vertical position
        captions.sort(key=lambda x: x["y"])
        
        for i, cap in enumerate(captions):
            # Target region is usually directly ABOVE the caption in QCVN appendix
            previous_y = captions[i - 1]["y"] if i > 0 else 0
            y_top = max(previous_y + 8, cap["y"] - 220) # Heuristic for sign height
            y_bot = cap["y"] - 5
            
            target_rect = fitz.Rect(page_rect.x0 + 40, y_top, page_rect.x1 - 40, y_bot)
            
            if target_rect.width > 20 and target_rect.height > 20:
                assets.append({
                    "code": cap["code"],
                    "name": cap["name"],
                    "rect": target_rect,
                    "caption_bbox": cap["bbox"]
                })
        
        return assets

    def extract_figures(self, pdf_path: str, doc_name: str):
        """Extracts images/figures/tables and their captions from a PDF."""
        logger.info(f"Extracting figures from {pdf_path}...")
        doc = fitz.open(pdf_path)
        doc_base_name = os.path.basename(pdf_path).replace(".pdf", "")
        
        # Regex for captions common in QCVN (e.g., Hình G.42, Biển số 101, Biển)
        caption_re = re.compile(
            r'^\s*(?:Hình|Bảng|Biển\s+số|Biển)\s+([A-ZĐ]?\s*\.?\s*\d+(?:\.\d+)*[a-zA-ZĐđ]?|[A-Z]{1,3}\.\d+(?:\.\d+)*[a-zA-Z]?)(?:\s*[\-\:]\s*(.*))?',
            re.IGNORECASE
        )
        
        extracted_assets = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_dict = page.get_text("dict")
            blocks = page_dict["blocks"]
            page_rect = page.rect
            
            # Appendix Mode for sign-heavy pages
            paths_count = len(page.get_drawings())
            normalized_page_text = TextNormalizer.normalize_vietnamese(page.get_text("text"))
            sign_caption_count = len(re.findall(r"Biển\s*số\s*(?:DP|IE|P|W|R|I|S|E)\s*\.?\s*\d", normalized_page_text, re.IGNORECASE))
            if paths_count > 500 or sign_caption_count >= 2:
                logger.info(f" - [APPENDIX MODE] Page {page_num} has {sign_caption_count} sign captions. Using caption bands.")
                bands = self._extract_sign_page_by_caption_bands(page, page_rect)
                for band in bands:
                    self._save_asset(
                        page,
                        page_num,
                        doc_base_name,
                        doc_name,
                        band["code"],
                        band["name"],
                        band["rect"],
                        band["caption_bbox"],
                        extracted_assets,
                        source_mode="appendix_band",
                    )
                continue

            graphics_bboxes = self._get_graphical_regions(page)
            
            for block in blocks:
                if block["type"] == 0: # text
                    for line in block["lines"]:
                        raw_line_text = "".join(span["text"] for span in line["spans"]).strip()
                        line_text = TextNormalizer.normalize_vietnamese(raw_line_text)
                        match = caption_re.search(line_text)
                        if not match:
                            continue

                        code = re.sub(r"\s+", "", match.group(1))
                        name = (match.group(2) or line_text[match.end(1):]).strip(" -:\t")
                        cap_bbox = fitz.Rect(self._line_bbox(line))

                        target_bbox = self._find_best_region(cap_bbox, graphics_bboxes, page_rect)
                        
                        if target_bbox:
                            self._save_asset(
                                page,
                                page_num,
                                doc_base_name,
                                doc_name,
                                code,
                                name,
                                target_bbox,
                                cap_bbox,
                                extracted_assets,
                                source_mode="generic_caption",
                            )

        # Save metadata
        extracted_assets = self._postprocess_assets(extracted_assets)
        meta_path = os.path.join(self.asset_dir, f"{doc_base_name}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(extracted_assets, f, ensure_ascii=False, indent=2)
            
        doc.close()
        return extracted_assets

    def _save_asset(self, page, page_num, doc_base_name, doc_name, code, name, target_bbox, cap_bbox, assets_list, source_mode: str = "generic_caption"):
        # Add padding to ensure full content is captured
        target_bbox.x0 = max(0, target_bbox.x0 - 5)
        target_bbox.y0 = max(0, target_bbox.y0 - 5)
        target_bbox.x1 = min(page.rect.width, target_bbox.x1 + 5)
        target_bbox.y1 = min(page.rect.height, target_bbox.y1 + 5)
        
        code = re.sub(r"\s+", "", TextNormalizer.normalize_vietnamese(str(code or ""))).strip()
        name = TextNormalizer.normalize_vietnamese(str(name or ""))

        asset_hash = hashlib.sha1(f"{doc_base_name}_{page_num}_{code}_{target_bbox}".encode("utf-8")).hexdigest()[:10]
        asset_id = f"{doc_base_name}_{code}_{asset_hash}".replace(".", "_").replace("/", "_")
        img_filename = f"{asset_id}.png"
        img_path = os.path.join(self.asset_dir, img_filename)

        pix = page.get_pixmap(clip=target_bbox, matrix=fitz.Matrix(3, 3))
        pix.save(img_path)

        asset_info = {
            "id": asset_id,
            "code": code,
            "name": name,
            "doc_name": doc_name,
            "page": page_num,
            "bbox": list(target_bbox),
            "caption_bbox": list(cap_bbox) if cap_bbox else None,
            "image_path": os.path.join("data/processed/sign_assets", img_filename),
            "source_mode": source_mode,
        }
        assets_list.append(asset_info)
        logger.info(f" - [PRO-CROP] Extracted: {code} - {name} on Page {page_num}")

    def _caption_quality(self, asset: dict) -> int:
        name = TextNormalizer.normalize_vietnamese(str(asset.get("name") or ""))
        if not name:
            return -100

        lower = name.lower()
        score = len(name)
        if asset.get("source_mode") == "appendix_band":
            score += 80
        if re.search(r'\bbiển\s+số\s+(?:dp|ie|p|w|r|i|s|e)\s*\.?\s*\d', lower, re.IGNORECASE):
            score -= 60
        if any(k in lower for k in ["chỉ là", "trường hợp", "không áp dụng", "ví dụ"]):
            score -= 50
        if len(name.split()) < 3:
            score -= 25
        if any(
            lower.startswith(prefix)
            for prefix in [
                "làn đường",
                "hết làn đường",
                "hướng đi",
                "biển gộp",
                "kết thúc",
                "đường dành",
            ]
        ):
            score += 30
        return score

    def _postprocess_assets(self, assets: list[dict]) -> list[dict]:
        """Normalize duplicate sign metadata and repair weak captions by sign code."""
        best_by_code: dict[str, dict] = {}
        for asset in assets:
            code = re.sub(r"\s+", "", TextNormalizer.normalize_vietnamese(str(asset.get("code") or ""))).strip()
            name = TextNormalizer.normalize_vietnamese(str(asset.get("name") or ""))
            asset["code"] = code
            asset["name"] = name
            if not code:
                continue
            asset["caption_quality"] = self._caption_quality(asset)
            current = best_by_code.get(code)
            if code and (current is None or asset["caption_quality"] > current["caption_quality"]):
                best_by_code[code] = asset

        seen = set()
        out = []
        for asset in assets:
            if not asset.get("code"):
                continue
            key = (asset.get("code"), asset.get("image_path"))
            if key in seen:
                continue
            seen.add(key)
            best = best_by_code.get(asset.get("code") or "")
            if best and best.get("name") and best["caption_quality"] >= asset.get("caption_quality", -100):
                asset["name"] = best["name"]
                asset["caption_source"] = best.get("source_mode") or asset.get("source_mode")
            if not str(asset.get("name") or "").strip() and asset.get("code"):
                asset["name"] = f"Biển số {asset['code']}"
                asset["caption_source"] = "code_fallback"
            out.append(asset)
        return out

    def _find_best_region(self, cap_bbox: fitz.Rect, regions: list[fitz.Rect], page_rect: fitz.Rect) -> fitz.Rect:
        candidates = []
        for r in regions:
            dist_v = min(abs(cap_bbox.y0 - r.y1), abs(cap_bbox.y1 - r.y0))
            if dist_v < 250:
                overlap_h = max(0, min(cap_bbox.x1, r.x1) - max(cap_bbox.x0, r.x0))
                if overlap_h > 0 or abs(cap_bbox.x0 - r.x0) < 150:
                    candidates.append(r)
        
        if not candidates:
            fallback = fitz.Rect(page_rect.x0 + 20, max(0, cap_bbox.y0 - 220), page_rect.x1 - 20, cap_bbox.y0 - 5)
            return fallback
            
        res = candidates[0]
        for c in candidates[1:]:
            dist = min(abs(res.y0 - c.y1), abs(res.y1 - c.y0))
            if dist < 40:
                res |= c
        return res

    def _line_bbox(self, line):
        boxes = [span["bbox"] for span in line["spans"]]
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    extractor = FigureExtractor(project_root)
    extractor.extract_figures(os.path.join(project_root, "data/raw/51-bgtvt-kem.pdf"), "QCVN 41:2024")
