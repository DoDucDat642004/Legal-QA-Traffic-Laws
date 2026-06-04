from __future__ import annotations

import re
from typing import Any


HEADING_RE = re.compile(r"^(#{2,3})\s+(.*\S)\s*$")


def split_markdown_sections(answer: str) -> list[dict[str, str]]:
    text = (answer or "").strip()
    if not text:
        return []

    sections: list[dict[str, str]] = []
    current_title = "Mở đầu"
    current_lines: list[str] = []
    saw_heading = False

    for raw_line in text.splitlines():
        match = HEADING_RE.match(raw_line.strip())
        if match:
            body = "\n".join(current_lines).strip()
            if body or saw_heading or current_title != "Mở đầu":
                sections.append({"title": current_title, "body": body})
            current_title = match.group(2).strip()
            current_lines = []
            saw_heading = True
            continue
        current_lines.append(raw_line)

    body = "\n".join(current_lines).strip()
    if body or not sections:
        sections.append({"title": current_title, "body": body})

    return [section for section in sections if section["title"] or section["body"]]


def vision_display_rows(vision: dict[str, Any] | None) -> list[tuple[str, str]]:
    if not vision:
        return []

    rows: list[tuple[str, str]] = []

    is_sign = vision.get("is_traffic_sign")
    if is_sign is not None:
        rows.append(("Nhận diện", "Có" if is_sign else "Không"))

    for label, key in [
        ("Mã tin cậy", "trusted_codes"),
        ("Mã dự đoán", "candidate_codes"),
        ("Nhóm biển", "sign_group"),
        ("Hình dạng", "shape"),
        ("Độ tin cậy", "confidence"),
        ("Màu chủ đạo", "dominant_colors"),
        ("Ký hiệu", "symbol"),
        ("Chữ/số", "text"),
    ]:
        value = vision.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value if item)
        elif key == "confidence":
            try:
                value = f"{float(value):.2f}"
            except (TypeError, ValueError):
                value = str(value)
        else:
            value = str(value)
        rows.append((label, value))

    return rows
