import re
import unicodedata

def clean_noise(raw_data):
    """
    Remove headers, footers, and formatting noise while preserving legal structure.
    """
    # Unicode normalization for consistent Vietnamese character handling
    text = unicodedata.normalize('NFC', raw_data)
    
    # Remove Gazette (CÔNG BÁO) headers and page metadata
    gazette_pattern = r'\d*\s*CÔNG BÁO/Số\s*\d+\s*\+\s*\d+/Ngày\s*\d+-\d+-\d+\s*\d*'
    text = re.sub(gazette_pattern, '', text, flags=re.IGNORECASE)

    # Strip administrative boilerplate using multiline matching
    meta_patterns = [
        r'^VGP$', r'^Người ký:.*$', r'^Email:.*$', r'^Cơ quan:.*$', r'^Thời gian ký:.*$', 
        r'^CHINHPHU\.VN$', r'^VĂN BẢN QUY PHẠM PHÁP LUẬT$', r'^CHỦ TỊCH NƯỚC - QUỐC HỘI$', 
        r'^VĂN PHÒNG CHÍNH PHỦ XUẤT BẢN.*$', r'^CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM$', 
        r'^Độc lập - Tự do - Hạnh phúc$', r'^QUỐC HỘI$', r'^LUẬT$', r'^Luật số:.*$'
    ]
    combined_meta = r'(?im)(' + '|'.join(meta_patterns) + r')'
    text = re.sub(combined_meta, '', text)

    # Cleanup artifacts and escaped characters
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
    text = text.replace('\\', '')

    # Ensure Articles start on new lines to facilitate parsing
    text = re.sub(r'\s*(Điều\s+\d+\.)', r'\n\1', text, flags=re.IGNORECASE)

    # Reconstruction logic to merge sentences fragmented by PDF line breaks
    lines = text.split('\n')
    cleaned_lines = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line: continue
        
        if i < len(lines) - 1:
            next_line = lines[i+1].strip()
            if not next_line:
                cleaned_lines.append(line + "\n")
                continue
            
            # Merging heuristics:
            # 1. Ends with a grammatical connector
            # 2. Next line starts with lowercase
            # 3. No terminal punctuation and next line is not a new structural marker
            is_connector = re.search(r'(tại|của|và|hoặc|là|các|những|theo|đến|về|trong|cho|điểm|khoản|điều|chương)\s*$', line, re.IGNORECASE)
            next_is_lower = next_line[0].islower()
            no_punctuation = not re.search(r'[.;:]$', line)
            
            is_heading = re.match(r'^(Điều\s+\d+\.|Chương\s+[IVXLCDM]+\b)', line, re.IGNORECASE)
            is_next_marker = re.match(r'^(Điều\s+\d+\.|Chương\s+[IVXLCDM]+\b|\d+\.\s|[a-z]\)\s)', next_line, re.IGNORECASE)
            
            if (is_connector or next_is_lower) or (no_punctuation and not is_next_marker and not is_heading):
                cleaned_lines.append(line + " ")
            else:
                cleaned_lines.append(line + "\n")
        else:
            cleaned_lines.append(line)
            
    text = "".join(cleaned_lines)

    # Final whitespace normalization
    text = re.sub(r' +', ' ', text) 
    text = re.sub(r'\n{2,}', '\n', text)
    
    return text.strip()
