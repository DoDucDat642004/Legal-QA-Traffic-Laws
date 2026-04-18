import re
from src.data_pipeline.nlp_utils import extract_legal_keywords

LAW_NAME = "Luật Trật tự, an toàn giao thông đường bộ (36/2024/QH15)"

def parse_legal_document(cleaned_text):
    """
    Parse cleaned text into structured JSON: Chapter -> Article -> Clause.
    Enrich output with metadata for Vector and Sparse search support.
    """
    final_dataset = []
    
    # Segment document into chapters based on "Chương" and Roman numerals
    # capturing group split: [pre-text, header1, body1, header2, body2...]
    chapter_blocks = re.split(r'(?m)^(Chương\s+[IVXLCDM]+\b)', cleaned_text)
    
    for i in range(1, len(chapter_blocks), 2):
        chapter_id = chapter_blocks[i].strip()
        chapter_body = chapter_blocks[i+1].strip()
        
        # Extract Chapter Title (first line after header)
        chapter_parts = chapter_body.split('\n', 1)
        chapter_title = chapter_parts[0].strip()
        chapter_content = chapter_parts[1].strip() if len(chapter_parts) > 1 else ""
        
        # Parse Articles (Điều) within the chapter body
        # (?m)^ matches start of line, capturing until the next "Điều" or end of block
        article_pattern = r'(?m)^Điều\s+(?P<id>\d+)\.\s+(?P<body_text>.*?)(?=(?:^Điều\s+\d+\.)|\Z)'
        article_matches = re.finditer(article_pattern, chapter_content, re.DOTALL | re.IGNORECASE)
        
        for match in article_matches:
            article_id = match.group('id')
            body_text = match.group('body_text').strip()
            
            # Split Title (first line) and Content (rest) of the Article
            parts = body_text.split('\n', 1)
            article_title = parts[0].strip() if len(parts) > 0 else ""
            article_full_body = parts[1].strip() if len(parts) > 1 else body_text

            # Sub-chunking: Attempt to split into numbered clauses (Khoản)
            clause_pattern = r'(?m)^(\d+)\.\s+(.*?)(?=(?:^\d+\.\s+)|\Z)'
            clause_matches = list(re.finditer(clause_pattern, article_full_body, re.DOTALL))

            if clause_matches:
                for clause_match in clause_matches:
                    clause_id = clause_match.group(1)
                    clause_content = clause_match.group(2).strip()
                    
                    # Embedding context: Mix metadata with content for Dense Search (PhoBERT)
                    embedding_text = f"[{LAW_NAME} | {chapter_id}: {chapter_title} | Điều {article_id}: {article_title} | Khoản {clause_id}] {clause_content}"
                    keywords = extract_legal_keywords(clause_content)
                    
                    final_dataset.append({
                        "chapter_id": chapter_id, "chapter_title": chapter_title, 
                        "article_id": article_id, "article_title": article_title,
                        "clause_id": clause_id, "reference": f"Khoản {clause_id}, Điều {article_id}",
                        "content": clause_content,
                        "keywords": keywords,                  # Sparse search
                        "text_for_embedding": embedding_text   # Dense search (RAG)
                    })
            else:
                # Handle single-block articles
                embedding_text = f"[{LAW_NAME} | {chapter_id}: {chapter_title} | Điều {article_id}: {article_title}] {article_full_body}"
                keywords = extract_legal_keywords(article_full_body)
                
                final_dataset.append({
                    "chapter_id": chapter_id, "chapter_title": chapter_title, 
                    "article_id": article_id, "article_title": article_title,
                    "clause_id": "Entire", "reference": f"Điều {article_id}",
                    "content": article_full_body,
                    "keywords": keywords,
                    "text_for_embedding": embedding_text
                })
            
    return final_dataset
