import re
from src.data_pipeline.nlp_utils import extract_legal_keywords

LAW_NAME = "Luật Trật tự, an toàn giao thông đường bộ (36/2024/QH15)"

def parse_legal_document(cleaned_text):
    """
    Parse cleaned text into structured JSON hierarchy: Chapter -> Article -> Clause.
    Enriches articles with metadata for hybrid search.
    """
    final_dataset = []
    
    # Segment by Chapter using Roman numeral markers
    chapter_blocks = re.split(r'(?m)^(Chương\s+[IVXLCDM]+\b)', cleaned_text)
    
    for i in range(1, len(chapter_blocks), 2):
        chapter_id = chapter_blocks[i].strip()
        chapter_body = chapter_blocks[i+1].strip()
        
        # Extract title from the first line of the chapter block
        chapter_parts = chapter_body.split('\n', 1)
        chapter_title = chapter_parts[0].strip()
        chapter_content = chapter_parts[1].strip() if len(chapter_parts) > 1 else ""
        
        # Match Articles (Điều) using lookahead to prevent overlapping
        article_pattern = r'(?m)^Điều\s+(?P<id>\d+)\.\s+(?P<body_text>.*?)(?=(?:^Điều\s+\d+\.)|\Z)'
        article_matches = re.finditer(article_pattern, chapter_content, re.DOTALL | re.IGNORECASE)
        
        for match in article_matches:
            article_id = match.group('id')
            body_text = match.group('body_text').strip()
            
            parts = body_text.split('\n', 1)
            article_title = parts[0].strip() if len(parts) > 0 else ""
            article_full_body = parts[1].strip() if len(parts) > 1 else body_text

            # Attempt segmentation into numbered clauses (Khoản)
            clause_pattern = r'(?m)^(\d+)\.\s+(.*?)(?=(?:^\d+\.\s+)|\Z)'
            clause_matches = list(re.finditer(clause_pattern, article_full_body, re.DOTALL))

            if clause_matches:
                for clause_match in clause_matches:
                    clause_id = clause_match.group(1)
                    clause_content = clause_match.group(2).strip()
                    
                    # Context assembly for Dense Search (RAG)
                    embedding_text = f"[{LAW_NAME} | {chapter_id}: {chapter_title} | Điều {article_id}: {article_title} | Khoản {clause_id}] {clause_content}"
                    keywords = extract_legal_keywords(clause_content)
                    
                    final_dataset.append({
                        "chapter_id": chapter_id, "chapter_title": chapter_title, 
                        "article_id": article_id, "article_title": article_title,
                        "clause_id": clause_id, "reference": f"Khoản {clause_id}, Điều {article_id}",
                        "content": clause_content,
                        "keywords": keywords,
                        "text_for_embedding": embedding_text
                    })
            else:
                # Fallback for articles without numbered clauses
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
