import re
from underthesea import pos_tag
import advertools as adv

# Vietnamese stopwords for keyword filtering
VI_STOPWORDS = adv.stopwords['vietnamese']

def extract_legal_keywords(text):
    """
    Extract key legal entities and nouns to support Sparse Search (BM25).
    """
    keywords = set()

    # Capture quantitative legal entities: currency, speed, age, units
    entities = re.findall(r'\b(\d+(?:[\.,]\d+)*(?:\s?(?:đồng|km/h|tuổi|năm|tháng|kg|tấn|mét|m)))\b', text, re.IGNORECASE)
    for entity in entities:
        keywords.add(entity.lower())

    # Part-of-Speech tagging for noun extraction
    tagged_words = pos_tag(text)

    for word, tag in tagged_words:
        # Retention criteria: Nouns (N) and Proper Nouns (Np) > 2 chars, excluding stopwords
        if tag in ['N', 'Np'] and len(word) > 2:
            word_clean = word.lower().strip()
            if word_clean not in VI_STOPWORDS:
                keywords.add(word_clean)

    return list(keywords)
