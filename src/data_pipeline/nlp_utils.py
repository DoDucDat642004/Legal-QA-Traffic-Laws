import re
from underthesea import pos_tag
import advertools as adv

# Load Vietnamese stopwords for token filtering
VI_STOPWORDS = adv.stopwords['vietnamese']

def extract_legal_keywords(text):
    """
    Extract key legal entities and nouns to support Hybrid Search (BM25).
    """
    keywords = set()

    # Capture specific legal entities: currency, speed, age, units
    # Example: "1.000.000 đồng", "50 km/h", "18 tuổi"
    entities = re.findall(r'\b(\d+(?:[\.,]\d+)*(?:\s?(?:đồng|km/h|tuổi|năm|tháng|kg|tấn|mét|m)))\b', text, re.IGNORECASE)
    for entity in entities:
        keywords.add(entity.lower())

    # Filter nouns and proper nouns using POS tagging
    # Underthesea tag format: [('Làn đường', 'N'), ('là', 'V'), ...]
    tagged_words = pos_tag(text)

    for word, tag in tagged_words:
        # Keep Nouns (N) and Proper Nouns (Np) > 2 characters
        if tag in ['N', 'Np'] and len(word) > 2:
            word_clean = word.lower().strip()
            if word_clean not in VI_STOPWORDS:
                keywords.add(word_clean)

    return list(keywords)
