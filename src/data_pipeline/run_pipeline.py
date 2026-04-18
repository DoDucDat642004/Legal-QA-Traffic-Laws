import os
import json
import sys

# Add project root to sys.path to support absolute imports
# This allows running the script from anywhere within the project
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_pipeline.cleaner import clean_noise
from src.data_pipeline.extractor_pdf import pdf_to_txt
from src.data_pipeline.parser import parse_legal_document

def main():
    # 1. Step 1: Extract PDF pages into raw text
    pdf_to_txt()

    input_txt = os.path.join(project_root, "data/raw/luat_giao_thong_raw.txt")
    output_json = os.path.join(project_root, "data/processed/articles.json")

    if not os.path.exists(input_txt):
        print(f"Error: Raw file {input_txt} not found. Run extraction step first.")
        return

    # 2. Step 2: Load and Clean raw text data
    with open(input_txt, 'r', encoding='utf-8') as file:
        raw_text = file.read()
    
    cleaned_text = clean_noise(raw_text)

    # Output cleaned text for manual structural inspection
    debug_path = os.path.join(project_root, "data/processed/debug_cleaned.txt")
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    with open(debug_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    # 3. Step 3: Segment and Parse text into structured Article objects
    articles_data = parse_legal_document(cleaned_text)

    # 4. Step 4: Final Summary and JSON Export
    print(f"--- Pipeline Execution Complete ---")
    print(f"Parsed Sections: {len(articles_data)}")

    if articles_data:
        # Sort by article ID (numeric fallback)
        articles_data.sort(key=lambda x: int(x['article_id']) if x['article_id'].isdigit() else 999)
        
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(articles_data, f, ensure_ascii=False, indent=4)
        
        print(f"First Entry: {articles_data[0]['reference']}")
        print(f"Success! Final data saved: {output_json}")
    else:
        print("Failed to parse document structure. Check debug_cleaned.txt.")

if __name__ == "__main__":
    main()
