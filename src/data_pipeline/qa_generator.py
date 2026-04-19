import json
import os
import time
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from src.data_pipeline.data_split import split_dataset

# CONFIGURATION & ENVIRONMENT
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("CRITICAL: GEMINI_API_KEY missing from environment.")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)
MODEL_ID = "gemma-4-31b-it"

def generate_qa_pair(context_text: str, reference: str, retries: int = 3) -> list:
    """
    Generates high-quality, legally grounded Q&A pairs using Gemini.
    """
    # English instructions for precision, Vietnamese for content generation.
    prompt = f"""
    ROLE: Senior AI Data Engineer & Vietnamese Traffic Law Expert.
    
    TASK: Perform exhaustive info extraction from [LEGAL CONTEXT]. 
    Generate up to 10 diverse, mutually exclusive Q&A pairs. 
    If context is short, generate maximum possible unique pairs.

    EXTRACTION STRATEGY:
    1. Sub-clause Mining: Target specific points (a, b, c) to ensure no detail is missed.
    2. Conditions & Exceptions: Target phrases like "trừ trường hợp", "nếu", "khi".
    3. Entities: Identify authorities, quantified fines, and legal deadlines.

    DIVERSITY MATRIX:
    - Layman/Citizen: Practical scenarios ("Hôm qua tôi...", "Liệu có được...").
    - Vulnerable User: Pedestrians, cyclists, safety focus.
    - Logistics: Enterprise liability, commercial transport rules.
    - Enforcement: Procedures, document checks, violations.
    - Debater: Argumentative edge-cases and compliance disputes.
    - Learner: Clear definitions of legal terminology.

    RULES:
    1. GROUNDING: Use ONLY [LEGAL CONTEXT]. No hallucinations.
    2. STANDALONE: Questions must be clear without context (replace pronouns with nouns).
    3. LANGUAGE: Vietnamese for "question" and "answer" fields.
    4. SUFFIX: Every answer must end with: (Theo {reference}).
    5. INTENT: Classify as: "Fine", "Definition", "Rule", "Authority", "Condition_Exception", "Procedure", "Other".

    [LEGAL CONTEXT]:
    {context_text}

    OUTPUT: JSON Array only. No markdown formatting.
    """
    
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2, # Low temperature for strict grounding
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_level="HIGH"
                    )
                )
            )
            
            if not response.text:
                continue
            
            raw_json = response.text.strip()
            if raw_json.startswith("```json"):
                raw_json = raw_json[7:-3].strip()
                
            qa_list = json.loads(raw_json)
            
            if isinstance(qa_list, list) and len(qa_list) > 0:
                return qa_list
            
        except Exception as e:
            print(f"WARN: API Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(2)
            
    return []

def run_generate_qa_pair():
    """
    Orchestrates the QA generation pipeline with checkpoint/resume support.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    input_json = os.path.join(project_root, "data/processed/articles.json")
    output_dir = os.path.join(project_root, "data/qa_pairs")
    output_file = os.path.join(output_dir, "dataset_qa.json")
    output_dir_splits = os.path.join(project_root, "data/qa_pairs/splits")

    if not os.path.exists(input_json):
        print(f"ERROR: Missing source file: {input_json}")
        sys.exit(1)

    with open(input_json, 'r', encoding='utf-8') as f:
        articles_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    full_dataset = []
    processed_refs = set()

    # Load checkpoint to resume progress
    if os.path.exists(output_file):
        print(f"Checkpoint detected: {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                full_dataset = json.load(f)
            processed_refs = {item['reference'] for item in full_dataset}
            print(f"-> Resumed: {len(full_dataset)} existing questions.")
            print(f"-> Skipping: {len(processed_refs)} processed articles.\n")
        except json.JSONDecodeError:
            print("Checkpoint corrupted. Starting fresh.")

    global_id = len(full_dataset) + 1

    for idx, article in enumerate(articles_data):
        reference = article.get('reference', 'Unknown')
        
        if reference in processed_refs:
            continue

        content = article.get('content', '')
        if len(content) < 30:
            continue

        print(f"[{idx+1}/{len(articles_data)}] Processing: {reference}")
        context = article.get('text_for_embedding', content)
        
        qa_batch = generate_qa_pair(context, reference)
        
        if qa_batch:
            for qa in qa_batch:
                if not all(k in qa for k in ("question", "answer", "intent")):
                    continue
                    
                full_dataset.append({
                    "id": f"QA_{global_id:05d}",
                    "intent": qa.get("intent", "Other"),
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "reference": reference,
                    "context": context,
                    "metadata": {
                        "chapter": article.get("chapter_id"),
                        "article_id": article.get("article_id"),
                        "clause_id": article.get("clause_id", "Entire")
                    }
                })
                global_id += 1

            # Persistent save after each article to prevent data loss on interruption
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(full_dataset, f, ensure_ascii=False, indent=4)
            
            processed_refs.add(reference)
            print(f"   Collected questions: {len(full_dataset)}")
            time.sleep(1)

    print("\nGeneration complete. Finalizing splits...")
    os.makedirs(output_dir_splits, exist_ok=True)
    split_dataset(full_dataset, output_dir_splits)

if __name__ == "__main__":
    run_generate_qa_pair()
