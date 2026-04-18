import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    """
    Extract raw text from a PDF file while preserving layout as much as possible.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return None

    full_text = ''
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text('text')
        full_text += text + '\n'
    
    doc.close()
    return full_text

def pdf_to_txt():
    """
    Main extraction logic: Locate project paths, extract text from legal PDFs, and merge into a single raw text file.
    """
    # Base directory relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    # Input/Output definitions
    input_pdf_1 = os.path.join(project_root, "data/raw/36-2024-qh15.pdf")
    input_pdf_2 = os.path.join(project_root, "data/raw/36-2024-qh15_tiep.pdf")
    output_txt = os.path.join(project_root, "data/raw/luat_giao_thong_raw.txt")

    print(f"Extraction started. Root: {project_root}")

    # Process PDF parts
    text1 = extract_text_from_pdf(input_pdf_1)
    text2 = extract_text_from_pdf(input_pdf_2)

    # Save logic based on extraction results
    if text1 is not None and text2 is not None:
        with open(output_txt, 'w', encoding='utf-8') as file:
            file.write(text1 + '\n' + text2)
        print(f"Merged output saved: {output_txt}")
    elif text1 is not None:
        with open(output_txt, 'w', encoding='utf-8') as file:
            file.write(text1)
        print(f"Partial output saved (Part 1): {output_txt}")
    elif text2 is not None:
        with open(output_txt, 'w', encoding='utf-8') as file:
            file.write(text2)
        print(f"Partial output saved (Part 2): {output_txt}")
    else:
        print("Failed to process PDFs. Check file paths and document integrity.")
