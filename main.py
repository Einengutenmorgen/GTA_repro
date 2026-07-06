# main.py
import glob
import json
from PyPDF2 import PdfReader
from gta_pipeline import run_open_coding, run_axial_coding, run_selective_coding

DATA_DIR = "data/raw_TeachingWithSharedData"

def extract_and_chunk_interviews(chunk_size=1500):
    """
    Reads only the interview PDFs from the data directory and splits them 
    into character-length chunks to safely fit inside the LLM context window.
    """
    # Target only Instructor and Student interviews (ignore codebook/metadata)
    pdf_files = glob.glob(f"{DATA_DIR}/Furlong-et-al_Interview*.pdf")
    text_chunks = []
    
    for file_path in pdf_files:
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"
        
        # Minimal chunking strategy 
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        text_chunks.extend(chunks)
        
    return text_chunks

def main():
    # Set to "proprietary" if you want to use OpenAI directly
    MODEL_TO_USE = "proprietary" 
    
    print("=== Phase 0: Data Extraction ===")
    chunks = extract_and_chunk_interviews()
    print(f"Extracted {len(chunks)} text chunks from interview PDFs.\n")

    print("=== Phase 1: Open Coding ===")
    open_codes = run_open_coding(chunks, MODEL_TO_USE)
    with open("output_open_codes.json", "w") as f:
        json.dump(open_codes, f, indent=4)
    print("Saved -> output_open_codes.json\n")

    print("=== Phase 2: Axial Coding ===")
    axial_relations = run_axial_coding(open_codes, MODEL_TO_USE)
    with open("output_axial_codes.md", "w") as f:
        f.write(axial_relations)
    print("Saved -> output_axial_codes.md\n")

    print("=== Phase 3: Selective Coding ===")
    final_theory = run_selective_coding(axial_relations, MODEL_TO_USE)
    with open("output_final_theory.md", "w") as f:
        f.write(final_theory)
    print("Saved -> output_final_theory.md\n")

    print("Experiment complete! Compare 'output_final_theory.md' with 'Furlong-et-al_Codebook.pdf'.")

if __name__ == "__main__":
    main()