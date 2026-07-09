# main.py
import os 
import glob
import json
from PyPDF2 import PdfReader
from gta_pipeline import run_open_coding, run_axial_coding, run_selective_coding
from chunking import chunk_transcript
import time

BASE_DATA_DIR = "data/RelationshipQuality"

def extract_and_chunk_interviews(target_dir, pairs_per_chunk=1):
    """Q&A-aware chunking: one interviewer-question + participant-answer per chunk
    (or `pairs_per_chunk` pairs batched). Nothing dropped."""
    pdf_files = glob.glob(os.path.join(target_dir, "**", "*.pdf"), recursive=True)
    text_chunks = []
    
    for file_path in sorted(pdf_files):
        print(f"    Reading: {os.path.basename(file_path)}")
        text_chunks.extend(chunk_transcript(file_path, pairs_per_chunk=pairs_per_chunk))

    return text_chunks

def main():
    # Set to "proprietary" if you want to use OpenAI directly
    MODEL_TO_USE = "proprietary" 
    
    target_countries = [
        #"Silan-Ciruelas_BRA",
        #"Silan-Ciruelas_FRA",
        #"Silan-Ciruelas_PHL",
        #"Silan-Ciruelas_TUR",
        #"Silan-Ciruelas_USA"
        "Silan-Ciruelas_USA/Silan-Ciruelas_USA_Opt1"
        #"Silan-Ciruelas_USA/Silan-Ciruelas_USA_Opt1and2"
    ]

    for folder_name in target_countries:
        # 1. Setup paths
        country_dir = os.path.join(BASE_DATA_DIR, folder_name)
        
        # Create a safe name for folders (e.g., replaces slashes)
        run_name = folder_name.replace("/", "_")
        
        # Create a dedicated output directory for this specific run
        timecode = time.strftime("%Y%m%d%H%M%S")
        output_dir = os.path.join(BASE_DATA_DIR, f"output_{run_name}_{timecode}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=======================================================")
        print(f"🚀 STARTING EXPERIMENT FOR: {run_name}")
        print(f"=======================================================")

        # 2. Extract Data
        print("=== Phase 0: Data Extraction ===")
        chunks = extract_and_chunk_interviews(country_dir)
        
        if not chunks:
            print(f"No text extracted for {run_name}. Skipping...\n")
            continue
            
        print(f"Extracted {len(chunks)} text chunks from PDFs.\n")

        # 3. Open Coding
        print("=== Phase 1: Open Coding ===")
        open_codes = run_open_coding(chunks, MODEL_TO_USE)
        open_out_path = os.path.join(output_dir, "output_open_codes.json")
        
        with open(open_out_path, "w") as f:
            json.dump(open_codes, f, indent=4)
        print(f"Saved -> {open_out_path}\n")

        #contuinue from step 2 
        # with open('/Users/christophhau/Desktop/GTA/data/RelationshipQuality/output_Silan-Ciruelas_USA_Silan-Ciruelas_USA_Opt1/output_open_codes.json', "r") as f:
        #     open_codes=json.load(f)
        
        # for oc in open_codes:
        #     print(oc)
        #     continue


        # 4. Axial Coding (Now correctly saving as JSON)
        print("=== Phase 2: Axial Coding ===")
        axial_relations = run_axial_coding(open_codes, MODEL_TO_USE)
        axial_out_path = os.path.join(output_dir, "output_axial_codes.json")
        
        with open(axial_out_path, "w") as f:
            json.dump(axial_relations, f, indent=4)
        print(f"Saved -> {axial_out_path}\n")

        # 5. Selective Coding (Still saves as Markdown)
        print("=== Phase 3: Selective Coding ===")
        final_theory = run_selective_coding(axial_relations, MODEL_TO_USE)
        theory_out_path = os.path.join(output_dir, "output_final_theory.md")
        
        with open(theory_out_path, "w") as f:
            f.write(final_theory)
        print(f"Saved -> {theory_out_path}\n")

if __name__ == "__main__":
    main()