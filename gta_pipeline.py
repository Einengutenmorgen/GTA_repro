# gta_pipeline.py
from llm_client import call_llm
from prompts import OPEN_CODING_PROMPT, AXIAL_CODING_PROMPT, SELECTIVE_CODING_PROMPT

def run_open_coding(text_chunks, model_type="local"):
    """Extracts base concepts from individual chunks."""
    open_codes = []
    total = len(text_chunks)
    
    for i, chunk in enumerate(text_chunks):
        print(f"  -> Open Coding chunk {i+1}/{total}...")
        response = call_llm(OPEN_CODING_PROMPT, chunk, model_type)
        open_codes.append({
            "chunk_id": i + 1,
            "codes": response
        })
    return open_codes

def run_axial_coding(open_codes, model_type="local"):
    """Groups open codes into relational categories."""
    # Combine all extracted codes into a single string for the prompt
    combined_codes = "\n".join([f"Chunk {item['chunk_id']}: {item['codes']}" for item in open_codes])
    
    print("  -> Running Axial Coding synthesis...")
    return call_llm(AXIAL_CODING_PROMPT, combined_codes, model_type)

def run_selective_coding(axial_relations, model_type="local"):
    """Synthesizes relations into a final core theory."""
    print("  -> Running Selective Coding synthesis...")
    return call_llm(SELECTIVE_CODING_PROMPT, axial_relations, model_type)