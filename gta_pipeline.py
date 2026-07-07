# gta_pipeline.py
from llm_client import call_llm
from prompts import OPEN_CODING_PROMPT, AXIAL_CODING_PROMPT, SELECTIVE_CODING_PROMPT
import json

def run_open_coding(text_chunks, model_type="local"):
    """Extracts base concepts from individual chunks as structured JSON."""
    open_codes = []
    total = len(text_chunks)
    
    for i, chunk in enumerate(text_chunks):
        print(f"  -> Open Coding chunk {i+1}/{total}...")
        raw_response = call_llm(OPEN_CODING_PROMPT, chunk, model_type)

        if not raw_response or not raw_response.strip():
            print(f"  -> Warning: empty response for chunk {i+1}. Marking as failed.")
            open_codes.append({"__status__": "failed", "chunk_id": i + 1, "reason": "empty_response"})
            continue
        
        try:
            # Clean and parse the JSON array
            clean_response = raw_response.strip().strip("```json").strip("```")
            structured_codes = json.loads(clean_response)
            
            # Add these codes to our master list
            open_codes.extend(structured_codes)
        except json.JSONDecodeError:
            print(f"  -> Warning: LLM failed to return valid JSON for chunk {i+1}. Skipping.")
            open_codes.append({"__status__": "failed", "chunk_id": i + 1, "reason": "json_parse_error"})

            
    return open_codes

def run_axial_coding(open_codes, model_type="local"):
    """Groups open codes into relational categories with traceability."""
    # Combine the code and the raw text passage for maximum context
    combined_codes = "\n".join([
        f"Code: {item.get('open_code', '')} | Context: {item.get('text_passage', '')}" 
        for item in open_codes
    ])
    
    print("  -> Running Axial Coding synthesis...")
    raw_response = call_llm(AXIAL_CODING_PROMPT, combined_codes, model_type)
    
    try:
        clean_response = raw_response.strip().strip("```json").strip("```")
        structured_axial_codes = json.loads(clean_response)
        return structured_axial_codes
    except json.JSONDecodeError:
        print("  -> Warning: LLM failed to return valid JSON. Returning raw text.")
        return {"error": "JSON parse failed", "raw_output": raw_response}

def run_selective_coding(axial_relations, model_type="local"):
    """Synthesizes relations into a final core theory."""
    print("  -> Running Selective Coding synthesis...")
    
    # Convert the parsed JSON object back to a formatted string for the LLM prompt
    relations_text = json.dumps(axial_relations, indent=2)
    
    return call_llm(SELECTIVE_CODING_PROMPT, relations_text, model_type)