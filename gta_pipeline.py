# gta_pipeline.py
from llm_client import call_llm
from prompt_registry import get_prompts, DEFAULT_TRADITION
import json

def run_open_coding(chunks, model_type="local", tradition=DEFAULT_TRADITION):
    """Extracts base concepts from each chunk as structured JSON.

    `chunks` is a list of chunk dicts from chunk_transcript (each carries
    'text' plus identity fields). Every emitted code is stamped with its
    originating chunk_id so axial categories remain traceable back to
    participant/question for the empty-slot escalation ladder. The prompt
    input is chunk['text'], byte-identical to the previous string pipeline.

    `tradition` selects the prompt set ("straussian" | "charmaz") via the
    registry, replacing the old manual comment-toggle of imports.
    """
    P = get_prompts(tradition)
    open_codes = []
    total = len(chunks)
    
    for i, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        print(f"  -> Open Coding chunk {i+1}/{total} ({chunk_id})...")
        raw_response = call_llm(P.open, chunk["text"], model_type)

        if not raw_response or not raw_response.strip():
            print(f"  -> Warning: empty response for {chunk_id}. Marking as failed.")
            open_codes.append({"__status__": "failed", "chunk_id": chunk_id, "reason": "empty_response"})
            continue
        
        try:
            # Clean and parse the JSON array
            clean_response = raw_response.strip().strip("```json").strip("```")
            structured_codes = json.loads(clean_response)
            
            # Stamp provenance onto each code, then add to the master list
            for code in structured_codes:
                code["chunk_id"] = chunk_id
            open_codes.extend(structured_codes)
        except json.JSONDecodeError:
            print(f"  -> Warning: LLM failed to return valid JSON for {chunk_id}. Marking as failed.")
            open_codes.append({"__status__": "failed", "chunk_id": chunk_id, "reason": "json_parse_error"})

            
    return open_codes

def run_axial_coding(open_codes, model_type="local", tradition=DEFAULT_TRADITION):
    """Groups open codes into relational categories with traceability."""
    P = get_prompts(tradition)
    # Combine the code and the raw text passage for maximum context
    combined_codes = "\n".join([
        f"Code: {item.get('open_code', '')} | Context: {item.get('text_passage', '')}" 
        for item in open_codes
    ])
    
    print("  -> Running Axial Coding synthesis...")
    raw_response = call_llm(P.axial, combined_codes, model_type)
    
    try:
        clean_response = raw_response.strip().strip("```json").strip("```")
        structured_axial_codes = json.loads(clean_response)
        return structured_axial_codes
    except json.JSONDecodeError:
        print("  -> Warning: LLM failed to return valid JSON. Returning raw text.")
        return {"error": "JSON parse failed", "raw_output": raw_response}

def run_selective_coding(axial_relations, model_type="local", tradition=DEFAULT_TRADITION):
    """Synthesizes relations into a final core theory."""
    P = get_prompts(tradition)
    print("  -> Running Selective Coding synthesis...")
    
    # Convert the parsed JSON object back to a formatted string for the LLM prompt
    relations_text = json.dumps(axial_relations, indent=2)
    
    return call_llm(P.selective, relations_text, model_type)