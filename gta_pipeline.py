# gta_pipeline.py
from llm_client import call_llm
from prompt_registry import (
    get_prompts,
    get_charmaz_recursion_prompts,
    _fill,
    DEFAULT_TRADITION,
    DEFAULT_DATASET,
)
import json

def run_open_coding(chunks, model_type="local", tradition=DEFAULT_TRADITION, dataset=DEFAULT_DATASET):
    """Extracts base concepts from each chunk as structured JSON.

    `chunks` is a list of chunk dicts from chunk_transcript (each carries
    'text' plus identity fields). Every emitted code is stamped with its
    originating chunk_id so axial categories remain traceable back to
    participant/question for the empty-slot escalation ladder. The prompt
    input is chunk['text'], byte-identical to the previous string pipeline.

    `tradition` selects the prompt set ("straussian" | "charmaz") via the
    registry, replacing the old manual comment-toggle of imports.

    `dataset` selects which STUDY CONTEXT is spliced into that prompt set
    ("silan" default | "semeval"); see prompt_registry.get_prompts and
    study_contexts.py. Defaults to "silan", so existing callers that only
    pass `tradition` are unaffected.
    """
    P = get_prompts(tradition, dataset=dataset)
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

def run_axial_coding(open_codes, model_type="local", tradition=DEFAULT_TRADITION, dataset=DEFAULT_DATASET):
    """Groups open codes into relational categories with traceability.

    `dataset` behaves as in run_open_coding (default "silan" preserves prior
    behavior for callers that only pass `tradition`).
    """
    P = get_prompts(tradition, dataset=dataset)
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

def run_selective_coding(axial_relations, model_type="local", tradition=DEFAULT_TRADITION, dataset=DEFAULT_DATASET):
    """Synthesizes relations into a final core theory.

    `dataset` behaves as in run_open_coding (default "silan" preserves prior
    behavior for callers that only pass `tradition`).
    """
    P = get_prompts(tradition, dataset=dataset)
    print("  -> Running Selective Coding synthesis...")
    
    # Convert the parsed JSON object back to a formatted string for the LLM prompt
    relations_text = json.dumps(axial_relations, indent=2)
    
    return call_llm(P.selective, relations_text, model_type)


# ===========================================================================
# Charmaz constructivist-GT steps (memo-writing + integration).
#
# These are ADDITIVE and only invoked on the Charmaz arm; the Straussian path
# above is untouched. Memos are real artifacts (they feed the reflection loop
# and the final integration), so they get their own pipeline functions rather
# than being folded into coding. See prompts_charmaz_recursion.py.
# ===========================================================================

def run_initial_memo(open_codes, model_type="local", dataset=DEFAULT_DATASET):
    """Charmaz Step 4 — initial memo-writing over the initial codes.

    Reason-then-write: the prompt asks the model to reason about meaning and
    cross-incident comparison before emitting each memo. Returns a parsed dict
    ({"memos": [...]}) or a fallback envelope on parse failure.

    `dataset` selects the dataset-appropriate prompt via
    prompt_registry.get_charmaz_recursion_prompts (replacing the old direct
    import from prompts_charmaz_recursion.py + study_contexts.
    swap_integration_closers). The skeleton puts {coded_data} at the very end
    of the prompt (data-last convention), so it's filled in via `_fill()` here
    and the fully-assembled prompt is sent as the whole system message, with
    an empty user turn -- mirroring run_memo_sorting's existing pattern.
    """
    R = get_charmaz_recursion_prompts(dataset)
    coded_data = "\n".join(
        f"- {oc.get('open_code', '')}  ::  {oc.get('text_passage', '')}"
        for oc in open_codes
        if isinstance(oc, dict) and "__status__" not in oc
    )
    prompt = _fill(R.initial_memo, coded_data=coded_data)
    print("  -> Writing initial memos...")
    raw = call_llm(prompt, "", model_type)
    return _safe_parse(raw, fallback={"memos": [], "__raw__": raw})


def run_advanced_memo(focused_categories, model_type="local", dataset=DEFAULT_DATASET):
    """Charmaz Step 6 — advanced memo-writing over the focused categories.

    Must name thin/ambiguous areas: those feed gap-focused re-sampling in the
    reflection loop. Returns a parsed dict ({"memos": [...]}).

    `dataset` behaves as in run_initial_memo: pulls the dataset-appropriate
    template from prompt_registry.get_charmaz_recursion_prompts and fills its
    trailing {focused_categories} placeholder via `_fill()`.
    """
    R = get_charmaz_recursion_prompts(dataset)
    cats_text = json.dumps(focused_categories, indent=2)
    prompt = _fill(R.advanced_memo, focused_categories=cats_text)
    print("  -> Writing advanced memos...")
    raw = call_llm(prompt, "", model_type)
    return _safe_parse(raw, fallback={"memos": [], "__raw__": raw})


def run_memo_sorting(focused_categories, memos, model_type="local", dataset=DEFAULT_DATASET):
    """Charmaz Step 9 — sort & integrate memos into the theoretical account.

    This is the Charmaz arm's terminal step (replaces the selective-coding call
    for this tradition). Returns the integrated account as text (Markdown).

    `dataset` selects the dataset-appropriate closing/integration language via
    prompt_registry.get_charmaz_recursion_prompts -- this prompt previously
    named "relationship quality" verbatim regardless of dataset, which is why
    a semeval Charmaz run's final theoretical account came back with leftover
    Silan relationship-quality language. Fixed here (now via the two-axis
    registry rather than a post-hoc string swap); defaults to "silan".
    """
    R = get_charmaz_recursion_prompts(dataset)
    prompt = _fill(
        R.memo_sorting,
        focused_categories=json.dumps(focused_categories, indent=2),
        memos=json.dumps(memos, indent=2),
    )
    print("  -> Sorting memos into integrated theoretical account...")
    return call_llm(prompt, "", model_type)


def _safe_parse(raw, fallback):
    """Shared parse-and-clean for the Charmaz memo calls (same contract as the
    coding functions above)."""
    if not raw or not raw.strip():
        return fallback
    try:
        clean = raw.strip().strip("```json").strip("```")
        return json.loads(clean)
    except json.JSONDecodeError:
        return fallback