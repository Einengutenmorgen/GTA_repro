# main.py
import os 
import glob
import json
from PyPDF2 import PdfReader
from gta_pipeline import run_open_coding, run_axial_coding, run_selective_coding
from chunking import chunk_transcript, iter_qa_units
from question_sim import build_question_sim_cache
from slot_recursion import resolve_category, empty_slots
from llm_client import call_llm
import time

BASE_DATA_DIR = "data/RelationshipQuality"

def extract_and_chunk_interviews(target_dir, pairs_per_chunk=1, unit=None):
    """Q&A-aware chunking: one interviewer-question + participant-answer per chunk
    (or `pairs_per_chunk` pairs batched). Nothing dropped.

    Returns (chunks, chunk_index):
      chunks       : ordered list of structured chunk dicts (fed to open coding)
      chunk_index  : {chunk_id: chunk_dict} lookup, the backing store for the
                     empty-slot escalation ladder (re-pass a participant's
                     interview; retrieve similar Q&A). Persisted per run so
                     re-passes are reconstructable and auditable.
    """
    pdf_files = glob.glob(os.path.join(target_dir, "**", "*.pdf"), recursive=True)
    chunks = []
    
    for file_path in sorted(pdf_files):
        print(f"    Reading: {os.path.basename(file_path)}")
        chunks.extend(chunk_transcript(file_path, pairs_per_chunk=pairs_per_chunk, unit=unit))

    chunk_index = {c["chunk_id"]: c for c in chunks}
    if len(chunk_index) != len(chunks):
        # chunk_ids collide only if two PDFs share a stem; surface it loudly.
        raise ValueError(
            f"Duplicate chunk_id detected in {target_dir}: "
            f"{len(chunks)} chunks but {len(chunk_index)} unique ids. "
            "Two source PDFs likely share a filename stem."
        )
    return chunks, chunk_index


def build_qa_index(country_dir, unit=None):
    """QA-unit list (one question each) for the question-similarity cache.

    Built from iter_qa_units, NOT the coding chunks, so retrieval grain is
    independent of pairs_per_chunk.
    """
    qa_units = []
    for pdf in sorted(glob.glob(os.path.join(country_dir, "**", "*.pdf"), recursive=True)):
        qa_units.extend(iter_qa_units(pdf, unit=unit))
    return qa_units


def run_slot_recursion(axial_relations, open_codes, chunk_index, qsim, model_type):
    """Straussian empty-slot escalation over every axial category.

    Returns (resolved_categories, traces). Categories with no empty slots pass
    through untouched (their trace notes 'no empty slots'). Only meaningful for
    the Straussian paradigm model; Charmaz focused coding has no slots.
    """
    if not isinstance(axial_relations, list):
        print("  -> Axial output is not a list (parse failure upstream); skipping recursion.")
        return axial_relations, []

    # seam 1: open_code string -> its dict (carries chunk_id)
    open_code_lookup = {
        oc.get("open_code"): oc
        for oc in open_codes
        if isinstance(oc, dict) and "open_code" in oc and "__status__" not in oc
    }

    # seam 2: participant source_id -> full interview text (chunks in q_index order)
    by_source = {}
    for c in chunk_index.values():
        by_source.setdefault(c["source_id"], []).append(c)
    for src in by_source:
        by_source[src].sort(key=lambda c: c.get("q_index", 0))
    def interview_text_for_source(src):
        return "\n\n".join(c["text"] for c in by_source.get(src, []))

    # LLM binding: (system, user) -> str, model fixed
    llm = lambda sp, ut: call_llm(sp, ut, model_type)

    resolved, traces = [], []
    n_with_empties = sum(1 for cat in axial_relations
                         if isinstance(cat, dict) and empty_slots(cat))
    print(f"  -> {n_with_empties}/{len(axial_relations)} categories have empty paradigm slots.")

    for i, cat in enumerate(axial_relations):
        if not isinstance(cat, dict):
            resolved.append(cat)
            continue
        empties = empty_slots(cat)
        if empties:
            print(f"  -> Category {i+1}/{len(axial_relations)} "
                  f"'{cat.get('axial_category','?')}' empty: {empties}")
        out = resolve_category(
            cat, open_code_lookup, chunk_index, qsim, llm, interview_text_for_source
        )
        trace = out.pop("__slot_trace__", None)
        resolved.append(out)
        traces.append({"axial_category": cat.get("axial_category"),
                       "index": i, "trace": trace})
    return resolved, traces

def main():
    # Set to "proprietary" if you want to use OpenAI directly
    MODEL_TO_USE = "proprietary" 
    # Which GT tradition drives the prompts + whether slot recursion runs.
    # "straussian" -> open/axial/selective + paradigm-slot escalation ladder
    # "charmaz"    -> initial/focused/theoretical, NO slot recursion (no paradigm)
    TRADITION = "straussian"
    
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
        output_dir = os.path.join(BASE_DATA_DIR, f"output_{run_name}_{TRADITION}_{MODEL_TO_USE}_{timecode}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=======================================================")
        print(f"🚀 STARTING EXPERIMENT FOR: {run_name}")
        print(f"=======================================================")

        # 2. Extract Data
        print("=== Phase 0: Data Extraction ===")
        # `unit` (country code) tags every chunk for later cross-country work;
        # derive it from the leaf country folder name.
        unit = os.path.basename(folder_name.split("/")[0]).replace("Silan-Ciruelas_", "")
        chunks, chunk_index = extract_and_chunk_interviews(country_dir, unit=unit)
        
        if not chunks:
            print(f"No text extracted for {run_name}. Skipping...\n")
            continue
            
        print(f"Extracted {len(chunks)} text chunks from PDFs.\n")

        # Persist the chunk index: backing store for the empty-slot escalation
        # ladder and an audit trail of exactly what was fed to open coding.
        index_out_path = os.path.join(output_dir, "chunk_index.json")
        with open(index_out_path, "w") as f:
            json.dump(chunk_index, f, indent=4)
        print(f"Saved -> {index_out_path}\n")

        # Build the question-similarity cache (Straussian rung-2 retrieval).
        # Grain = qa_units, independent of chunking. Skipped for Charmaz.
        qsim = None
        if TRADITION == "straussian":
            print("Building question-similarity index...")
            qa_units = build_qa_index(country_dir, unit=unit)
            qsim = build_question_sim_cache(qa_units)
            qsim.save(os.path.join(output_dir, "question_sim_cache.pkl"))
            print(f"  {qsim.summary()}\n")

        # 3. Open Coding
        print("=== Phase 1: Open Coding ===")
        open_codes = run_open_coding(chunks, MODEL_TO_USE, tradition=TRADITION)
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
        axial_relations = run_axial_coding(open_codes, MODEL_TO_USE, tradition=TRADITION)
        axial_out_path = os.path.join(output_dir, "output_axial_codes.json")
        
        with open(axial_out_path, "w") as f:
            json.dump(axial_relations, f, indent=4)
        print(f"Saved -> {axial_out_path}\n")

        # 4b. Empty-slot escalation (Straussian paradigm recall only)
        if TRADITION == "straussian" and qsim is not None:
            print("=== Phase 2b: Empty-Slot Escalation (paradigm recall) ===")
            axial_relations, slot_traces = run_slot_recursion(
                axial_relations, open_codes, chunk_index, qsim, MODEL_TO_USE
            )
            resolved_out_path = os.path.join(output_dir, "output_axial_codes_resolved.json")
            with open(resolved_out_path, "w") as f:
                json.dump(axial_relations, f, indent=4)
            trace_out_path = os.path.join(output_dir, "output_slot_traces.json")
            with open(trace_out_path, "w") as f:
                json.dump(slot_traces, f, indent=4)
            print(f"Saved -> {resolved_out_path}")
            print(f"Saved -> {trace_out_path}\n")

        # 5. Selective Coding (Still saves as Markdown)
        print("=== Phase 3: Selective Coding ===")
        final_theory = run_selective_coding(axial_relations, MODEL_TO_USE, tradition=TRADITION)
        theory_out_path = os.path.join(output_dir, "output_final_theory.md")
        
        with open(theory_out_path, "w") as f:
            f.write(final_theory)
        print(f"Saved -> {theory_out_path}\n")

if __name__ == "__main__":
    main()