# main.py
import os 
import glob
import json
#from PyPDF2 import PdfReader
from gta_pipeline import run_open_coding, run_axial_coding, run_selective_coding
from gta_pipeline import run_initial_memo, run_advanced_memo, run_memo_sorting
from chunking import chunk_transcript, iter_qa_units
from article_chunking import extract_and_chunk_articles
from question_sim import build_question_sim_cache
from slot_recursion import resolve_category, empty_slots
from charmaz_loop import (
    slice_sources, run_charmaz_loop,
    DEFAULT_SATURATION_THRESHOLD, DEFAULT_MAX_ITERATIONS,
)
from llm_client import call_llm
from prompt_registry import DEFAULT_DATASET
import time

# --- Charmaz loop parameters (control-relevant; disclosed) ------------------
# slice_size    : interviews per slice; slice 1 seeds the forward pass, the
#                 rest feed the reflection loop.
# SATURATION_*  : absorbed-fraction stop threshold (tuned like alignment thr).
# MAX_ITER      : hard cap on loop passes.
CHARMAZ_SLICE_SIZE = 5
CHARMAZ_SATURATION_THRESHOLD = DEFAULT_SATURATION_THRESHOLD
CHARMAZ_MAX_ITERATIONS = DEFAULT_MAX_ITERATIONS

# --- Dataset selection (independent of TRADITION, set in main()) -----------
# "silan"   -> (default) Silan-Ciruelas relationship-quality interviews.
#              PDF transcripts under BASE_DATA_DIR, Q&A-aware chunking
#              (chunking.py), question-similarity index + Straussian
#              slot-escalation. Unchanged from before `dataset` existed.
# "semeval" -> SemEval-2025 Task 10 entity-framing articles. Plain-text
#              files under SEMEVAL_BASE_DATA_DIR, whole-article chunking
#              (article_chunking.py). No question-similarity index / slot
#              escalation -- those are Q&A-interview-specific retrieval
#              mechanisms that don't apply to standalone articles; they are
#              skipped for this dataset rather than adapted.
# Only the STUDY CONTEXT the model sees changes between datasets (see
# study_contexts.py / prompt_registry.get_prompts); tradition-specific
# coding logic below is untouched either way.
BASE_DATA_DIR = "data/RelationshipQuality"

# Point this at the root of your SemEval-2025 Task 10 download -- the
# official release layout is:
#   <SEMEVAL_BASE_DATA_DIR>/{train,dev}/labels/<LANG>/subtask-{1,2,3}-annotations.txt  (GOLD LABELS -- never read)
#   <SEMEVAL_BASE_DATA_DIR>/{train,dev,test}/raw-documents/<LANG>/<article_id>.txt     (articles -- what we chunk)
# `target_folders` below must therefore point at a raw-documents/<LANG>
# subtree, NOT a split root, so the labels/ sibling directory is never even
# globbed. article_chunking.py additionally hard-excludes any "labels"
# directory component or "annotation"-like filename as defense in depth, in
# case target_folders is ever pointed too high up the tree by mistake.
SEMEVAL_BASE_DATA_DIR = "data/dataset"

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
    #TRADITION = "straussian"
    TRADITION = "charmaz"

    # Which STUDY CONTEXT the prompts are focused on. Defaults to "silan" --
    # unchanged behavior from before this existed. Set to "semeval" to run
    # the same TRADITION over the SemEval-2025 Task 10 article corpus instead.
    # DATASET = DEFAULT_DATASET  # "silan"
    DATASET = "semeval"

    if DATASET == "silan":
        print('Select silian as data')
        base_dir = BASE_DATA_DIR
        target_folders = [
            #"Silan-Ciruelas_BRA",
            #"Silan-Ciruelas_FRA",
            #"Silan-Ciruelas_PHL",
            #"Silan-Ciruelas_TUR",
            #"Silan-Ciruelas_USA"
            "Silan-Ciruelas_USA/Silan-Ciruelas_USA_Opt1"
            #"Silan-Ciruelas_USA/Silan-Ciruelas_USA_Opt1and2"
        ]
    elif DATASET == "semeval":
        print('Select Semeval as data')
        base_dir = SEMEVAL_BASE_DATA_DIR
        # Official release layout: <split>/raw-documents/<LANG>. Pick the
        # split (train/dev/test) and language(s) to run. NEVER point this at
        # a split root (e.g. "train") -- that would sit "labels/" and
        # "raw-documents/" side by side and, combined with the recursive
        # glob, risk sweeping gold annotation files in as if they were
        # articles (article_chunking.py excludes them defensively too, but
        # don't rely on that as the only safeguard).
        target_folders = [
            "train/raw-documents/EN",
            #"train/raw-documents/BG",
            #"train/raw-documents/HI",
            #"train/raw-documents/PT",
            #"train/raw-documents/RU",
            #"dev/raw-documents/EN",
            #"test/raw-documents/EN",   # test has no labels/ at all
        ]
    else:
        raise ValueError(f"unknown DATASET {DATASET!r}; expected 'silan' or 'semeval'")

    for folder_name in target_folders:
        # 1. Setup paths
        country_dir = os.path.join(base_dir, folder_name)

        # Create a safe name for folders (e.g., replaces slashes)
        run_name = folder_name.replace("/", "_")

        # Create a dedicated output directory for this specific run. Naming
        # is unchanged from before `dataset` existed when DATASET is the
        # default ("silan"); the dataset tag is only added to the folder
        # name for non-default datasets, so existing output-path conventions
        # for Silan runs are preserved exactly.
        timecode = time.strftime("%Y%m%d%H%M%S")
        print(f'starting at {timecode}')
        if DATASET == DEFAULT_DATASET:
            output_dir = os.path.join(base_dir, f"output_{run_name}_{TRADITION}_{MODEL_TO_USE}_{timecode}")
        else:
            output_dir = os.path.join(base_dir, f"output_{run_name}_{TRADITION}_{DATASET}_{MODEL_TO_USE}_{timecode}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n=======================================================")
        if DATASET == DEFAULT_DATASET:
            print(f"🚀 STARTING EXPERIMENT FOR: {run_name}")
        else:
            print(f"🚀 STARTING EXPERIMENT FOR: {run_name} (dataset={DATASET})")
        print(f"=======================================================")

        # 2. Extract Data
        print("=== Phase 0: Data Extraction ===")
        if DATASET == "silan":
            # `unit` (country code) tags every chunk for later cross-country
            # work; derive it from the leaf country folder name.
            unit = os.path.basename(folder_name.split("/")[0]).replace("Silan-Ciruelas_", "")
            chunks, chunk_index = extract_and_chunk_interviews(country_dir, unit=unit)
            print(f"Extracted {len(chunks)} text chunks from PDFs.\n")
        else:  # semeval
            # `unit` (language code) tags every chunk, mirroring the Silan
            # country-code convention; derive it from the leaf language
            # folder name (target_folders entries look like
            # "train/raw-documents/EN" -> unit = "EN").
            unit = os.path.basename(folder_name.rstrip("/"))
            chunks, chunk_index = extract_and_chunk_articles(country_dir, unit=unit)
            print(f"Extracted {len(chunks)} article chunks.\n")

        if not chunks:
            print(f"No text extracted for {run_name}. Skipping...\n")
            continue

        # Persist the chunk index: backing store for the empty-slot escalation
        # ladder and an audit trail of exactly what was fed to open coding.
        index_out_path = os.path.join(output_dir, "chunk_index.json")
        with open(index_out_path, "w") as f:
            json.dump(chunk_index, f, indent=4)
        print(f"Saved -> {index_out_path}\n")

        # Build the question-similarity cache (Straussian rung-2 retrieval).
        # Grain = qa_units, independent of chunking. Q&A-interview-specific,
        # so only built for the "silan" dataset; skipped for Charmaz and for
        # "semeval" (whole-article chunks have no question/answer structure
        # to index, so slot escalation is skipped too -- run_straussian_arm
        # already no-ops the escalation step whenever qsim is None).
        qsim = None
        if DATASET == "silan" and TRADITION == "straussian":
            print("Building question-similarity index...")
            qa_units = build_qa_index(country_dir, unit=unit)
            qsim = build_question_sim_cache(qa_units)
            qsim.save(os.path.join(output_dir, "question_sim_cache.pkl"))
            print(f"  {qsim.summary()}\n")

        # The coding phases branch by tradition. The Straussian block is
        # unchanged from prior versions (open→axial→slot-escalation→selective).
        # The Charmaz block runs its own slice-driven constructivist flow
        # (initial+memo on slice 1 → focused+memo → reflection loop over the
        # remaining slices → memo-sorting integration).
        if TRADITION == "straussian":
            run_straussian_arm(chunks, chunk_index=chunk_index,
                               qsim=qsim, output_dir=output_dir, model=MODEL_TO_USE,
                               dataset=DATASET)
        elif TRADITION == "charmaz":
            run_charmaz_arm(chunks, chunk_index=chunk_index, output_dir=output_dir,
                            model=MODEL_TO_USE, dataset=DATASET)
        else:
            raise ValueError(f"unknown TRADITION {TRADITION!r}")


def run_straussian_arm(chunks, chunk_index, qsim, output_dir, model, dataset=DEFAULT_DATASET):
    """Straussian open→axial→slot-escalation→selective. Behavior preserved
    verbatim from the single-tradition version for dataset="silan" (the
    default); only lifted into a function so the Charmaz arm can sit beside
    it without touching this path. `dataset` is forwarded to every
    run_*_coding call, selecting which STUDY CONTEXT the prompts use -- the
    coding logic here (paradigm slots, JSON contract) does not change."""
    TRADITION = "straussian"
    MODEL_TO_USE = model

    # 3. Open Coding
    print("=== Phase 1: Open Coding ===")
    open_codes = run_open_coding(chunks, MODEL_TO_USE, tradition=TRADITION, dataset=dataset)
    open_out_path = os.path.join(output_dir, "output_open_codes.json")

    with open(open_out_path, "w") as f:
        json.dump(open_codes, f, indent=4)
    print(f"Saved -> {open_out_path}\n")

    # 4. Axial Coding (Now correctly saving as JSON)
    print("=== Phase 2: Axial Coding ===")
    axial_relations = run_axial_coding(open_codes, MODEL_TO_USE, tradition=TRADITION, dataset=dataset)
    axial_out_path = os.path.join(output_dir, "output_axial_codes.json")

    with open(axial_out_path, "w") as f:
        json.dump(axial_relations, f, indent=4)
    print(f"Saved -> {axial_out_path}\n")

    # 4b. Empty-slot escalation (Straussian paradigm recall only)
    if qsim is not None:
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
    final_theory = run_selective_coding(axial_relations, MODEL_TO_USE, tradition=TRADITION, dataset=dataset)
    theory_out_path = os.path.join(output_dir, "output_final_theory.md")

    with open(theory_out_path, "w") as f:
        f.write(final_theory)
    print(f"Saved -> {theory_out_path}\n")


def run_charmaz_arm(chunks, chunk_index, output_dir, model, dataset=DEFAULT_DATASET):
    """Charmaz constructivist-GT arm: slice-driven initial→focused→loop→integrate.

    Slice 1 seeds the forward pass (initial coding + initial memo → focused
    coding + advanced memo). The reflection loop then processes the remaining
    slices, testing whether existing focused categories absorb each fresh slice
    and re-sampling (re-coding) where they do not, until saturation / max-iter /
    corpus exhaustion. The final integrated account comes from memo-sorting.

    `dataset` is forwarded to the initial/focused coding calls (registry-backed,
    STUDY-CONTEXT-swappable). NOTE: the memo-writing and memo-sorting steps
    below (run_initial_memo / run_advanced_memo / run_memo_sorting) call
    prompts_charmaz_recursion.py directly, NOT through prompt_registry, and
    that module's prompts still hardcode Silan relationship-quality framing
    (e.g. "...participants' lived sense of relationship quality" in the final
    theoretical-account prompt). That's out of scope for this change -- if you
    run the Charmaz arm to completion on dataset="semeval", expect that
    mismatch in the memo/integration steps specifically; only initial/focused
    coding are dataset-aware so far.
    """
    MODEL_TO_USE = model
    llm = lambda sp, ut: call_llm(sp, ut, MODEL_TO_USE)

    # Partition participants into slices; slice 1 seeds the forward pass.
    all_sources = [c["source_id"] for c in chunks]
    slices = slice_sources(all_sources, CHARMAZ_SLICE_SIZE)
    if not slices:
        print("No participants to code for Charmaz arm. Skipping.\n")
        return
    seed_sources = set(slices[0])
    remaining_slices = slices[1:]
    seed_chunks = [c for c in chunks if c["source_id"] in seed_sources]
    print(f"Charmaz slices: {len(slices)} (slice_size={CHARMAZ_SLICE_SIZE}); "
          f"seed slice has {len(seed_sources)} participant(s), "
          f"{len(remaining_slices)} slice(s) feed the reflection loop.\n")

    # --- Phase 1: Initial Coding (slice 1) ----------------------------------
    print("=== Phase 1: Initial Coding (seed slice) ===")
    initial_codes = run_open_coding(seed_chunks, MODEL_TO_USE, tradition="charmaz", dataset=dataset)
    with open(os.path.join(output_dir, "output_initial_codes.json"), "w") as f:
        json.dump(initial_codes, f, indent=4)
    print(f"Saved -> output_initial_codes.json\n")

    # --- Phase 1b: Initial Memo-Writing -------------------------------------
    print("=== Phase 1b: Initial Memo-Writing ===")
    initial_memo = run_initial_memo(initial_codes, MODEL_TO_USE)
    with open(os.path.join(output_dir, "output_initial_memos.json"), "w") as f:
        json.dump(initial_memo, f, indent=4)
    print(f"Saved -> output_initial_memos.json\n")

    # --- Phase 2: Focused Coding (slice 1) ----------------------------------
    print("=== Phase 2: Focused Coding (seed slice) ===")
    focused = run_axial_coding(initial_codes, MODEL_TO_USE, tradition="charmaz", dataset=dataset)
    with open(os.path.join(output_dir, "output_focused_codes.json"), "w") as f:
        json.dump(focused, f, indent=4)
    print(f"Saved -> output_focused_codes.json\n")

    # Focused output may be a parse-failure dict; normalize to a list.
    focused_categories = focused if isinstance(focused, list) else []

    # --- Phase 2b: Advanced Memo-Writing (names thin areas) -----------------
    print("=== Phase 2b: Advanced Memo-Writing ===")
    advanced_memo = run_advanced_memo(focused_categories, MODEL_TO_USE)
    with open(os.path.join(output_dir, "output_advanced_memos_seed.json"), "w") as f:
        json.dump(advanced_memo, f, indent=4)
    print(f"Saved -> output_advanced_memos_seed.json\n")

    # --- Phase 3: Reflection Loop (theoretical re-sampling over slices) -----
    print("=== Phase 3: Reflection Loop (slice-driven saturation) ===")
    loop = run_charmaz_loop(
        initial_categories=focused_categories,
        initial_advanced_memo=advanced_memo,
        remaining_slices=remaining_slices,
        chunk_index=chunk_index,
        call_llm=llm,
        saturation_threshold=CHARMAZ_SATURATION_THRESHOLD,
        max_iterations=CHARMAZ_MAX_ITERATIONS,
    )
    print(f"  -> stop_reason={loop['stop_reason']!r}, "
          f"saturation_reached={loop['saturation_reached']}, "
          f"iterations={loop['n_iterations']}")

    focused_categories = loop["final_categories"]
    final_advanced_memo = loop["final_advanced_memo"]

    with open(os.path.join(output_dir, "output_focused_codes_final.json"), "w") as f:
        json.dump(focused_categories, f, indent=4)
    with open(os.path.join(output_dir, "output_advanced_memos_final.json"), "w") as f:
        json.dump(final_advanced_memo, f, indent=4)
    # The change-tree is the publishable artifact of the loop.
    with open(os.path.join(output_dir, "output_charmaz_change_tree.json"), "w") as f:
        json.dump({
            "saturation_reached": loop["saturation_reached"],
            "stop_reason": loop["stop_reason"],
            "n_iterations": loop["n_iterations"],
            "slice_size": CHARMAZ_SLICE_SIZE,
            "saturation_threshold": CHARMAZ_SATURATION_THRESHOLD,
            "max_iterations": CHARMAZ_MAX_ITERATIONS,
            "change_tree": loop["change_tree"],
        }, f, indent=4)
    print(f"Saved -> output_focused_codes_final.json")
    print(f"Saved -> output_advanced_memos_final.json")
    print(f"Saved -> output_charmaz_change_tree.json\n")

    # --- Phase 4: Memo Sorting & Integration (theoretical account) ----------
    print("=== Phase 4: Sorting & Integration ===")
    final_theory = run_memo_sorting(focused_categories, final_advanced_memo, MODEL_TO_USE)
    with open(os.path.join(output_dir, "output_final_theory.md"), "w") as f:
        f.write(final_theory or "")
    print(f"Saved -> output_final_theory.md\n")

if __name__ == "__main__":
    main()