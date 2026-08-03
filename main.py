# main.py
import os
import glob
import json
#from PyPDF2 import PdfReader
from gta_pipeline import run_open_coding, run_axial_coding, run_selective_coding
from gta_pipeline import run_initial_memo, run_advanced_memo, run_memo_sorting
from chunking import chunk_transcript, iter_qa_units
from article_chunking import extract_and_chunk_articles
from question_sim import build_question_sim_cache, build_sentence_sim_index
from slot_recursion import resolve_category, empty_slots
from charmaz_loop import (
    slice_sources, run_charmaz_loop,
    DEFAULT_SATURATION_THRESHOLD, DEFAULT_MAX_ITERATIONS,
)
from llm_client import call_llm
from prompt_registry import DEFAULT_DATASET, get_taxonomy_match_prompt, _fill
import time

# --- Charmaz loop parameters (control-relevant; disclosed) ------------------
# slice_size    : interviews per slice; slice 1 seeds the forward pass, the
#                 rest feed the reflection loop.
# SATURATION_*  : absorbed-fraction stop threshold (tuned like alignment thr).
# MAX_ITER      : hard cap on loop passes.
CHARMAZ_SLICE_SIZE = 5
CHARMAZ_SATURATION_THRESHOLD = DEFAULT_SATURATION_THRESHOLD
CHARMAZ_MAX_ITERATIONS = DEFAULT_MAX_ITERATIONS

# --- Straussian escalation-ladder parameters (control-relevant, disclosed) --
# STRAUSSIAN_SKIP_RUNG1: None -> dataset-conditioned default (skipped for
# semeval, run for silan -- see slot_recursion.resolve_category). Set
# True/False explicitly to override for BOTH datasets uniformly (e.g. for a
# rung-1-forced-on SemEval ablation run).
STRAUSSIAN_SKIP_RUNG1 = None

# --- Taxonomy-seeded coding parameters (control-relevant, disclosed) --------
# Additional experiment (see GTA_taxonomy_seeded_experiment.md): SKIP emergent
# axial coding and inject a pre-existing SemEval taxonomy as the fixed axial
# layer, then MATCH the emergent open codes against it and type the edge cases.
# This is orthogonal to TRADITION and to the §14 slot ladder -- when
# SEED_TAXONOMY is on, the seeded-matching arm runs instead of the normal
# tradition arm.
#
# SEED_TAXONOMY : master switch. False -> normal pipeline (unchanged).
# SEED_KIND     : "entity_role" (Subtask 1, domain-agnostic 22 fine sub-roles)
#                 | "narrative" (Subtask 2, domain-scoped sub-narratives). Two
#                 INDEPENDENT experiments; pick one per run.
# SEED_LEVEL    : "fine" (built). "coarse" (roll-up, Experiment E3) is deferred
#                 and raises in taxonomy_registry.load_taxonomy.
# SEED_LABELS_DIR : path to the SemEval labels/<LANG> gold tree, passed
#                 EXPLICITLY and used ONLY for eval scoring (never a prompt).
#                 None -> soft edge-typing only, no hard scoring. Kept separate
#                 from target_folders on purpose so gold can only enter via the
#                 scorer. Leave None unless you are scoring.
SEED_TAXONOMY = True
SEED_KIND = "entity_role"
SEED_LEVEL = "fine"
SEED_LABELS_DIR = None

# --- Dataset selection (independent of TRADITION, set in main()) -----------
# "silan"   -> (default) Silan-Ciruelas relationship-quality interviews.
#              PDF transcripts under BASE_DATA_DIR, Q&A-aware chunking
#              (chunking.py), question-similarity index + Straussian
#              slot-escalation. Unchanged from before `dataset` existed.
# "semeval" -> SemEval-2025 Task 10 entity-framing articles. Plain-text
#              files under SEMEVAL_BASE_DATA_DIR, whole-article chunking
#              (article_chunking.py). Straussian slot escalation now DOES run
#              for this dataset: rung 1 (full-source re-feed) is skipped by
#              default (a chunk already IS the whole article), rung 2 uses a
#              sentence-grain similarity index (build_sentence_sim_index)
#              queried by open-code text_passages and deduped to article
#              grain -- see question_sim.py / slot_recursion.py.
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


def run_slot_recursion(axial_relations, open_codes, chunk_index, qsim, model_type,
                        dataset=DEFAULT_DATASET, skip_rung1=None):
    """Straussian empty-slot escalation over every axial category.

    Returns (resolved_categories, traces). Categories with no empty slots pass
    through untouched (their trace notes 'no empty slots'). Only meaningful for
    the Straussian paradigm model; Charmaz focused coding has no slots.

    `dataset` is forwarded to resolve_category (see slot_recursion.py) and
    branches both the rung-1 default (skipped for semeval) and the rung-2
    retrieval mechanism (cross-participant Q&A for silan, cross-article
    sentence-grain retrieval for semeval). `skip_rung1` is forwarded as-is
    (None -> dataset-conditioned default; see resolve_category).
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
            cat, open_code_lookup, chunk_index, qsim, llm, interview_text_for_source,
            dataset=dataset, skip_rung1=skip_rung1,
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
    #TRADITION = "charmaz"

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
            "dev/raw-documents/EN",
            #"dev/raw-documents/BG",
            #"dev/raw-documents/HI",
            #"dev/raw-documents/PT",
            #"dev/raw-documents/RU",
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
            # "dev/raw-documents/EN" -> unit = "EN").
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

        # Build the question/sentence-similarity cache (Straussian rung-2
        # retrieval). Only built for TRADITION=="straussian" (Charmaz has no
        # paradigm slots to escalate) AND when NOT running the seeded-taxonomy
        # arm (the seeded arm skips emergent axial + slot escalation, so it
        # never needs qsim). Grain differs by dataset:
        #   "silan"   -> qa_units grain (one row per interviewer question),
        #                independent of chunking; retrieval is cross-
        #                participant Q&A (top_k_other_participant).
        #   "semeval" -> sentence grain (article chunks have no Q&A structure
        #                to index), built directly from `chunks`; retrieval
        #                is cross-article via text_passage queries
        #                (top_k_other_source_by_text), deduped to article
        #                grain. See question_sim.build_sentence_sim_index and
        #                slot_recursion.resolve_category.
        qsim = None
        if TRADITION == "straussian" and not SEED_TAXONOMY:
            print("Building question/sentence-similarity index...")
            if DATASET == "silan":
                qa_units = build_qa_index(country_dir, unit=unit)
                qsim = build_question_sim_cache(qa_units)
            elif DATASET == "semeval":
                qsim = build_sentence_sim_index(chunks)
            qsim.save(os.path.join(output_dir, "question_sim_cache.pkl"))
            print(f"  {qsim.summary()}\n")

        # The coding phases branch by tradition. The Straussian block is
        # unchanged from prior versions (open→axial→slot-escalation→selective).
        # The Charmaz block runs its own slice-driven constructivist flow
        # (initial+memo on slice 1 → focused+memo → reflection loop over the
        # remaining slices → memo-sorting integration).
        if SEED_TAXONOMY:
            # Taxonomy-seeded arm: open coding -> seed the fixed axial layer
            # -> match open codes against it -> type edge cases (+ optional
            # eval scoring). Runs INSTEAD of the tradition arm; TRADITION is
            # ignored on this path (open coding still uses it for its prompt,
            # defaulting to straussian).
            run_seeded_taxonomy_arm(
                chunks, chunk_index=chunk_index, output_dir=output_dir,
                model=MODEL_TO_USE, dataset=DATASET,
                seed_kind=SEED_KIND, seed_level=SEED_LEVEL,
                labels_dir=SEED_LABELS_DIR,
                open_coding_tradition=("charmaz" if TRADITION == "charmaz" else "straussian"),
            )
        elif TRADITION == "straussian":
            run_straussian_arm(chunks, chunk_index=chunk_index,
                               qsim=qsim, output_dir=output_dir, model=MODEL_TO_USE,
                               dataset=DATASET, skip_rung1=STRAUSSIAN_SKIP_RUNG1)
        elif TRADITION == "charmaz":
            run_charmaz_arm(chunks, chunk_index=chunk_index, output_dir=output_dir,
                            model=MODEL_TO_USE, dataset=DATASET)
        else:
            raise ValueError(f"unknown TRADITION {TRADITION!r}")


def run_straussian_arm(chunks, chunk_index, qsim, output_dir, model,
                        dataset=DEFAULT_DATASET, skip_rung1=None):
    """Straussian open→axial→slot-escalation→selective. Behavior preserved
    verbatim from the single-tradition version for dataset="silan" (the
    default); only lifted into a function so the Charmaz arm can sit beside
    it without touching this path. `dataset` is forwarded to every
    run_*_coding call, selecting which STUDY CONTEXT the prompts use -- the
    coding logic here (paradigm slots, JSON contract) does not change.

    Empty-slot escalation (Phase 2b) now runs for BOTH datasets when qsim is
    not None: silan via cross-participant Q&A retrieval (unchanged), semeval
    via cross-article sentence-grain retrieval, with rung 1 (full-source
    re-feed) skipped by default for semeval -- see slot_recursion.py.
    `skip_rung1` (None -> dataset-conditioned default) is forwarded to
    run_slot_recursion -> resolve_category."""
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
            axial_relations, open_codes, chunk_index, qsim, MODEL_TO_USE,
            dataset=dataset, skip_rung1=skip_rung1,
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


def run_seeded_taxonomy_arm(chunks, chunk_index, output_dir, model,
                            dataset, seed_kind, seed_level="fine",
                            labels_dir=None, open_coding_tradition="straussian"):
    """Taxonomy-seeded coding arm (additional experiment).

    Flow: open coding (unchanged) -> SKIP emergent axial coding -> inject the
    pre-existing SemEval taxonomy as the fixed axial layer -> match each open
    code against it with BOTH a deterministic embedding matcher and an
    interpretive LLM matcher -> type the edge cases -> optionally hard-score
    against the gold key (eval-only). See GTA_taxonomy_seeded_experiment.md.

    Two independent experiments select via seed_kind:
      "entity_role" -> Subtask 1, domain-agnostic 22 fine sub-roles.
      "narrative"   -> Subtask 2, domain-scoped sub-narratives (per-article
                       domain resolved from the filename via domain_of).

    FIREWALL: only the taxonomy SCHEMA reaches the model (via the match prompt's
    {taxonomy_block}); the gold answer key (labels_dir) is loaded ONLY here for
    scoring and never enters a prompt.
    """
    import taxonomy_registry as TR
    import taxonomy_match as TM

    MODEL_TO_USE = model
    llm = lambda sp, ut: call_llm(sp, ut, MODEL_TO_USE)

    # 1. Open Coding (unchanged; still emergent)
    print("=== Phase 1: Open Coding (seeded-taxonomy experiment) ===")
    open_codes = run_open_coding(chunks, MODEL_TO_USE,
                                 tradition=open_coding_tradition, dataset=dataset)
    with open(os.path.join(output_dir, "output_open_codes.json"), "w") as f:
        json.dump(open_codes, f, indent=4)
    print(f"Saved -> output_open_codes.json\n")

    # 2. Seed the axial layer (fine). For entity_role: one domain-agnostic
    #    taxonomy for all codes. For narrative: domain-scoped, so we partition
    #    the open codes by their article's domain and match each partition
    #    against that domain's tree.
    print(f"=== Phase 2: Seeded Axial Layer ({seed_kind}, level={seed_level}) ===")
    match_prompt = get_taxonomy_match_prompt(dataset).match

    def make_render_prompt(taxonomy):
        # Build the (system, user) closure taxonomy_match.llm_match expects.
        block = taxonomy.taxonomy_block()  # SCHEMA ONLY -- never gold
        if taxonomy.domain:
            domain_note = (f"This article is from the {taxonomy.domain} "
                           f"('{'Ukraine-Russia War' if taxonomy.domain=='URW' else 'Climate Change'}') domain. ")
        else:
            domain_note = ""
        def render(open_code, _tax):
            system = _fill(match_prompt, domain_note=domain_note, taxonomy_block=block)
            user = _fill(
                "Code: {open_code}\nEvidence passage: {text_passage}",
                open_code=str(open_code.get("open_code", "")),
                text_passage=str(open_code.get("text_passage", "")),
            )
            return system, user
        return render

    if seed_kind == "narrative":
        # partition codes by article domain
        buckets = {"URW": [], "CC": [], None: []}
        for oc in TM.iter_valid_open_codes(open_codes):
            sid = oc.get("source_id") or (oc.get("chunk_id", "").rsplit("_c", 1)[0])
            dom = TR.domain_of(sid)
            buckets.setdefault(dom, []).append(oc)
        if buckets[None]:
            print(f"  -> WARNING: {len(buckets[None])} open code(s) from articles "
                  f"whose domain could not be parsed from the filename; skipped "
                  f"from narrative matching (they need a domain to be scoped).")
        typed_all = []
        seed_axial_all = []
        for dom in ("URW", "CC"):
            if not buckets[dom]:
                continue
            taxonomy = TR.load_taxonomy(dataset, "narrative", level=seed_level, domain=dom)
            seed_axial_all.extend(taxonomy.as_axial_relations())
            typed_all.extend(_match_and_type(buckets[dom], taxonomy, llm,
                                             make_render_prompt(taxonomy), TM))
        typed = typed_all
        # category signals need a per-domain view; compute per tree and merge.
        cat_sig = {"per_domain": {}}
        for dom in ("URW", "CC"):
            if not buckets[dom]:
                continue
            taxonomy = TR.load_taxonomy(dataset, "narrative", level=seed_level, domain=dom)
            dom_typed = [t for t in typed if (t.get("emb_category_id") or "").startswith(dom + "/")]
            cat_sig["per_domain"][dom] = TM.category_signals(dom_typed, taxonomy)
        taxonomy_for_gold = None  # narrative gold scored article-level, tree-agnostic
    else:
        taxonomy = TR.load_taxonomy(dataset, seed_kind, level=seed_level, domain=None)
        seed_axial_all = taxonomy.as_axial_relations()
        typed = _match_and_type(TM.iter_valid_open_codes(open_codes), taxonomy, llm,
                                make_render_prompt(taxonomy), TM)
        cat_sig = TM.category_signals(typed, taxonomy)
        taxonomy_for_gold = taxonomy

    # persist the seeded axial layer (format-identical to emergent axial output)
    with open(os.path.join(output_dir, "output_axial_codes_seeded.json"), "w") as f:
        json.dump(seed_axial_all, f, indent=4)

    # 3. Persist the assignment table (pre-populated manual-typing sheet) + signals
    rows = TM.assignment_rows(typed)
    with open(os.path.join(output_dir, "output_taxonomy_assignments.json"), "w") as f:
        json.dump(rows, f, indent=4)
    _write_csv(os.path.join(output_dir, "output_taxonomy_assignments.csv"), rows)
    with open(os.path.join(output_dir, "output_taxonomy_category_signals.json"), "w") as f:
        json.dump(cat_sig, f, indent=4, default=list)
    n_edge = sum(1 for t in typed if t.get("is_edge_case"))
    print(f"  -> matched {len(typed)} open codes; {n_edge} flagged as edge cases.")
    print(f"Saved -> output_taxonomy_assignments.(json|csv)")
    print(f"Saved -> output_taxonomy_category_signals.json\n")

    # 4. Optional eval scoring (EVAL-ONLY; gold never entered a prompt above)
    if labels_dir:
        print("=== Phase 3: Gold Scoring (eval-only) ===")
        gold = TR.load_gold(dataset, seed_kind, labels_dir)
        report = TM.score_against_gold(typed, gold, seed_kind)  # entity_role -> None w/o entity resolution
        if report is not None:
            with open(os.path.join(output_dir, "output_taxonomy_gold_score.json"), "w") as f:
                json.dump(report, f, indent=4)
            print(f"Saved -> output_taxonomy_gold_score.json\n")


def _match_and_type(codes, taxonomy, llm, render_prompt, TM):
    """Run both matchers over a code list against one taxonomy and type edges."""
    emb = TM.embed_match(codes, taxonomy)
    # align LLM results to the SAME valid-code ordering embed_match used
    valid = TM.iter_valid_open_codes(codes)
    llm_results = [TM.llm_match(oc, taxonomy, render_prompt, llm) for oc in valid]
    return TM.classify_edge_cases(emb, llm_results, taxonomy=taxonomy)


def _write_csv(path, rows):
    import csv
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_charmaz_arm(chunks, chunk_index, output_dir, model, dataset=DEFAULT_DATASET):
    """Charmaz constructivist-GT arm: slice-driven initial→focused→loop→integrate.

    Slice 1 seeds the forward pass (initial coding + initial memo → focused
    coding + advanced memo). The reflection loop then processes the remaining
    slices, testing whether existing focused categories absorb each fresh slice
    and re-sampling (re-coding) where they do not, until saturation / max-iter /
    corpus exhaustion. The final integrated account comes from memo-sorting.

    `dataset` is forwarded to the initial/focused coding calls AND to the
    memo-writing / reflection-loop / memo-sorting steps below (run_initial_memo
    / run_advanced_memo / run_charmaz_loop / run_memo_sorting). All of these
    now route through prompt_registry.get_charmaz_recursion_prompts, which
    assembles each prompt from the dataset axis (DATASET_BLOCKS) at once --
    this closes a confirmed leak: a real semeval run's output_final_theory.md
    once came back with literal "relationship quality" / "participants
    construct" language baked into the memo-sorting prompt's closing sentence,
    because run_memo_sorting didn't accept a `dataset` argument at all
    originally.
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
    initial_memo = run_initial_memo(initial_codes, MODEL_TO_USE, dataset=dataset)
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
    advanced_memo = run_advanced_memo(focused_categories, MODEL_TO_USE, dataset=dataset)
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
        dataset=dataset,
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
    final_theory = run_memo_sorting(focused_categories, final_advanced_memo, MODEL_TO_USE, dataset=dataset)
    with open(os.path.join(output_dir, "output_final_theory.md"), "w") as f:
        f.write(final_theory or "")
    print(f"Saved -> output_final_theory.md\n")

if __name__ == "__main__":
    main()