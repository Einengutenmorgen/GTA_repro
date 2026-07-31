# slot_recursion.py
"""
Straussian empty-slot escalation ladder (paradigm recall).

For an axial category with an empty paradigm slot (condition / action_interaction
/ consequence), recover the missing slot by widening the evidence net, target
call by target call, and stop as soon as it's filled — or leave it honestly
empty after exhaustion.

Ladder
------
  Rung 1  full interviews of CONTRIBUTING participants
          - one EXTRACT call per contributing interview (all empty slots batched)
          - one AGGREGATE_RESOLVE call over the pooled extractions
          - SILAN: runs by default.
          - SEMEVAL: SKIPPED by default (skip_rung1, resolved dataset-
            conditionally in resolve_category). A SemEval chunk already IS
            the full article -- open coding already saw it whole, so
            "re-feed the full source" recovers no new information the way it
            does for a Silan chunk (a lossy slice of an interview). Can be
            forced back on via skip_rung1=False for an ablation run.
  Rung 2+ cross-source similar evidence (widening k = 3 -> 4 -> 5), one
          CROSS_RESOLVE call per k, over newly-retrieved evidence only.
          - SILAN: query = the qa_units (questions) behind this category's
            supporting open codes; retrieval is cross-participant Q&A via
            QuestionSimCache.top_k_other_participant, at Q&A-unit grain.
          - SEMEVAL: query = the exact quoted text_passages of this
            category's supporting open codes (every open code already
            carries one); retrieval is cross-article via
            QuestionSimCache.top_k_other_source_by_text, built at
            one-row-per-SENTENCE grain (no Q&A structure to index) but
            deduped and returned at the ARTICLE grain, with the full article
            text resolved via chunk_index -- same principle Silan already
            uses (_format_retrieved_qa pulls the full chunk text, not the
            bare question/sentence).
  Terminal leave any still-empty slot as "" (honest empty)

Slot-emptiness is checked PROGRAMMATICALLY (deterministic, reproducible).
Every rung is logged into a trace so the escalation is fully reconstructable —
the trace is the publishable artifact of how each fill was reached.

Nothing here re-embeds or re-chunks; it consumes the persisted chunk_index and
a prebuilt QuestionSimCache (Q&A-grain for Silan, sentence-grain for SemEval,
built by question_sim.build_question_sim_cache / build_sentence_sim_index
respectively).
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

from prompt_registry import (
    SLOT_QUESTIONS,
    get_straussian_slot_prompts,
    _fill,
    DEFAULT_DATASET,
)

PARADIGM_SLOTS = ("condition", "action_interaction", "consequence")

START_K = 3
MAX_K = 5


# ---------------------------------------------------------------------------
# Programmatic slot checking (deterministic)
# ---------------------------------------------------------------------------

def empty_slots(category: dict) -> List[str]:
    """Return paradigm slots that are empty/missing. Pure, deterministic."""
    out = []
    for slot in PARADIGM_SLOTS:
        v = category.get(slot)
        if v is None or (isinstance(v, str) and not v.strip()):
            out.append(slot)
    return out


# ---------------------------------------------------------------------------
# Context builders (pure string assembly)
# ---------------------------------------------------------------------------

def _filled_slots_context(category: dict, empties: List[str]) -> str:
    filled = [s for s in PARADIGM_SLOTS if s not in empties]
    if not filled:
        return "  (none established yet)"
    return "\n".join(f"  - {s}: {category.get(s)}" for s in filled)


def _empty_slot_questions(empties: List[str]) -> str:
    return "\n".join(f"  - {s}: {SLOT_QUESTIONS[s]}" for s in empties)


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------

def resolve_category(
    category: dict,
    open_code_lookup: Dict[str, dict],
    chunk_index: Dict[str, dict],
    qsim,
    call_llm: Callable[[str, str], str],
    interview_text_for_source: Callable[[str], str],
    start_k: int = START_K,
    max_k: int = MAX_K,
    dataset: str = DEFAULT_DATASET,
    skip_rung1: Optional[bool] = None,
) -> dict:
    """Fill empty paradigm slots for one axial category via the ladder.

    Parameters
    ----------
    category : the axial category dict (mutated copy is returned).
    open_code_lookup : {open_code_string -> open_code_dict} so a category's
        supporting_open_codes resolve to chunk_ids (and thence participants).
    chunk_index : {chunk_id -> chunk dict} (persisted in Phase 0).
    qsim : QuestionSimCache for cross-participant/cross-article retrieval.
        For dataset=="silan" this is a Q&A-grain cache built by
        build_question_sim_cache; for dataset=="semeval" it is a
        sentence-grain cache built by build_sentence_sim_index.
    call_llm : (system_prompt, user_text) -> raw string. Caller binds model.
    interview_text_for_source : source_id -> full interview text (caller builds
        this from chunk_index, concatenating that participant's chunks in order).
        Only used when rung 1 runs.
    start_k, max_k : widening bounds for rung 2+.
    dataset : forwarded to the EXTRACT / AGGREGATE_RESOLVE / CROSS_RESOLVE
        prompts via prompt_registry.get_straussian_slot_prompts, and branches
        rung-2 query construction/retrieval (see below).
    skip_rung1 : whether to skip the full-source re-feed rung. None (default)
        means dataset-conditioned: skipped for "semeval" (a SemEval chunk
        already IS the full article -- open coding already saw it whole, so
        re-feeding it recovers nothing new the way re-feeding a lossy Silan
        interview slice does), run for "silan" (unchanged). Pass True/False
        explicitly to override for either dataset (e.g. a SemEval
        rung-1-forced-on ablation run).

    Returns a NEW category dict with slots filled where possible and a
    "__slot_trace__" key logging every rung. Unresolved slots stay "".
    """
    if skip_rung1 is None:
        skip_rung1 = (dataset == "semeval")

    cat = dict(category)  # don't mutate caller's object
    empties = empty_slots(cat)
    trace: List[dict] = []

    if not empties:
        cat["__slot_trace__"] = [{"rung": "none", "note": "no empty slots"}]
        return cat

    # Resolve contributing participants from supporting_open_codes -> chunk_id -> source_id.
    # Always computed regardless of skip_rung1: still needed for rung-2 SemEval
    # cross-article retrieval exclusion (a category's own contributing sources
    # must never be retrieved back as "new" evidence for itself).
    contributing_sources: List[str] = []
    seen_src = set()
    for oc in cat.get("supporting_open_codes", []):
        rec = open_code_lookup.get(oc)
        if not rec:
            continue
        cid = rec.get("chunk_id")
        chunk = chunk_index.get(cid) if cid else None
        if chunk:
            src = chunk["source_id"]
            if src not in seen_src:
                seen_src.add(src)
                contributing_sources.append(src)

    # ---- RUNG 1: per-interview extraction -> aggregate resolve --------------
    filled_ctx = _filled_slots_context(cat, empties)
    empty_q = _empty_slot_questions(empties)

    R = get_straussian_slot_prompts(dataset)

    if not skip_rung1:
        extractions: List[dict] = []
        for src in contributing_sources:
            interview = interview_text_for_source(src)
            prompt = _fill(
                R.extract,
                axial_category=cat.get("axial_category", ""),
                reasoning=cat.get("reasoning", ""),
                filled_slots_context=filled_ctx,
                empty_slot_questions=empty_q,
            )
            raw = call_llm(prompt, interview)
            parsed = _safe_json(raw)
            extractions.append({"source_id": src, "raw": raw, "parsed": parsed})

        trace.append({
            "rung": "1_extract",
            "contributing_sources": contributing_sources,
            "n_calls": len(contributing_sources),
            "extractions": extractions,
        })

        # aggregate-resolve over pooled evidence
        aggregated = _format_aggregated_evidence(extractions, empties)
        agg_prompt = _fill(
            R.aggregate_resolve,
            axial_category=cat.get("axial_category", ""),
            filled_slots_context=filled_ctx,
            empty_slot_questions=empty_q,
            aggregated_evidence=aggregated,
        )
        agg_raw = call_llm(agg_prompt, "")
        agg_parsed = _safe_json(agg_raw)
        applied = _apply_resolutions(cat, agg_parsed, empties)
        trace.append({
            "rung": "1_aggregate_resolve",
            "raw": agg_raw,
            "parsed": agg_parsed,
            "slots_filled": applied,
        })

        empties = empty_slots(cat)
        if not empties:
            cat["__slot_trace__"] = trace
            return cat
    else:
        trace.append({"rung": "1_extract", "note": "skipped (skip_rung1=True)"})
        # empties stays as originally computed -- don't recompute from an
        # empty trace.

    # ---- RUNG 2+: cross-participant / cross-article widening ---------------
    if dataset == "silan":
        # query = the questions behind this category's supporting open codes
        query_chunk_ids = []
        for oc in cat.get("supporting_open_codes", []):
            rec = open_code_lookup.get(oc)
            if rec and rec.get("chunk_id"):
                # only chunk_ids that exist in the question index are valid queries
                try:
                    qsim._row(rec["chunk_id"])
                    query_chunk_ids.append(rec["chunk_id"])
                except KeyError:
                    pass
        query_chunk_ids = list(dict.fromkeys(query_chunk_ids))  # dedupe, keep order
    elif dataset == "semeval":
        # query = the exact quoted text_passages of this category's
        # supporting open codes (every open code, both traditions/datasets,
        # already carries one -- no new extraction needed).
        query_texts = []
        seen_texts = set()
        for oc in cat.get("supporting_open_codes", []):
            rec = open_code_lookup.get(oc)
            passage = rec.get("text_passage", "").strip() if rec else ""
            if passage and passage not in seen_texts:
                seen_texts.add(passage)
                query_texts.append(passage)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected 'silan' or 'semeval'")

    prev_ids: set = set()
    for k in range(start_k, max_k + 1):
        if dataset == "silan":
            hits = qsim.top_k_other_participant(query_chunk_ids, k=k)
        else:  # semeval
            hits = qsim.top_k_other_source_by_text(
                query_texts, k=k, exclude_source_ids=contributing_sources
            )
        new_hits = [h for h in hits if h["chunk_id"] not in prev_ids]
        prev_ids.update(h["chunk_id"] for h in hits)

        if not new_hits and k > start_k:
            trace.append({"rung": f"2_cross_k{k}", "note": "no new retrievals; skipped"})
            continue

        retrieved_qa = _format_retrieved_qa(new_hits, chunk_index)
        empty_q = _empty_slot_questions(empties)
        cross_prompt = _fill(
            R.cross_resolve,
            axial_category=cat.get("axial_category", ""),
            filled_slots_context=_filled_slots_context(cat, empties),
            empty_slot_questions=empty_q,
            retrieved_qa=retrieved_qa,
        )
        cross_raw = call_llm(cross_prompt, "")
        cross_parsed = _safe_json(cross_raw)
        applied = _apply_resolutions(cat, cross_parsed, empties)
        trace.append({
            "rung": f"2_cross_k{k}",
            "k": k,
            "n_new_retrieved": len(new_hits),
            "retrieved_chunk_ids": [h["chunk_id"] for h in new_hits],
            "raw": cross_raw,
            "parsed": cross_parsed,
            "slots_filled": applied,
        })

        empties = empty_slots(cat)
        if not empties:
            break

    # ---- terminal: leave remaining empty, recorded --------------------------
    still_empty = empty_slots(cat)
    for s in still_empty:
        cat[s] = ""  # normalize missing/None to explicit empty string
    if still_empty:
        trace.append({"rung": "terminal", "left_empty": still_empty})

    cat["__slot_trace__"] = trace
    return cat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(raw: str):
    if not raw or not raw.strip():
        return None
    try:
        clean = raw.strip().strip("```json").strip("```")
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


def _format_aggregated_evidence(extractions: List[dict], empties: List[str]) -> str:
    """Group extracted evidence by slot across all interviews, as prompt text."""
    by_slot: Dict[str, List[str]] = {s: [] for s in empties}
    for ex in extractions:
        parsed = ex.get("parsed")
        if not isinstance(parsed, dict):
            continue
        for item in parsed.get("extractions", []):
            slot = item.get("slot")
            if slot not in by_slot:
                continue
            for ev in item.get("evidence", []):
                psg = ev.get("text_passage", "").strip()
                rsn = ev.get("reasoning", "").strip()
                if psg or rsn:
                    by_slot[slot].append(f"[{ex['source_id']}] \"{psg}\" — {rsn}")
    lines = []
    for slot in empties:
        lines.append(f"### {slot}")
        ev = by_slot[slot]
        lines.extend(f"  - {e}" for e in ev) if ev else lines.append("  (no evidence extracted)")
    return "\n".join(lines)


def _format_retrieved_qa(hits: List[dict], chunk_index: Dict[str, dict]) -> str:
    lines = []
    for h in hits:
        chunk = chunk_index.get(h["chunk_id"], {})
        text = chunk.get("text", h.get("question", ""))
        lines.append(f"[{h['source_id']} | sim={h['similarity']:.3f}]\n{text}")
    return "\n\n".join(lines) if lines else "(none)"


def _apply_resolutions(cat: dict, parsed, empties: List[str]) -> List[str]:
    """Apply non-null resolutions to still-empty slots. Returns slots filled."""
    if not isinstance(parsed, dict):
        return []
    filled = []
    for res in parsed.get("resolutions", []):
        slot = res.get("slot")
        val = res.get("value")
        if slot in empties and slot in PARADIGM_SLOTS:
            if val is not None and isinstance(val, str) and val.strip() and val.strip().lower() != "null":
                cat[slot] = val.strip()
                filled.append(slot)
    return filled
