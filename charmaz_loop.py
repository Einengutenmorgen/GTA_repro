# charmaz_loop.py
"""
Charmaz constructivist-GT reflection loop (slice-driven).

Parallel to slot_recursion.py (the Straussian slot ladder). Where the Straussian
loop is triggered by an empty paradigm slot and widens the evidence net for ONE
category, the Charmaz loop is triggered by a fresh SLICE of interviews and asks
whether the existing focused categories already absorb it — the computational
analogue of theoretical sampling + saturation testing on a FIXED corpus.

The loop, per iteration (experiment-setup §1, §4):
  1. Applicability test: label a fresh slice with existing focused categories.
       absorbed_fraction = fits / codeable_units
  2. If absorbed_fraction >= saturation_coverage_threshold  -> SATURATION.
     Else:
       S3'  initial-code the UNABSORBED units (gap-focused iter prompt)
       S5'  adapted focused coding: integrate new codes, form new categories,
            possibly revise existing ones (rename/split/merge)
       S6   refine the advanced memo (re-surfaces thin areas for next round)
       -> continue to the next slice
  3. Stop on the FIRST of: saturation reached / max_iterations / corpus exhausted.

Substitution honesty (grounding §1): this is theoretical RE-sampling of a fixed
corpus, never recruitment. On corpus exhaustion we STOP and record
`saturation_reached=False, reason="corpus_exhausted"` rather than re-sampling
seen data (which would trivially pass the applicability test and manufacture
false saturation).

Determinism: the loop SHAPE is deterministic given the slices (fixed slice
order, programmatic fraction check, fixed threshold). The fills are LLM calls;
same seed/temperature discipline as the forward pass applies. The change-tree
captures raw outputs so run-to-run variance is auditable (RQ1).

Nothing here re-embeds or re-chunks; it consumes caller-supplied slices and a
caller-bound LLM. Slicing itself is caller-supplied (see slice_units) so the
orchestrator stays pure and testable.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

from prompt_registry import get_charmaz_recursion_prompts, _fill, DEFAULT_DATASET

# Defaults — all control-relevant, all disclosed (experiment-setup §6).
DEFAULT_SATURATION_THRESHOLD = 0.80   # absorbed fraction that counts as "most"
DEFAULT_MAX_ITERATIONS = 5            # hard cap on loop passes (slice-aware; not the shared 3)


# ---------------------------------------------------------------------------
# Slicing (caller-usable helper; kept here so main.py need not reimplement it)
# ---------------------------------------------------------------------------

def slice_sources(source_ids: List[str], slice_size: int) -> List[List[str]]:
    """Partition participant source_ids into ordered slices of `slice_size`.

    Slicing is by PARTICIPANT (source_id), not by chunk, so a slice is a set of
    whole interviews — the unit theoretical sampling operates on. Order is the
    caller's order (stable); the first slice seeds the forward pass and the rest
    feed the loop.
    """
    if slice_size < 1:
        raise ValueError("slice_size must be >= 1")
    seen = list(dict.fromkeys(source_ids))  # dedupe, keep order
    return [seen[i:i + slice_size] for i in range(0, len(seen), slice_size)]


def units_for_sources(chunk_index: Dict[str, dict], sources: List[str]) -> List[dict]:
    """Return codeable units (chunks) for a set of participants, in order.

    A "codeable unit" is one chunk dict from chunk_index. Grain therefore
    follows chunking (qa_units at pairs_per_chunk=1), consistent with the
    retrieval index. Applicability is measured over these units.
    """
    srcset = set(sources)
    units = [c for c in chunk_index.values() if c.get("source_id") in srcset]
    units.sort(key=lambda c: (c.get("source_id", ""), c.get("q_index", 0)))
    return units


# ---------------------------------------------------------------------------
# JSON helper (same contract as the rest of the pipeline)
# ---------------------------------------------------------------------------

def _safe_json(raw: str):
    if not raw or not raw.strip():
        return None
    try:
        clean = raw.strip().strip("```json").strip("```")
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Context formatting (pure string assembly)
# ---------------------------------------------------------------------------

def _format_categories(categories: List[dict]) -> str:
    """Focused categories -> prompt text. Paradigm-free (no slots)."""
    if not categories:
        return "  (none yet)"
    lines = []
    for c in categories:
        if not isinstance(c, dict):
            continue
        name = c.get("axial_category", "?")
        codes = c.get("supporting_open_codes", []) or []
        lines.append(f"- {name}")
        if c.get("reasoning"):
            lines.append(f"    rationale: {c['reasoning']}")
        if codes:
            preview = "; ".join(str(x) for x in codes[:12])
            more = f" (+{len(codes) - 12} more)" if len(codes) > 12 else ""
            lines.append(f"    codes: {preview}{more}")
    return "\n".join(lines)


def _format_thin_areas(memo: Optional[dict]) -> str:
    """Advanced-memo thin areas -> prompt text for gap-focused re-sampling."""
    if not isinstance(memo, dict):
        return "  (none named yet)"
    areas: List[str] = []
    for m in memo.get("memos", []):
        if not isinstance(m, dict):
            continue
        cat = m.get("category", "?")
        for ta in m.get("thin_areas", []) or []:
            areas.append(f"- [{cat}] {ta}")
    return "\n".join(areas) if areas else "  (none named yet)"


def _format_units(units: List[dict]) -> str:
    """Codeable units -> enumerated prompt text for the applicability test."""
    lines = []
    for u in units:
        uid = u.get("chunk_id", "?")
        lines.append(f"[{uid}]\n{u.get('text', '')}")
    return "\n\n".join(lines) if lines else "(none)"


def _format_new_codes(codes: List[dict]) -> str:
    lines = []
    for c in codes:
        if not isinstance(c, dict):
            continue
        tag = " (in-vivo)" if c.get("in_vivo") else ""
        lines.append(f"- {c.get('open_code', '')}{tag}  ::  {c.get('text_passage', '')}")
    return "\n".join(lines) if lines else "(none)"


# ---------------------------------------------------------------------------
# Per-step LLM moves
# ---------------------------------------------------------------------------

def _applicability_test(categories, units, call_llm, dataset=DEFAULT_DATASET) -> dict:
    """Label a fresh slice with existing categories. Returns parsed result +
    absorbed fraction. Missing/failed assignments count as does_not_fit so a
    parse failure can never manufacture saturation."""
    R = get_charmaz_recursion_prompts(dataset)
    prompt = _fill(
        R.applicability_test,
        existing_categories=_format_categories(categories),
        fresh_units=_format_units(units),
    )
    raw = call_llm(prompt, "")
    parsed = _safe_json(raw)
    assignments = parsed.get("assignments", []) if isinstance(parsed, dict) else []

    by_id = {a.get("unit_id"): a for a in assignments if isinstance(a, dict)}
    n_total = len(units)
    n_fit = 0
    unabsorbed_ids: List[str] = []
    for u in units:
        uid = u.get("chunk_id")
        a = by_id.get(uid)
        if a and a.get("verdict") == "fits":
            n_fit += 1
        else:
            unabsorbed_ids.append(uid)  # unknown/failed -> treat as not fitting

    fraction = (n_fit / n_total) if n_total else 0.0
    return {
        "raw": raw,
        "parsed": parsed,
        "n_units": n_total,
        "n_fit": n_fit,
        "absorbed_fraction": fraction,
        "unabsorbed_chunk_ids": unabsorbed_ids,
    }


def _initial_code_iter(categories, thin_areas_text, units, call_llm, dataset=DEFAULT_DATASET) -> List[dict]:
    """S3' — gap-focused initial coding of the unabsorbed units."""
    data_text = _format_units(units)
    R = get_charmaz_recursion_prompts(dataset)
    prompt = _fill(
        R.initial_coding_iter,
        existing_categories=_format_categories(categories),
        thin_areas=thin_areas_text,
    )
    raw = call_llm(prompt, data_text)
    parsed = _safe_json(raw)
    if not isinstance(parsed, list):
        return []
    # stamp provenance where resolvable (unit order is not guaranteed, so we
    # cannot map codes to chunk_ids reliably; leave chunk_id off iter codes but
    # keep them well-formed for focused integration).
    return [c for c in parsed if isinstance(c, dict) and c.get("open_code")]


def _focused_code_iter(categories, new_codes, call_llm, dataset=DEFAULT_DATASET) -> dict:
    """S5' — integrate new codes, forming/revising categories. Returns the full
    updated category set + typed change list."""
    R = get_charmaz_recursion_prompts(dataset)
    prompt = _fill(
        R.focused_coding_iter,
        existing_categories=_format_categories(categories),
        new_codes=_format_new_codes(new_codes),
    )
    raw = call_llm(prompt, "")
    parsed = _safe_json(raw)
    if not isinstance(parsed, dict):
        return {"categories": categories, "changes_made": [], "raw": raw, "parsed": parsed}
    updated = parsed.get("categories")
    if not isinstance(updated, list) or not updated:
        updated = categories  # never silently drop the category set on a bad parse
    return {
        "categories": updated,
        "changes_made": parsed.get("changes_made", []) if isinstance(parsed.get("changes_made"), list) else [],
        "raw": raw,
        "parsed": parsed,
    }


def _advanced_memo(categories, call_llm, dataset=DEFAULT_DATASET) -> dict:
    """S6 — refine the advanced memo (re-surfaces thin areas)."""
    R = get_charmaz_recursion_prompts(dataset)
    prompt = _fill(
        R.advanced_memo,
        focused_categories=_format_categories(categories),
    )
    raw = call_llm(prompt, "")
    parsed = _safe_json(raw)
    return parsed if isinstance(parsed, dict) else {"memos": [], "__raw__": raw}


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------

def run_charmaz_loop(
    initial_categories: List[dict],
    initial_advanced_memo: dict,
    remaining_slices: List[List[str]],
    chunk_index: Dict[str, dict],
    call_llm: Callable[[str, str], str],
    saturation_threshold: float = DEFAULT_SATURATION_THRESHOLD,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    dataset: str = DEFAULT_DATASET,
) -> dict:
    """Run the slice-driven reflection loop.

    Parameters
    ----------
    initial_categories : focused categories from the forward pass on slice 1.
    initial_advanced_memo : the advanced memo from slice 1 (carries thin areas).
    remaining_slices : ordered list of slices (each a list of source_ids) NOT
        yet seen by the forward pass. The loop consumes them in order.
    chunk_index : {chunk_id -> chunk dict} for resolving units per slice.
    call_llm : (system_prompt, user_text) -> raw string. Caller binds the model.
    saturation_threshold : absorbed fraction that counts as saturation.
    max_iterations : hard cap on loop passes.
    dataset : forwarded to every internal LLM step (applicability test,
        gap-focused initial/focused coding, advanced-memo refinement) via
        prompt_registry.get_charmaz_recursion_prompts, so a "semeval" run
        stays dataset-consistent through every iteration, not just the seed
        pass. Defaults to "silan".

    Returns a dict:
      {
        "final_categories": [...],
        "final_advanced_memo": {...},
        "saturation_reached": bool,
        "stop_reason": "saturation" | "max_iterations" | "corpus_exhausted",
        "n_iterations": int,
        "change_tree": [ per-iteration record ],  # the publishable artifact
      }

    The change_tree records, per iteration: the slice tested, the applicability
    fraction, whether it saturated, and — when it did not — the new codes, the
    typed category changes, and the resulting category set. This makes the
    diff between iterations reconstructable, not just the endpoints.
    """
    categories = list(initial_categories)
    advanced_memo = dict(initial_advanced_memo) if isinstance(initial_advanced_memo, dict) else {"memos": []}
    change_tree: List[dict] = []

    saturation_reached = False
    stop_reason = "corpus_exhausted"  # default if we run out of slices

    n_slices = len(remaining_slices)
    for it, slice_sources_ids in enumerate(remaining_slices, start=1):
        if it > max_iterations:
            stop_reason = "max_iterations"
            break

        units = units_for_sources(chunk_index, slice_sources_ids)
        thin_text = _format_thin_areas(advanced_memo)

        # 1. Applicability test on the fresh slice ---------------------------
        test = _applicability_test(categories, units, call_llm, dataset=dataset)
        fraction = test["absorbed_fraction"]

        record: dict = {
            "iteration_n": it,
            "slice_sources": slice_sources_ids,
            "n_units": test["n_units"],
            "n_fit": test["n_fit"],
            "absorbed_fraction": round(fraction, 4),
            "threshold": saturation_threshold,
            "applicability_raw": test["raw"],
        }

        # 2. Saturation decision --------------------------------------------
        if fraction >= saturation_threshold:
            record["decision"] = "saturated"
            record["changes_made"] = []
            record["resulting_categories"] = [c.get("axial_category") for c in categories if isinstance(c, dict)]
            change_tree.append(record)
            saturation_reached = True
            stop_reason = "saturation"
            break

        record["decision"] = "not_saturated"

        # 3. Integrate the unabsorbed data ----------------------------------
        unabsorbed = [u for u in units if u.get("chunk_id") in set(test["unabsorbed_chunk_ids"])]
        new_codes = _initial_code_iter(categories, thin_text, unabsorbed, call_llm, dataset=dataset)
        record["n_unabsorbed_units"] = len(unabsorbed)
        record["new_initial_codes"] = new_codes

        focused = _focused_code_iter(categories, new_codes, call_llm, dataset=dataset)
        categories = focused["categories"]
        record["changes_made"] = focused["changes_made"]
        record["focused_raw"] = focused["raw"]

        # 4. Refine the advanced memo (new thin areas for next round) --------
        advanced_memo = _advanced_memo(categories, call_llm, dataset=dataset)
        record["advanced_memo"] = advanced_memo
        record["resulting_categories"] = [c.get("axial_category") for c in categories if isinstance(c, dict)]

        change_tree.append(record)

        if it == n_slices:
            # consumed the last available slice without saturating
            stop_reason = "corpus_exhausted"

    return {
        "final_categories": categories,
        "final_advanced_memo": advanced_memo,
        "saturation_reached": saturation_reached,
        "stop_reason": stop_reason,
        "n_iterations": len(change_tree),
        "change_tree": change_tree,
    }
