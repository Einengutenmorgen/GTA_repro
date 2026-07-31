# study_contexts.py
"""
Per-dataset STUDY CONTEXT blocks for the GT prompt modules.

Both tradition modules (prompts.py = Straussian, prompts_charmaz.py =
Charmaz) hard-code a "STUDY CONTEXT" section near the top of each of
their three prompts (open/initial, axial/focused, selective/theoretical)
that orients the model to the Silan-Ciruelas relationship-quality
interview dataset. To run the same two traditions over a second
dataset -- SemEval-2025 Task 10 (news/opinion articles that portray
named entities) -- the ONLY thing that may differ is that STUDY CONTEXT
section. Coding rules, JSON output contracts, and output field names
must stay byte-identical between datasets so tradition remains the
single comparison axis.

This module intentionally does NOT edit prompts.py or prompts_charmaz.py.
Instead it holds the new dataset's context block plus a small,
line-based splicing helper (`swap_study_context`) that locates the
STUDY CONTEXT header + its contiguous bullet lines inside an existing
prompt string and replaces *only* that span. Everything before and
after the span -- including blank-line spacing, coding instructions,
and the JSON contract -- is left untouched by construction, so there is
no manual re-typing of tradition-specific text to drift out of sync.

IMPORTANT (load-bearing for the study): SEMEVAL_STUDY_CONTEXT must
never name or hint at the SemEval-2025 Task 10 label scheme -- not the
3 narrative archetypes (Protagonist / Antagonist / Innocent) and not
the 22 fine-grained roles (Guardian, Deceiver, Tyrant, Scapegoat, ...),
nor even that a fixed taxonomy exists. The study measures whether
inductive coding recovers that human scheme; leaking it here would
turn recovery into classification and destroy the comparison. This
block describes only the task shape and the analytic aim -- never the
categories.

Revision note (grounded in Piskorski et al., "SemEval-2025 Task 10:
Multilingual Characterization and Extraction of Narratives from Online
News"): the task's own framing and annotation guidelines describe
entity framing as assigning each entity a "role" that is "central to
the article's story," chosen from a "taxonomy of roles." The first
draft of this block used near-synonyms of that exact vocabulary --
"portrays," "stance," "characterization," "framing the text assigns"
-- which sits closer to restating the task than to a neutral orienting
block; a model with parametric knowledge of the paper could plausibly
connect "portrayal/characterization of named entities" straight back
to the protagonist/antagonist/innocent role scheme without a single
banned term appearing. The revision below instead anchors on concrete,
behavior-level description (what the text attributes to / claims about
/ associates with an entity) rather than the abstraction level ("role,"
"portrayal," "characterization") the task itself operates at, and
explicitly disclaims any predetermined scheme -- without naming or
even gesturing at what that scheme might look like (e.g. it does NOT
say anything like "not hero/villain roles," since naming a plausible
answer shape as a negative example is itself a leak).
"""
from __future__ import annotations

# --- SemEval-2025 Task 10 (entity framing in news/opinion articles) --------
# Kept deliberately parallel in register to the Silan STUDY CONTEXT blocks:
# an "orienting focus only" disclaimer, then bullets for genre, analytic
# aim, entity focus, and unit of analysis. Reused verbatim across both
# traditions and across all three coding stages (open/axial/selective,
# initial/focused/theoretical) -- the tradition-specific instructions
# below it already exist unchanged and are what varies instead.
#
# Deliberately avoids "portray(al)," "stance," "characterization," and
# "framing" -- the task's own vocabulary for what an entity role IS --
# in favor of concrete, behavior-level description (attributed actions,
# judgments, associated effects). See the revision note above for why.
SEMEVAL_STUDY_CONTEXT = """STUDY CONTEXT (orienting focus only — do NOT treat as a list of expected findings or as an allowed set of categories):
- Text genre: news and opinion articles on contested current-affairs topics (e.g. the war in Ukraine, or debates over climate-change policy). Each article names specific entities — people, organizations, or groups — and says things about them over the course of the piece.
- Analytic aim: to notice, in the article's own words, what the text actually attributes to or claims about each named entity — actions it says the entity took, judgments it makes about the entity, effects it ties the entity to. Build codes from those specific words and claims; do not assume any predetermined scheme of how entities function in the story, and do not sort entities into fixed types before the data itself shows you why.
- Entity focus: only entities that are central to the article's own story are worth coding; a name mentioned once in passing, with nothing said about it, is not.
- Unit of analysis: treat the whole article as a single unit; several entities may be named within it, but you are coding the article as a whole, not producing a separate profile per entity."""

# Registry of swappable dataset contexts, keyed by the `dataset` argument
# accepted by prompt_registry.get_prompts(). "silan" is intentionally NOT
# listed here -- it means "use the tradition module's own STUDY CONTEXT,
# unmodified," which is the default/backward-compatible path.
DATASET_CONTEXTS = {
    "semeval": SEMEVAL_STUDY_CONTEXT,
}


# ---------------------------------------------------------------------------
# Integration-closer swap (Charmaz theoretical / memo-sorting steps)
# ---------------------------------------------------------------------------
# prompts_charmaz.py's THEORETICAL_CODING_PROMPT (tradition="charmaz",
# accessed as PromptSet.selective) and prompts_charmaz_recursion.py's
# MEMO_SORTING_PROMPT (the Charmaz arm's actual terminal step -- called
# directly by gta_pipeline.run_memo_sorting, NOT through get_prompts) each
# name the Silan phenomenon in their CLOSING INSTRUCTION sentences, outside
# the swappable STUDY CONTEXT block that swap_study_context() handles.
# Those sentences were reached unmodified by every non-"silan" run until
# this was added: a real semeval Charmaz run's output_final_theory.md came
# back with literal "relationship quality" / "participants construct"
# language, because MEMO_SORTING_PROMPT's closer was never swapped and
# run_memo_sorting didn't even accept a `dataset` argument.
#
# INTEGRATION_CLOSERS maps each known Silan closer FRAGMENT (a short, exact
# substring -- deliberately not the whole surrounding sentence, so the same
# fragment matches its several near-duplicate occurrences across both
# prompt modules) to its per-dataset replacement. Same swap philosophy as
# STUDY CONTEXT: only the listed fragment changes; everything else in the
# prompt -- structure, JSON contract, surrounding instruction wording -- is
# left byte-identical. A fragment that isn't present in a given prompt is
# simply skipped, which is the expected, common case: every Straussian
# prompt and prompts_charmaz.py's initial/focused prompts contain none of
# these fragments at all.
#
# Replacement wording for "semeval" deliberately reuses SEMEVAL_STUDY_CONTEXT's
# own vocabulary (attributes to / claims about / named entities) and avoids
# the same banned near-synonyms (portray, stance, characterization, framing)
# -- see the revision note above for why those are excluded.
INTEGRATION_CLOSERS = {
    "participants construct relationship quality": {
        "semeval": "the corpus's articles attribute meaning to their named entities",
    },
    "participants' lived sense of relationship quality": {
        "semeval": "the corpus's overall pattern of attribution to its named entities",
    },
    "how quality is built, sustained, eroded": {
        "semeval": "how attributions to an entity are made, reinforced, or contested",
    },
}


def swap_integration_closers(prompt_text: str, dataset: str) -> str:
    """Replace any known Silan-specific integration-closer fragment with the
    per-dataset equivalent.

    No-op for dataset == "silan" (returns prompt_text unchanged) and for any
    fragment not present in prompt_text -- most prompts have none of these
    fragments at all, which is expected and fine. Unlike swap_study_context,
    this never raises on a non-match: it is meant to be applied defensively
    across every prompt in both traditions (coding, memo-writing, slot
    recursion) rather than only where a STUDY CONTEXT header is guaranteed
    to exist.
    """
    if dataset == "silan":
        return prompt_text
    for fragment, by_dataset in INTEGRATION_CLOSERS.items():
        replacement = by_dataset.get(dataset)
        if replacement and fragment in prompt_text:
            prompt_text = prompt_text.replace(fragment, replacement, 1)
    return prompt_text


def extract_study_context(prompt_text: str) -> str:
    """Return the STUDY CONTEXT header line plus its contiguous bullet lines.

    Finds the line whose stripped text starts with "STUDY CONTEXT", then
    greedily collects every immediately-following line that starts with
    "- " (a bullet), stopping at the first non-bullet line. This mirrors
    the shape used by every prompt in prompts.py / prompts_charmaz.py
    without depending on the exact (varying) header wording or bullet
    count in each one.
    """
    lines = prompt_text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("STUDY CONTEXT"):
            start = i
            break
    if start is None:
        raise ValueError("no 'STUDY CONTEXT' header found in prompt text")

    end = start + 1
    while end < len(lines) and lines[end].strip().startswith("-"):
        end += 1

    return "\n".join(lines[start:end])


def swap_study_context(prompt_text: str, new_context: str) -> str:
    """Return `prompt_text` with only its STUDY CONTEXT block replaced.

    Everything else in `prompt_text` -- the opening role sentence, any
    generic (non-dataset-specific) framing paragraph, tradition-specific
    coding rules, and the JSON output contract -- is left byte-identical,
    since only the exact substring found by `extract_study_context` is
    replaced.
    """
    old_context = extract_study_context(prompt_text)
    new_text = prompt_text.replace(old_context, new_context, 1)
    if new_text == prompt_text:
        raise ValueError(
            "STUDY CONTEXT swap had no effect; the extracted block was not "
            "found verbatim in the prompt text"
        )
    return new_text