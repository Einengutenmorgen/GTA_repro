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
block describes only the task shape (how the article portrays each
named entity) and the analytic aim -- never the categories.
"""
from __future__ import annotations

# --- SemEval-2025 Task 10 (entity framing in news/opinion articles) --------
# Kept deliberately parallel in register to the Silan STUDY CONTEXT blocks:
# an "orienting focus only" disclaimer, then bullets for genre, analytic
# aim, and unit of analysis. Reused verbatim across both traditions and
# across all three coding stages (open/axial/selective, initial/focused/
# theoretical) -- the tradition-specific instructions below it already
# exist unchanged and are what varies instead.
SEMEVAL_STUDY_CONTEXT = """STUDY CONTEXT (orienting focus only — do NOT treat as a list of expected findings or as an allowed set of categories):
- Text genre: news and opinion articles. Each article names one or more specific entities (people, organizations, states, or groups) and takes a stance toward at least one of them.
- Analytic aim: to surface how the article itself portrays each named entity — the stance, characterization, or framing the text assigns to that entity — described in the article's own terms.
- Unit of analysis: treat the whole article as a single unit; several entities may be named within it, but you are coding the article's overall portrayal of them, not scoring entities against a checklist."""

# Registry of swappable dataset contexts, keyed by the `dataset` argument
# accepted by prompt_registry.get_prompts(). "silan" is intentionally NOT
# listed here -- it means "use the tradition module's own STUDY CONTEXT,
# unmodified," which is the default/backward-compatible path.
DATASET_CONTEXTS = {
    "semeval": SEMEVAL_STUDY_CONTEXT,
}


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
