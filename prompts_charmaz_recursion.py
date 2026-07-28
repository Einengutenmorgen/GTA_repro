# prompts_charmaz_recursion.py
"""
Charmaz constructivist-GT recursion prompts: the memo, iterative-coding,
applicability-test, and memo-sorting prompts that drive the slice-driven
reflection loop in charmaz_loop.py.

Parallel to prompts_recursion.py (which serves the Straussian slot ladder).
Kept in a dedicated module so the two traditions never share prompt strings
and the constructivist commitments (gerunds, constant comparison, memo-driven
reflection, "no imposed structure") stay isolated from the paradigm model.

Design commitments encoded here (from the Charmaz theoretical grounding &
experiment setup):
  - Memos are REAL artifacts that reason before answering, and advanced memos
    must explicitly NAME thin/ambiguous areas (that naming feeds gap-focused
    re-sampling).
  - Iterative coding is gap-focused: on a new slice the model receives the
    existing focused categories and codes only what they do NOT already cover
    (initial) / integrates-then-forms-new (focused). This is constant
    comparison, not coding from scratch.
  - The applicability test measures how much of a FRESH slice the existing
    codes absorb; "does not fit" is a first-class, non-penalized answer, so
    genuine non-saturation is detectable rather than confabulated away.

All prompts use {curly} fields filled programmatically by charmaz_loop.py, and
emit strict JSON on the pipeline's existing parse-and-clean path — except the
memo prompts, whose free-text analytic body is intentionally prose (a memo is
writing), wrapped in a thin JSON envelope so it stays machine-addressable.
"""

# ---------------------------------------------------------------------------
# STEP 4 — Initial Memo-Writing (reason-then-write; real artifact)
# ---------------------------------------------------------------------------
INITIAL_MEMO_PROMPT = """You are a qualitative researcher writing INITIAL MEMOS in Kathy Charmaz's constructivist grounded theory.

A memo is you thinking on paper. It is not a summary. You reason about what an initial code means, what actions or processes it captures, and how incidents compare — BEFORE committing to any category.

THE INITIAL CODES (with the data they came from):
{coded_data}

YOUR TASK — for the significant/recurring codes, write a short analytic memo that:
- Explores the MEANING behind the code: what process or action is the participant engaged in?
- Compares incidents: where does this code appear across participants, and how does it vary?
- Notes whether the code feels like it is rising toward a tentative category, and why.
Reason first, in your own analytic voice; do not merely restate the code.

Output ONLY a valid JSON object with this exact structure:
{{
  "memos": [
    {{
      "focus_code": "the initial code this memo is about",
      "reasoning": "your analytic thinking: meaning, comparison across incidents, variation",
      "tentative_category": "a tentative category name if one is emerging, else null"
    }}
  ]
}}
Do not include markdown fences; output the raw JSON object only."""


# ---------------------------------------------------------------------------
# STEP 6 — Advanced Memo-Writing (reason-then-write; NAMES thin areas)
# ---------------------------------------------------------------------------
# The thin-area naming is load-bearing: charmaz_loop.py reads
# `thin_areas` to focus the next slice's coding (gap-directed re-sampling).
ADVANCED_MEMO_PROMPT = """You are a qualitative researcher writing ADVANCED MEMOS in Charmaz's constructivist grounded theory.

You are refining conceptual categories through constant comparison. You reason about each focused category's PROPERTIES, the data it subsumes, and how it compares to sibling categories — and you are candid about where the analysis is still THIN.

THE FOCUSED CATEGORIES (with their supporting codes):
{focused_categories}

YOUR TASK — for each focused category, write an advanced memo that:
- Articulates the category's PROPERTIES (its dimensions, its range of variation).
- Traces the data it subsumes and compares it against other categories (what distinguishes them? where do they overlap?).
- Surfaces underlying ASSUMPTIONS in how the category has been construed.
- Explicitly NAMES what is thin, ambiguous, or under-evidenced: which properties are asserted on little data, which distinctions are unclear, what you would want more data to resolve.

The thin-area naming is essential — it directs where the analysis looks next. Be specific and honest; do not paper over gaps.

Output ONLY a valid JSON object with this exact structure:
{{
  "memos": [
    {{
      "category": "the focused category this memo is about",
      "properties": ["property or dimension", "..."],
      "reasoning": "your analytic thinking: subsumed data, comparison to siblings, assumptions",
      "thin_areas": ["specific under-evidenced property / unclear distinction / what more data would resolve", "..."]
    }}
  ]
}}
Do not include markdown fences; output the raw JSON object only."""


# ---------------------------------------------------------------------------
# STEP 3' — Initial Coding INSIDE the loop (gap-focused, constant comparison)
# ---------------------------------------------------------------------------
# Receives existing focused categories + named thin areas; codes only what is
# NOT already captured. Mirrors a human coding new interviews through their
# existing analysis rather than from scratch.
INITIAL_CODING_ITER_PROMPT = """You are a qualitative researcher performing INITIAL CODING (Charmaz constructivist GT) on a NEW slice of interview data, as part of an ongoing analysis.

You are NOT starting from scratch. You already have an emerging set of focused categories. Through CONSTANT COMPARISON, you read this new data and code specifically what your existing categories do NOT already capture, and what elaborates or challenges them.

YOUR EXISTING FOCUSED CATEGORIES:
{existing_categories}

AREAS YOUR ANALYSIS IS CURRENTLY THIN ON (look especially for data that speaks to these):
{thin_areas}

CHARMAZ INITIAL CODING RULES (unchanged):
- Code with GERUNDS (action/process words ending in "-ing"): what is the participant DOING, feeling, experiencing?
- Stay close to the data; take the participant's point of view.
- Keep codes short, active, provisional.
- Preserve a participant's own striking phrase as an in-vivo code where one is present (mark it).
- IGNORE interviewer prompts and procedural/administrative talk.

Read the new interview data below and produce initial codes, prioritizing what is NEW relative to your existing categories and what fills the thin areas.

Output ONLY a valid JSON array with this exact structure:
[
  {{
    "open_code": "gerund-phrase initial code (action/process/meaning)",
    "text_passage": "exact corresponding text from the input",
    "in_vivo": false
  }}
]
Set "in_vivo" true only when the code is the participant's own verbatim phrase.
Do not include markdown fences; output the raw JSON array only."""


# ---------------------------------------------------------------------------
# STEP 5' — Focused Coding INSIDE the loop (integrate-then-form-new)
# ---------------------------------------------------------------------------
# Receives the current focused categories + new initial codes from S3'; extends
# the existing category system rather than rebuilding it. May revise codes
# themselves (rename/split/merge) — Charmaz permits re-seeing.
FOCUSED_CODING_ITER_PROMPT = """You are a qualitative researcher performing FOCUSED CODING (Charmaz constructivist GT) as part of an ongoing, iterative analysis.

You have an existing set of focused categories. New initial codes have just been produced from a new slice of data. Through CONSTANT COMPARISON you now INTEGRATE the new codes into your category system: attach them where they fit, and form new categories ONLY where existing ones genuinely cannot hold them. You may also REVISE existing categories (rename, split, or merge) if the new data shows they no longer fit — codes are your construction, not fixed.

YOUR EXISTING FOCUSED CATEGORIES:
{existing_categories}

NEW INITIAL CODES TO INTEGRATE:
{new_codes}

YOUR TASK:
- For each new code, integrate it into the most fitting existing category, or assign it to a new category if none fits.
- Where the new data warrants, revise categories: rename for a better fit, split an overloaded category, or merge redundant ones.
- Return the FULL updated set of focused categories (existing + revised + new), plus a typed list of the changes you made so the analytic trail is explicit.

Output ONLY a valid JSON object with this exact structure:
{{
  "categories": [
    {{
      "reasoning": "why these codes cohere; the significant/frequent action they share",
      "supporting_open_codes": ["exact", "initial", "codes", "in", "this", "category"],
      "axial_category": "focused category name (active/process phrasing)"
    }}
  ],
  "changes_made": [
    {{"type": "added",      "category": "name", "rationale": "..."}},
    {{"type": "elaborated", "category": "name", "new_properties": ["..."], "rationale": "..."}},
    {{"type": "renamed",    "from": "old name", "to": "new name", "rationale": "..."}},
    {{"type": "split",      "from": "old name", "into": ["a", "b"], "rationale": "..."}},
    {{"type": "merged",     "from": ["a", "b"], "into": "name", "rationale": "..."}}
  ]
}}
Return an empty "changes_made" list if the new codes all fit existing categories without revision.
Do not include markdown fences; output the raw JSON object only."""


# ---------------------------------------------------------------------------
# STEP 8 — Applicability test (saturation probe on a FRESH slice)
# ---------------------------------------------------------------------------
# Measures how much of a fresh slice the existing focused categories absorb.
# "does not fit" is first-class: the point is to DETECT non-saturation, not to
# force-fit. charmaz_loop.py computes the absorbed fraction from `assignments`.
APPLICABILITY_TEST_PROMPT = """You are a qualitative researcher testing THEORETICAL SUFFICIENCY (saturation) in Charmaz's constructivist grounded theory.

You have a set of focused categories built from data seen so far. Here is a FRESH slice of interview data the categories have NOT been built from. Your job is to judge, honestly, how well the EXISTING categories account for this new data — this is how we detect whether the analysis has saturated.

YOUR EXISTING FOCUSED CATEGORIES:
{existing_categories}

FRESH DATA TO TEST (each item is a codeable unit):
{fresh_units}

YOUR TASK:
For EACH codeable unit, decide whether an existing category already accounts for its analytic content:
- "fits": an existing category captures it (name that category).
- "does_not_fit": it raises an action/process/meaning NOT captured by any existing category (briefly say what is new).

Judge honestly. A unit that genuinely introduces something new MUST be marked "does_not_fit" — do not stretch a category to cover it. Detecting new content is the entire purpose of this test; false "fits" would hide non-saturation.

Output ONLY a valid JSON object with this exact structure:
{{
  "assignments": [
    {{"unit_id": "the unit's id", "verdict": "fits | does_not_fit", "category": "fitting category name or null", "note": "if does_not_fit, what analytic content is new"}}
  ]
}}
Do not include markdown fences; output the raw JSON object only."""


# ---------------------------------------------------------------------------
# STEP 9 — Sorting & finalizing memos into the integrated account
# ---------------------------------------------------------------------------
MEMO_SORTING_PROMPT = """You are a qualitative researcher performing THEORETICAL SORTING and INTEGRATION in Charmaz's constructivist grounded theory — the final analytic step.

Your categories are now developed and your memos accumulated. You sort the memos to fit the theoretical categories, establish the logical relationships among categories, and integrate them into a coherent theoretical account.

THE FINAL FOCUSED CATEGORIES:
{focused_categories}

THE ACCUMULATED ADVANCED MEMOS:
{memos}

YOUR TASK:
- Sort and integrate the memos under their categories.
- Establish how the categories relate to one another (which conditions, processes, or meanings connect them).
- Articulate an integrated, processual theoretical account of how participants construct relationship quality.

This is interpretive and co-constructed: you are theorizing relationships among categories, grounded in the memos and categories provided — NOT extracting a single pre-existing core via a fixed paradigm, and NOT importing the study topic as structure. Prefer processual language (how quality is built, sustained, eroded).

Write a short integrated theoretical account: name the central process, then explain in one or two paragraphs how the focused categories relate to constitute participants' lived sense of relationship quality."""
