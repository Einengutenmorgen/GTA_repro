# prompt_registry.py
"""
Single source of truth for every prompt used by gta_pipeline.py / charmaz_loop.py
/ slot_recursion.py / main.py, across both GT traditions (Straussian, Charmaz)
and both datasets (silan, semeval).

SUPERSEDES: prompts.py, prompts_charmaz.py, prompts_charmaz_recursion.py,
prompts_recursion.py, study_contexts.py. Those five files are left on disk
(each now carries a header saying so) for historical reference/diffing, but
nothing in the active pipeline imports them anymore -- everything routes
through this module.

ARCHITECTURE
------------
Every prompt is assembled from a SKELETON (a template string, one per
"prompt family" -- e.g. "open/initial coding", "applicability test") plus
BUILDING BLOCKS that vary along exactly two independent axes:

  - DATASET_BLOCKS[dataset]     -- tradition-agnostic. Research goal, text
                                    genre, what to call the unit of data
                                    ("participant" vs "article"), what to
                                    exclude as boilerplate, and the closing
                                    "what does this all add up to" phrasing
                                    used in theoretical/memo-sorting steps.

  - TRADITION_BLOCKS[tradition] -- dataset-agnostic. The actual GT coding
                                    method (paradigm slots vs. gerund
                                    coding), and the JSON output contracts.
                                    This is the axis the whole SemEval study
                                    depends on staying identical regardless
                                    of dataset -- tradition blocks reference
                                    dataset content ONLY via the small named
                                    tokens above ({subject_phrase} etc.),
                                    never anything dataset-specific typed
                                    out inline.

No prompt currently needs a true tradition-x-dataset-specific block (a
string that depends on BOTH axes at once and can't be built from the two
independent blocks above). If one is ever needed, add a `COMBINED_BLOCKS`
dict keyed by (tradition, dataset) rather than smuggling dataset-specific
text into a TRADITION_BLOCKS entry.

Filling is done with `_fill()`, a plain str.replace-based substitution --
NOT str.format(). This is deliberate: several of these prompts embed literal
JSON examples ('{"memos": [...]}'), and str.format() would try to parse
those braces as format fields and crash. _fill() only ever touches the
exact "{token_name}" substrings it's told to replace, so JSON example text
is never touched.

Two substitution passes happen at different times:
  1. Build-time (inside this module): dataset + tradition tokens are filled
     in immediately when a prompt is requested (get_prompts /
     get_charmaz_recursion_prompts / get_straussian_slot_prompts).
  2. Call-time (in gta_pipeline.py / charmaz_loop.py / slot_recursion.py):
     the caller fills in the runtime DATA placeholders that remain in the
     returned template (e.g. {coded_data}, {fresh_units}) with `_fill()`
     again, using whatever was actually produced by earlier pipeline steps.
Because both passes use the same non-greedy _fill(), and a data
placeholder that survives pass 1 is simply a token pass 1 didn't know
about, there is no escaping to manage anywhere -- write JSON examples with
plain single braces, exactly as they should appear in the final prompt.

DATA-LAST CONVENTION
---------------------
Every skeleton in this file puts variable-length runtime DATA (the fresh
slice being tested, the accumulated memos, the aggregated evidence, ...)
at the very end of the prompt, after the task instructions and the JSON
output contract. This was a specific, confirmed problem in the prior
version of APPLICABILITY_TEST_PROMPT (a large {fresh_units} block sat in
the middle, between the instructions and the JSON contract) and is fixed
here across the board, not just in that one prompt.
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TRADITION = "straussian"
DEFAULT_DATASET = "silan"
TRADITIONS = ("straussian", "charmaz")
DATASETS = ("silan", "semeval")


def _fill(template: str, **kwargs) -> str:
    """Replace literal '{name}' tokens via str.replace (never str.format).

    Safe to use on strings that contain unrelated '{' / '}' characters (JSON
    examples, remaining runtime-data placeholders) because only the exact
    "{key}" substrings named in kwargs are touched -- everything else in the
    template is left completely alone, unlike str.format() which parses
    every brace pair in the string.
    """
    for key, value in kwargs.items():
        template = template.replace("{" + key + "}", value)
    return template


# ============================================================================
# AXIS 1 — DATASET BLOCKS (tradition-agnostic)
# ============================================================================
# The ONLY things that differ between a "silan" and a "semeval" run of the
# SAME tradition. Keep every value here free of tradition-specific language
# (no "paradigm slot", no "gerund") so these blocks stay reusable as-is by
# either tradition's templates.
#
# semeval's `research_goal` states the analytic aim directly (per revision:
# earlier drafts were deliberately vague/hedging to avoid any hint of the
# answer space; that vagueness itself was judged too long and unclear, so
# this version says plainly what we're looking for -- attribution of agency
# and moral valence to named entities -- while still never naming the
# specific taxonomy (3 archetypes / 22 fine-grained roles) or asserting that
# a fixed scheme exists. See the leak-prevention checks in
# verify_semeval_prompts.py, which block "portray/stance/characterization/
# framing/role/taxonomy" and the literal role names regardless of how this
# text is worded.
DATASET_BLOCKS = {
    "silan": {
        "research_goal": (
            "to understand how people conceptualize and experience relationship "
            "quality — how they themselves define and describe what makes a "
            "relationship high or low in quality, in their own words."
        ),
        "genre_note": (
            "semi-structured research interview transcripts. Each transcript "
            "opens with an administrative preamble (interviewer introductions, "
            "estimated duration, recording consent, data-sharing options); "
            "speakers alternate, and interviewer turns are prompts, not data."
        ),
        "source_label": "interview",
        "subject_phrase": "the participant",
        "subject_plural": "participants",
        "exclusion_rule": (
            "IGNORE non-substantive and procedural passages: consent / "
            "recording / data-sharing talk, interviewer logistics (duration, "
            "eligibility, thanks/closing), and bare demographic form-filling "
            "read back verbatim (unless the participant attaches substantive "
            "meaning to it). Code only content that bears on the research "
            "goal as expressed by the participant."
        ),
        "integration_closer": (
            "how participants construct relationship quality, and what that "
            "construction reveals about the lived sense of relational "
            "quality in this dataset"
        ),
        "process_examples": "how quality is built, sustained, eroded",
    },
    "semeval": {
        "research_goal": (
            "to identify how the news and opinion articles in this corpus "
            "attribute agency, responsibility, and moral valence to the "
            "named entities they discuss — what the text says an entity "
            "did, judged, or caused — and what analytic pattern that "
            "attribution forms across the corpus. Do not assume a "
            "predetermined set of entity types; the pattern must emerge "
            "from the data."
        ),
        "genre_note": (
            "news and opinion articles on contested current-affairs topics. "
            "Each article names specific entities — people, organizations, "
            "or groups — and makes claims about them over the course of the "
            "piece."
        ),
        "source_label": "article",
        "subject_phrase": "the article",
        "subject_plural": "articles",
        "exclusion_rule": (
            "IGNORE non-substantive boilerplate: bylines, datelines, image "
            "captions, and navigation/metadata text. Code only content that "
            "bears on what the article attributes to or claims about its "
            "named entities."
        ),
        "integration_closer": (
            "how this corpus's articles attribute agency and moral valence "
            "to their named entities, and what analytic pattern that "
            "attribution forms across the dataset"
        ),
        "process_examples": "how an attribution is made, reinforced, or contested",
    },
}


# ============================================================================
# AXIS 2 — TRADITION BLOCKS (dataset-agnostic)
# ============================================================================
# Everything below references dataset content ONLY via the named tokens
# defined in DATASET_BLOCKS ({subject_phrase}, {subject_plural},
# {source_label}, {exclusion_rule}, {research_goal}, {integration_closer},
# {process_examples}) — never anything dataset-specific typed out inline.
# JSON schema blocks are dataset-invariant by construction (no dataset
# tokens appear in them at all), which is what keeps output field names
# identical across datasets.

STRAUSSIAN = {
    "open_role": (
        "an expert qualitative researcher performing inductive open coding "
        "in a Straussian grounded-theory tradition"
    ),
    "open_rules": (
        "Stay close to what {subject_phrase} actually says; each code should "
        "name a concept, behavior, emotion, or incident grounded in a "
        "specific passage. Do not over-fragment: one code per distinct "
        "idea, not one per sentence."
    ),
    "open_json": """[
  {
    "open_code": "basic concept, behavior, emotion, or incident",
    "text_passage": "Exact corresponding text passage from the input"
  }
]""",
    "axial_role": (
        "an expert qualitative researcher performing axial coding in a "
        "Straussian grounded-theory tradition"
    ),
    "axial_intro": (
        "Let the categories emerge from the codes; do not impose the "
        "research goal as a category structure. If some open codes are "
        "procedural noise that slipped through open coding, exclude them "
        "rather than building a category around them."
    ),
    "axial_rules": (
        "PARADIGM SLOTS — condition, action_interaction, consequence:\n"
        "Fill each slot ONLY if the grouped open codes actually provide "
        "evidence for it. If the codes do not speak to a slot, return an "
        "empty string \"\" for that slot rather than inventing or "
        "stretching to fill it. An honestly empty slot is a valid, "
        "informative outcome — it signals that the paradigm model may not "
        "fully fit this category, and a later targeted pass will attempt "
        "to fill it from additional data. Do NOT confabulate to avoid an "
        "empty slot."
    ),
    "axial_json": """[
  {
    "reasoning": "Explanation or thought process of grouping",
    "supporting_open_codes": ["list", "of", "exact", "open", "codes", "used"],
    "axial_category": "Name of the overarching category",
    "condition": "What conditions give rise to this? (\\"\\" if not evidenced)",
    "action_interaction": "What behaviors or interactions occur? (\\"\\" if not evidenced)",
    "consequence": "What is the outcome? (\\"\\" if not evidenced)"
  }
]""",
    "selective_role": (
        "an expert qualitative researcher performing selective coding in a "
        "Straussian grounded-theory tradition"
    ),
    "selective_rules": (
        "Synthesize the axial relationships into a single 'Selective Code' "
        "or Core Category that explains the central phenomenon of the "
        "entire dataset. The core category must be grounded in the axial "
        "relationships provided, not in the stated research goal.\n\n"
        "Provide the Core Category name, followed by a brief paragraph "
        "explaining the grounded theory."
    ),
}

CHARMAZ = {
    "open_role": (
        "a qualitative researcher coding {source_label} data using Kathy "
        "Charmaz's CONSTRUCTIVIST grounded theory (initial coding)"
    ),
    "open_rules": (
        "CHARMAZ INITIAL CODING — follow these rules:\n"
        "- Code with GERUNDS (action words ending in \"-ing\"). Name what "
        "{subject_phrase} is doing — attributing, judging, deciding, "
        "describing — not the abstract topic. Write action-phrase codes "
        "such as \"Blaming an institution for a decision\", \"Justifying a "
        "controversial choice\", \"Downplaying a risk\" — NOT static nouns "
        "like \"Blame\", \"Justification\", \"Risk\".\n"
        "- Stay close to the data. Code the specific action or meaning in "
        "THIS passage; do not jump to abstract themes or categories (that "
        "is later, focused coding — avoid premature conceptual leaps).\n"
        "- Keep codes SHORT and active. Each names an action, process, or "
        "expressed meaning.\n"
        "- Code quickly and stay open; codes are provisional.\n"
        "- Take {subject_phrase}'s point of view: what is happening from "
        "where {subject_phrase} stands?"
    ),
    "open_json": """[
  {
    "open_code": "gerund-phrase initial code (action/process/meaning)",
    "text_passage": "exact corresponding text from the input"
  }
]""",
    "axial_role": (
        "a qualitative researcher performing Charmaz's FOCUSED coding "
        "(constructivist grounded theory)"
    ),
    "axial_intro": (
        "Let the categories emerge from the codes; do not impose the "
        "research goal as a category structure. If some initial codes are "
        "procedural noise that slipped through initial coding, exclude "
        "them rather than building a category around them."
    ),
    "axial_rules": (
        "CHARMAZ FOCUSED CODING — follow these rules:\n"
        "- Review the initial codes and SELECT the most SIGNIFICANT and "
        "most FREQUENT ones — the codes that carry the most analytic "
        "weight and best account for the data.\n"
        "- Use those selected codes to synthesize larger segments of data "
        "into focused categories. A focused category groups initial codes "
        "that speak to the same action or process.\n"
        "- Name focused categories in an active, meaning-preserving way."
    ),
    "axial_json": """[
  {
    "reasoning": "why these initial codes cohere; what significant/frequent action they share",
    "supporting_open_codes": ["exact", "initial", "codes", "grouped", "here"],
    "axial_category": "name of the focused category (active/process phrasing)"
  }
]""",
    "selective_role": (
        "a qualitative researcher performing Charmaz's THEORETICAL coding "
        "(constructivist grounded theory)"
    ),
    "selective_rules": (
        "CHARMAZ THEORETICAL CODING — follow these rules:\n"
        "- Consider how the FOCUSED categories relate to one another, and "
        "articulate an integrated, coherent theoretical account of "
        "{integration_closer}.\n"
        "- This is INTERPRETIVE and co-constructed: you are theorizing "
        "relationships among categories, not extracting a single "
        "pre-existing \"core\" via a fixed paradigm.\n"
        "- Ground every claim in the focused categories provided, not in "
        "the stated research goal.\n"
        "- Prefer processual language ({process_examples}) over static "
        "labels.\n\n"
        "Provide a short theoretical account: name the central process, "
        "then explain in a brief paragraph how the focused categories "
        "relate to constitute {integration_closer}."
    ),
}

TRADITION_BLOCKS = {"straussian": STRAUSSIAN, "charmaz": CHARMAZ}


# ============================================================================
# SKELETONS — shared by both traditions for open / axial / selective
# ============================================================================
_OPEN_SKELETON = """You are {open_role}.

STUDY CONTEXT (orienting focus only — do NOT treat as a list of expected findings or as an allowed set of categories):
- Research goal: {research_goal}
- Text genre: {genre_note}

This context tells you what the study is broadly about and what kind of text you are reading. It deliberately does NOT tell you which concepts, themes, or categories to expect — those must emerge from the data. Do not force the data toward the goal and do not invent codes to match it.

{open_rules}

{exclusion_rule}

Read the following {source_label} text and extract initial codes.

Output ONLY a valid JSON array of objects with the following exact structure:
{open_json}
Do not include any markdown formatting (like ```json), just the raw JSON array."""

_AXIAL_SKELETON = """You are {axial_role}.

STUDY CONTEXT (orienting focus only — do NOT treat as a list of expected findings or as an allowed set of categories):
- Research goal: {research_goal}

{axial_intro}

Review the following list of initial codes. Group them into categories by identifying relationships.
CRITICAL: You must track the source of every code.

{axial_rules}

Output ONLY a valid JSON array of objects with the following exact structure:
{axial_json}
Do not include any markdown formatting (like ```json), just the raw JSON array."""

_SELECTIVE_SKELETON = """You are {selective_role}.

STUDY CONTEXT (orienting focus only — do NOT treat as a list of expected findings or as an allowed set of categories):
- Research goal: {research_goal}

{selective_rules}"""


@dataclass(frozen=True)
class PromptSet:
    tradition: str
    module_name: str
    open: str
    axial: str
    selective: str
    dataset: str = DEFAULT_DATASET


def get_prompts(tradition: str = DEFAULT_TRADITION, dataset: str = DEFAULT_DATASET) -> PromptSet:
    """Return the three prompts (open, axial, selective) for a tradition,
    assembled fresh from TRADITION_BLOCKS[tradition] + DATASET_BLOCKS[dataset].

    Raises on unknown tradition or unknown dataset. `dataset` defaults to
    "silan".
    """
    trad_key = tradition.strip().lower()
    if trad_key not in TRADITIONS:
        raise ValueError(f"unknown tradition {tradition!r}; expected one of {TRADITIONS}")
    ds_key = dataset.strip().lower()
    if ds_key not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")

    T = TRADITION_BLOCKS[trad_key]
    D = DATASET_BLOCKS[ds_key]

    open_role = _fill(T["open_role"], **D)
    open_rules = _fill(T["open_rules"], **D)
    open_prompt = _fill(
        _OPEN_SKELETON,
        open_role=open_role, open_rules=open_rules, open_json=T["open_json"], **D,
    )

    axial_role = _fill(T["axial_role"], **D)
    axial_intro = _fill(T["axial_intro"], **D)
    axial_rules = _fill(T["axial_rules"], **D)
    axial_prompt = _fill(
        _AXIAL_SKELETON,
        axial_role=axial_role, axial_intro=axial_intro, axial_rules=axial_rules,
        axial_json=T["axial_json"], **D,
    )

    selective_role = _fill(T["selective_role"], **D)
    selective_rules = _fill(T["selective_rules"], **D)
    selective_prompt = _fill(_SELECTIVE_SKELETON, selective_role=selective_role,
                              selective_rules=selective_rules, **D)

    return PromptSet(
        tradition=trad_key,
        module_name="prompt_registry",
        open=open_prompt,
        axial=axial_prompt,
        selective=selective_prompt,
        dataset=ds_key,
    )


# ============================================================================
# CHARMAZ-ONLY: memo / reflection-loop / integration prompts
# ============================================================================
# These are what run_charmaz_arm actually calls (NOT .selective above, which
# main.py's Charmaz arm currently never uses). Dataset tokens are filled at
# build time; the runtime DATA placeholders below (e.g. {coded_data}) are
# left untouched here and filled later by the caller via _fill() again.
# Every one puts its data block LAST, after the JSON contract.

_INITIAL_MEMO_SKELETON = """You are a qualitative researcher writing INITIAL MEMOS in Kathy Charmaz's constructivist grounded theory.

A memo is you thinking on paper. It is not a summary. You reason about what an initial code means, what actions or processes it captures, and how incidents compare — BEFORE committing to any category.

YOUR TASK — for the significant/recurring codes, write a short analytic memo that:
- Explores the MEANING behind the code: what process or action is {subject_phrase} engaged in?
- Compares incidents: where does this code appear across {subject_plural}, and how does it vary?
- Notes whether the code feels like it is rising toward a tentative category, and why.
Reason first, in your own analytic voice; do not merely restate the code.

Output ONLY a valid JSON object with this exact structure:
{
  "memos": [
    {
      "focus_code": "the initial code this memo is about",
      "reasoning": "your analytic thinking: meaning, comparison across incidents, variation",
      "tentative_category": "a tentative category name if one is emerging, else null"
    }
  ]
}
Do not include markdown fences; output the raw JSON object only.

THE INITIAL CODES (with the data they came from):
{coded_data}"""

_ADVANCED_MEMO_SKELETON = """You are a qualitative researcher writing ADVANCED MEMOS in Charmaz's constructivist grounded theory.

You are refining conceptual categories through constant comparison. You reason about each focused category's PROPERTIES, the data it subsumes, and how it compares to sibling categories — and you are candid about where the analysis is still THIN.

YOUR TASK — for each focused category, write an advanced memo that:
- Articulates the category's PROPERTIES (its dimensions, its range of variation).
- Traces the data it subsumes and compares it against other categories (what distinguishes them? where do they overlap?).
- Surfaces underlying ASSUMPTIONS in how the category has been construed.
- Explicitly NAMES what is thin, ambiguous, or under-evidenced: which properties are asserted on little data, which distinctions are unclear, what you would want more data to resolve.

The thin-area naming is essential — it directs where the analysis looks next. Be specific and honest; do not paper over gaps.

Output ONLY a valid JSON object with this exact structure:
{
  "memos": [
    {
      "category": "the focused category this memo is about",
      "properties": ["property or dimension", "..."],
      "reasoning": "your analytic thinking: subsumed data, comparison to siblings, assumptions",
      "thin_areas": ["specific under-evidenced property / unclear distinction / what more data would resolve", "..."]
    }
  ]
}
Do not include markdown fences; output the raw JSON object only.

THE FOCUSED CATEGORIES (with their supporting codes):
{focused_categories}"""

_INITIAL_CODING_ITER_SKELETON = """You are a qualitative researcher performing INITIAL CODING (Charmaz constructivist GT) on a NEW slice of {source_label} data, as part of an ongoing analysis.

You are NOT starting from scratch. You already have an emerging set of focused categories. Through CONSTANT COMPARISON, you read this new data and code specifically what your existing categories do NOT already capture, and what elaborates or challenges them.

CHARMAZ INITIAL CODING RULES (unchanged):
- Code with GERUNDS (action/process words ending in "-ing"): what is {subject_phrase} doing, judging, or experiencing?
- Stay close to the data; take {subject_phrase}'s point of view.
- Keep codes short, active, provisional.
- Preserve a striking verbatim phrase as an in-vivo code where one is present (mark it).
- {exclusion_rule}

Read the new {source_label} data and produce initial codes, prioritizing what is NEW relative to your existing categories and what fills the thin areas named below.

Output ONLY a valid JSON array with this exact structure:
[
  {
    "open_code": "gerund-phrase initial code (action/process/meaning)",
    "text_passage": "exact corresponding text from the input",
    "in_vivo": false
  }
]
Set "in_vivo" true only when the code is a striking verbatim phrase from the source.
Do not include markdown fences; output the raw JSON array only.

YOUR EXISTING FOCUSED CATEGORIES:
{existing_categories}

AREAS YOUR ANALYSIS IS CURRENTLY THIN ON (look especially for data that speaks to these):
{thin_areas}"""

_FOCUSED_CODING_ITER_SKELETON = """You are a qualitative researcher performing FOCUSED CODING (Charmaz constructivist GT) as part of an ongoing, iterative analysis.

You have an existing set of focused categories. New initial codes have just been produced from a new slice of data. Through CONSTANT COMPARISON you now INTEGRATE the new codes into your category system: attach them where they fit, and form new categories ONLY where existing ones genuinely cannot hold them. You may also REVISE existing categories (rename, split, or merge) if the new data shows they no longer fit — codes are your construction, not fixed.

YOUR TASK:
- For each new code, integrate it into the most fitting existing category, or assign it to a new category if none fits.
- Where the new data warrants, revise categories: rename for a better fit, split an overloaded category, or merge redundant ones.
- Return the FULL updated set of focused categories (existing + revised + new), plus a typed list of the changes you made so the analytic trail is explicit.

Output ONLY a valid JSON object with this exact structure:
{
  "categories": [
    {
      "reasoning": "why these codes cohere; the significant/frequent action they share",
      "supporting_open_codes": ["exact", "initial", "codes", "in", "this", "category"],
      "axial_category": "focused category name (active/process phrasing)"
    }
  ],
  "changes_made": [
    {"type": "added",      "category": "name", "rationale": "..."},
    {"type": "elaborated", "category": "name", "new_properties": ["..."], "rationale": "..."},
    {"type": "renamed",    "from": "old name", "to": "new name", "rationale": "..."},
    {"type": "split",      "from": "old name", "into": ["a", "b"], "rationale": "..."},
    {"type": "merged",     "from": ["a", "b"], "into": "name", "rationale": "..."}
  ]
}
Return an empty "changes_made" list if the new codes all fit existing categories without revision.
Do not include markdown fences; output the raw JSON object only.

YOUR EXISTING FOCUSED CATEGORIES:
{existing_categories}

NEW INITIAL CODES TO INTEGRATE:
{new_codes}"""

_APPLICABILITY_TEST_SKELETON = """You are a qualitative researcher testing THEORETICAL SUFFICIENCY (saturation) in Charmaz's constructivist grounded theory.

You have a set of focused categories built from data seen so far. You will be given a FRESH slice of {source_label} data the categories have NOT been built from. Your job is to judge, honestly, how well the EXISTING categories account for this new data — this is how we detect whether the analysis has saturated.

YOUR TASK:
For EACH codeable unit, decide whether an existing category already accounts for its analytic content:
- "fits": an existing category captures it (name that category).
- "does_not_fit": it raises an action/process/meaning NOT captured by any existing category (briefly say what is new).

Judge honestly. A unit that genuinely introduces something new MUST be marked "does_not_fit" — do not stretch a category to cover it. Detecting new content is the entire purpose of this test; false "fits" would hide non-saturation.

Output ONLY a valid JSON object with this exact structure:
{
  "assignments": [
    {"unit_id": "the unit's id", "verdict": "fits | does_not_fit", "category": "fitting category name or null", "note": "if does_not_fit, what analytic content is new"}
  ]
}
Do not include markdown fences; output the raw JSON object only.

YOUR EXISTING FOCUSED CATEGORIES:
{existing_categories}

FRESH DATA TO TEST (each item is a codeable unit):
{fresh_units}"""

_MEMO_SORTING_SKELETON = """You are a qualitative researcher performing THEORETICAL SORTING and INTEGRATION in Charmaz's constructivist grounded theory — the final analytic step.

Your categories are now developed and your memos accumulated. You sort the memos to fit the theoretical categories, establish the logical relationships among categories, and integrate them into a coherent theoretical account.

YOUR TASK:
- Sort and integrate the memos under their categories.
- Establish how the categories relate to one another (which conditions, processes, or meanings connect them).
- Articulate an integrated, processual theoretical account of {integration_closer}.

This is interpretive and co-constructed: you are theorizing relationships among categories, grounded in the memos and categories provided — NOT extracting a single pre-existing core via a fixed paradigm, and NOT importing the research goal as structure. Prefer processual language ({process_examples}).

Write a short integrated theoretical account: name the central process, then explain in one or two paragraphs how the focused categories interlock to produce {integration_closer}.

THE FINAL FOCUSED CATEGORIES:
{focused_categories}

THE ACCUMULATED ADVANCED MEMOS:
{memos}"""


@dataclass(frozen=True)
class CharmazRecursionPrompts:
    dataset: str
    initial_memo: str
    advanced_memo: str
    initial_coding_iter: str
    focused_coding_iter: str
    applicability_test: str
    memo_sorting: str


def get_charmaz_recursion_prompts(dataset: str = DEFAULT_DATASET) -> CharmazRecursionPrompts:
    """Charmaz memo / reflection-loop / integration prompt templates for one
    dataset. Each returned string still contains its runtime DATA
    placeholder(s) (e.g. {coded_data}) — fill those with `_fill()` at call
    time. Dataset tokens ({subject_phrase} etc.) are already resolved.
    """
    ds_key = dataset.strip().lower()
    if ds_key not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    D = DATASET_BLOCKS[ds_key]
    return CharmazRecursionPrompts(
        dataset=ds_key,
        initial_memo=_fill(_INITIAL_MEMO_SKELETON, **D),
        advanced_memo=_fill(_ADVANCED_MEMO_SKELETON, **D),
        initial_coding_iter=_fill(_INITIAL_CODING_ITER_SKELETON, **D),
        focused_coding_iter=_fill(_FOCUSED_CODING_ITER_SKELETON, **D),
        applicability_test=_fill(_APPLICABILITY_TEST_SKELETON, **D),
        memo_sorting=_fill(_MEMO_SORTING_SKELETON, **D),
    )


# ============================================================================
# STRAUSSIAN-ONLY: empty-slot escalation ladder prompts
# ============================================================================
# Used by slot_recursion.py. Exercised for BOTH datasets: "silan" (rung 1 +
# rung 2 via cross-participant Q&A retrieval) and "semeval" (rung 1 skipped
# by default; rung 2 via cross-article sentence-grain retrieval). These
# skeletons render through the dataset-generic tokens below ({source_label},
# {subject_phrase}, ...) so no dataset-specific wording needed to change to
# support semeval.
#
# JUDGMENT CALL (flagged for review, not resolved here): _CROSS_RESOLVE_SKELETON's
# framing -- "could not be filled from the sources that originally
# contributed to it" -- assumes rung 1 already ran and failed. For SemEval
# with skip_rung1=True (the default), CROSS_RESOLVE is the FIRST attempt at
# the slot, not a second look after a failed first one, so this line is
# mildly inaccurate for that path (nothing "already tried and failed").
# Left as-is (harmless imprecision, doesn't change what the model is asked to
# do) rather than adding a dataset/skip_rung1-conditioned variant -- revisit
# if this framing is ever found to measurably affect model behavior.

SLOT_QUESTIONS = {
    "condition": "What conditions, circumstances, or situations give rise to this category?",
    "action_interaction": "What actions, behaviors, or interactions occur within this category?",
    "consequence": "What outcomes or consequences result from this category?",
}

_EXTRACT_SKELETON = """You are an expert qualitative researcher performing axial coding in a Straussian grounded-theory tradition.

You are revisiting ONE {source_label}'s full text to look for evidence that fills a specific GAP in an axial category's paradigm model. This is targeted re-reading, not re-coding.

YOUR TASK:
Read the full {source_label} below. Extract any concrete evidence — specific passages and your reasoning about them — that could speak to the gap(s) named below. Stay grounded in what {subject_phrase} actually says.

CRITICAL:
- You are NOT deciding or writing the final slot. You are gathering candidate evidence that will later be aggregated across many sources.
- If this {source_label} contains NO relevant evidence for a gap, say so explicitly by returning an empty "evidence" list for that slot. Do not invent or stretch. An honest "nothing here" is valuable.

Output ONLY a valid JSON object with this exact structure:
{
  "extractions": [
    {
      "slot": "condition | action_interaction | consequence",
      "evidence": [
        {"text_passage": "exact quote from the source", "reasoning": "why this bears on the slot"}
      ]
    }
  ]
}
Return an empty "evidence" list for any slot this source does not speak to.
Do not include any markdown formatting, just the raw JSON object.

THE AXIAL CATEGORY:
- Name: {axial_category}
- Grouping rationale: {reasoning}
- Already-established paradigm slots (context — do NOT re-derive these):
{filled_slots_context}

THE GAP(S) TO INVESTIGATE (extract evidence for these ONLY):
{empty_slot_questions}"""

_AGGREGATE_RESOLVE_SKELETON = """You are an expert qualitative researcher performing axial coding in a Straussian grounded-theory tradition.

Candidate evidence for one or more EMPTY paradigm slots of an axial category has been gathered from multiple sources. Your job is to aggregate it and resolve each slot — or judge that the evidence is insufficient and the slot must remain empty.

YOUR TASK:
For each slot, synthesize the evidence into a concise paradigm-model statement grounded in the aggregated passages — OR return null if the evidence is too thin, absent, or contradictory to support an honest fill.

CRITICAL:
- Returning null for a slot is a legitimate, valuable outcome. The point is to find out whether the paradigm model actually fits this category, not to force a fill. Do NOT confabulate to avoid an empty slot.

Output ONLY a valid JSON object with this exact structure:
{
  "resolutions": [
    {"slot": "condition | action_interaction | consequence", "value": "resolved statement OR null", "reasoning": "what the aggregated evidence did or did not support"}
  ]
}
Do not include any markdown formatting, just the raw JSON object.

THE AXIAL CATEGORY:
- Name: {axial_category}
- Already-established paradigm slots (context):
{filled_slots_context}

THE SLOT(S) TO RESOLVE:
{empty_slot_questions}

AGGREGATED CANDIDATE EVIDENCE (across sources):
{aggregated_evidence}"""

_CROSS_RESOLVE_SKELETON = """You are an expert qualitative researcher performing axial coding in a Straussian grounded-theory tradition.

A paradigm slot of an axial category could not be filled from the sources that originally contributed to it. As a wider search, similar excerpts from OTHER sources have been retrieved. Use them to resolve the slot, or judge that they too are insufficient.

YOUR TASK:
For each slot, synthesize a paradigm-model statement grounded in these excerpts — OR return null if they do not support an honest fill.

CRITICAL:
- These excerpts come from sources that were NOT originally grouped into this category; treat them as broader context, and only fill a slot if the evidence genuinely applies. Returning null remains a legitimate, valuable outcome.

Output ONLY a valid JSON object with this exact structure:
{
  "resolutions": [
    {"slot": "condition | action_interaction | consequence", "value": "resolved statement OR null", "reasoning": "what the retrieved evidence did or did not support"}
  ]
}
Do not include any markdown formatting, just the raw JSON object.

THE AXIAL CATEGORY:
- Name: {axial_category}
- Already-established paradigm slots (context):
{filled_slots_context}

THE SLOT(S) TO RESOLVE:
{empty_slot_questions}

RETRIEVED SIMILAR EXCERPTS (from other sources):
{retrieved_qa}"""


@dataclass(frozen=True)
class StraussianSlotPrompts:
    dataset: str
    extract: str
    aggregate_resolve: str
    cross_resolve: str


def get_straussian_slot_prompts(dataset: str = DEFAULT_DATASET) -> StraussianSlotPrompts:
    """Straussian empty-slot escalation ladder prompt templates for one
    dataset. Each returned string still contains its runtime placeholders
    (e.g. {filled_slots_context}, {aggregated_evidence}) — fill those with
    `_fill()` at call time.
    """
    ds_key = dataset.strip().lower()
    if ds_key not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    D = DATASET_BLOCKS[ds_key]
    return StraussianSlotPrompts(
        dataset=ds_key,
        extract=_fill(_EXTRACT_SKELETON, **D),
        aggregate_resolve=_fill(_AGGREGATE_RESOLVE_SKELETON, **D),
        cross_resolve=_fill(_CROSS_RESOLVE_SKELETON, **D),
    )
