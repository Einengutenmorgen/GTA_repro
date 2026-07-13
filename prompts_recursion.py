# prompts_recursion.py
"""
Prompts for the Straussian empty-slot escalation ladder (paradigm recall).

Three distinct call shapes, used in sequence when a paradigm slot
(condition / action_interaction / consequence) comes back empty:

  1. EXTRACT  (rung 1, one call PER contributing interview)
     Given one full interview + the category + its already-filled slots as
     context, extract candidate evidence + reasoning for ONLY the empty
     slot(s). Does NOT decide the slot — defers to aggregation. May return
     nothing (honest empty).

  2. AGGREGATE_RESOLVE (rung 1 close, one call over all extractions)
     Given the per-interview extractions, resolve the empty slot(s) — or
     declare them still unfillable. Allowed to return empty.

  3. CROSS_RESOLVE (rung 2+, one call over k retrieved cross-participant Q&A)
     Given similar Q&A blocks from OTHER participants, resolve directly — or
     declare still empty. Allowed to return empty.

Design invariants
-----------------
- "Still empty after looking" is always a permitted output. The study measures
  whether the paradigm fits; filling a slot with thin confabulation because
  some text was retrieved would corrupt exactly that finding. Every resolving
  prompt must explicitly license an empty return.
- Multiple empty slots for one category are handled in a SINGLE per-interview
  extraction call (batched) to avoid re-feeding the same interview per slot.
- Output is strict JSON, matching the pipeline's existing parse-and-clean path.

The {SLOT_DEFINITIONS} block is filled programmatically with only the slots
that are actually empty, so the model is never asked about a filled slot.
"""

# Human-readable definitions injected per empty slot.
SLOT_QUESTIONS = {
    "condition": "What conditions, circumstances, or situations give rise to this category?",
    "action_interaction": "What actions, behaviors, or interactions occur within this category?",
    "consequence": "What outcomes or consequences result from this category?",
}


EXTRACT_PROMPT = """You are an expert qualitative researcher performing axial coding in a Straussian grounded-theory tradition.

You are revisiting ONE participant's full interview to look for evidence that fills a specific GAP in an axial category's paradigm model. This is targeted re-reading, not re-coding.

THE AXIAL CATEGORY:
- Name: {axial_category}
- Grouping rationale: {reasoning}
- Already-established paradigm slots (context — do NOT re-derive these):
{filled_slots_context}

THE GAP(S) TO INVESTIGATE (extract evidence for these ONLY):
{empty_slot_questions}

YOUR TASK:
Read the full interview below. Extract any concrete evidence — specific passages and your reasoning about them — that could speak to the gap(s). Stay grounded in what THIS participant actually says.

CRITICAL:
- You are NOT deciding or writing the final slot. You are gathering candidate evidence that will later be aggregated across many interviews.
- If this interview contains NO relevant evidence for a gap, say so explicitly by returning an empty "evidence" list for that slot. Do not invent or stretch. An honest "nothing here" is valuable.

Output ONLY a valid JSON object with this exact structure:
{{
  "extractions": [
    {{
      "slot": "condition | action_interaction | consequence",
      "evidence": [
        {{"text_passage": "exact quote from the interview", "reasoning": "why this bears on the slot"}}
      ]
    }}
  ]
}}
Return an empty "evidence" list for any slot this interview does not speak to.
Do not include any markdown formatting, just the raw JSON object."""


AGGREGATE_RESOLVE_PROMPT = """You are an expert qualitative researcher performing axial coding in a Straussian grounded-theory tradition.

Candidate evidence for one or more EMPTY paradigm slots of an axial category has been gathered from multiple participants' interviews. Your job is to aggregate it and resolve each slot — or judge that the evidence is insufficient and the slot must remain empty.

THE AXIAL CATEGORY:
- Name: {axial_category}
- Already-established paradigm slots (context):
{filled_slots_context}

THE SLOT(S) TO RESOLVE:
{empty_slot_questions}

AGGREGATED CANDIDATE EVIDENCE (across interviews):
{aggregated_evidence}

YOUR TASK:
For each slot, synthesize the evidence into a concise paradigm-model statement grounded in the aggregated passages — OR return null if the evidence is too thin, absent, or contradictory to support an honest fill.

CRITICAL:
- Returning null for a slot is a legitimate, valuable outcome. The point is to find out whether the paradigm model actually fits this category, not to force a fill. Do NOT confabulate to avoid an empty slot.

Output ONLY a valid JSON object with this exact structure:
{{
  "resolutions": [
    {{"slot": "condition | action_interaction | consequence", "value": "resolved statement OR null", "reasoning": "what the aggregated evidence did or did not support"}}
  ]
}}
Do not include any markdown formatting, just the raw JSON object."""


CROSS_RESOLVE_PROMPT = """You are an expert qualitative researcher performing axial coding in a Straussian grounded-theory tradition.

A paradigm slot of an axial category could not be filled from the interviews of the participants who originally contributed to it. As a wider search, Q&A excerpts addressing SIMILAR interview questions — drawn from OTHER participants — have been retrieved. Use them to resolve the slot, or judge that they too are insufficient.

THE AXIAL CATEGORY:
- Name: {axial_category}
- Already-established paradigm slots (context):
{filled_slots_context}

THE SLOT(S) TO RESOLVE:
{empty_slot_questions}

RETRIEVED Q&A FROM OTHER PARTICIPANTS (similar questions):
{retrieved_qa}

YOUR TASK:
For each slot, synthesize a paradigm-model statement grounded in these excerpts — OR return null if they do not support an honest fill.

CRITICAL:
- These excerpts come from participants who were NOT originally grouped into this category; treat them as broader context, and only fill a slot if the evidence genuinely applies. Returning null remains a legitimate, valuable outcome.

Output ONLY a valid JSON object with this exact structure:
{{
  "resolutions": [
    {{"slot": "condition | action_interaction | consequence", "value": "resolved statement OR null", "reasoning": "what the retrieved evidence did or did not support"}}
  ]
}}
Do not include any markdown formatting, just the raw JSON object."""
