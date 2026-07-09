# prompts_charmaz.py
# Constructivist Grounded Theory (Charmaz 2006/2014) prompt set.
# Replaces the Straussian open/axial/selective + condition-action-consequence
# paradigm with Charmaz's initial -> focused -> theoretical coding.
#
# Import-compatible with gta_pipeline.py: exposes OPEN/AXIAL/SELECTIVE names
# (mapped to initial/focused/theoretical) so no pipeline code changes.
# JSON output contracts are preserved so alignment.py still parses.

# --- INITIAL CODING (Charmaz's initial/line-by-line, replaces "open") --------
OPEN_CODING_PROMPT = """You are a qualitative researcher coding interview data using Kathy Charmaz's CONSTRUCTIVIST grounded theory (initial coding).

STUDY CONTEXT (orienting focus only — NOT a list of expected findings):
- Topic: how people conceptualize and experience relationship quality.
- Aim: surface how participants themselves define and describe what makes a relationship high or low in quality, in their own words.
- Genre: semi-structured research interview. Interviewer turns are prompts, not data.

CHARMAZ INITIAL CODING — follow these rules:
- Code with GERUNDS (action words ending in "-ing"). Name what the participant is DOING, feeling, or experiencing, not the topic. Write "Feeling safe with partner", "Weighing trust against fidelity", "Distinguishing love from infatuation" — NOT "Safety", "Trust", "Love".
- Stay close to the data. Code the specific action or meaning in THIS passage; do not jump to abstract themes or categories (that is later, focused coding — avoid premature conceptual leaps).
- Keep codes SHORT and active. Each names a action, process, or experienced meaning.
- Code quickly and stay open; codes are provisional.
- Take the participant's point of view: what is happening from where they stand?
- IGNORE interviewer prompts and any procedural/administrative talk (consent, logistics). Code only participant meaning bearing on the topic.

Read the interview passage and produce initial codes.

Output ONLY a valid JSON array with this exact structure:
[
  {
    "open_code": "gerund-phrase initial code (action/process/meaning)",
    "text_passage": "exact corresponding text from the input"
  }
]
Do not include markdown fences; output the raw JSON array only."""

# --- FOCUSED CODING (Charmaz's focused, replaces "axial") --------------------
# NOTE: no condition/action/consequence paradigm (Charmaz rejects it).
AXIAL_CODING_PROMPT = """You are a qualitative researcher performing Charmaz's FOCUSED coding (constructivist grounded theory).

STUDY CONTEXT (orienting only, not expected findings):
- Topic: how people conceptualize and experience relationship quality.
- Aim: how participants define what makes a relationship high or low in quality, in their own words.

CHARMAZ FOCUSED CODING — follow these rules:
- Review the initial codes and SELECT the most SIGNIFICANT and most FREQUENT ones — the codes that carry the most analytic weight and best account for the data.
- Use those selected codes to synthesize larger segments of data into focused categories. A focused category groups initial codes that speak to the same action or process.
- Name focused categories in an active, meaning-preserving way.
- Exclude any procedural noise that slipped through initial coding rather than building a category around it.

Output ONLY a valid JSON array with this exact structure:
[
  {
    "reasoning": "why these initial codes cohere; what significant/frequent action they share",
    "supporting_open_codes": ["exact", "initial", "codes", "grouped", "here"],
    "axial_category": "name of the focused category (active/process phrasing)"
  }
]
Do not include markdown fences; output the raw JSON array only."""

# --- THEORETICAL CODING (Charmaz's theoretical, replaces "selective") --------
SELECTIVE_CODING_PROMPT = """You are a qualitative researcher performing Charmaz's THEORETICAL coding (constructivist grounded theory).

STUDY CONTEXT (orienting only):
- Topic: how people conceptualize and experience relationship quality.

CHARMAZ THEORETICAL CODING — follow these rules:
- Consider how the FOCUSED categories relate to one another, and articulate an integrated, coherent theoretical account of how participants construct relationship quality.
- This is INTERPRETIVE and co-constructed: you are theorizing relationships among categories, not extracting a single pre-existing "core" via a fixed paradigm.
- Ground every claim in the focused categories provided, not in the stated study topic.
- Prefer processual language (how quality is built, sustained, eroded) over static labels.

Provide a short theoretical account: name the central process, then explain in a brief paragraph how the focused categories relate to constitute participants' lived sense of relationship quality."""
