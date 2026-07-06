# prompts.py

OPEN_CODING_PROMPT = """You are an expert qualitative researcher performing Grounded Theory Analysis. 
Read the following text from an interview and extract initial 'open codes'.

Output ONLY a valid JSON array of objects with the following exact structure:
[
  {
    "open_code": "basic concept, behavior, emotion, or incident",
    "text_passage": "Exact corresponding text passage from the input"
  }
]
Do not include any markdown formatting (like ```json), just the raw JSON array."""

AXIAL_CODING_PROMPT = """You are an expert qualitative researcher. Review the following list of open codes extracted from a set of interviews. Group them into 'axial codes' by identifying relationships. 
CRITICAL: You must track the source of every code.
Output ONLY a valid JSON array of objects with the following exact structure:
[
  {
    "reasoning": "Explanation or thought process of grouping",
    "supporting_open_codes": ["list", "of", "exact", "open", "codes", "used"],
    "axial_category": "Name of the overarching category", 
    "condition": "What conditions give rise to this?", 
    "action_interaction": "What behaviors or interactions occur?", 
    "consequence": "What is the outcome?"
  }
]
Do not include any markdown formatting (like ```json), just the raw JSON array."""

SELECTIVE_CODING_PROMPT = """You are an expert qualitative researcher.
Review the following axial relationships. Synthesize them into a single 'Selective Code' or Core Category that explains the central phenomenon of the entire dataset.
Provide the Core Category name, followed by a brief paragraph explaining the grounded theory."""