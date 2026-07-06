# prompts.py

OPEN_CODING_PROMPT = """You are an expert qualitative researcher performing Grounded Theory Analysis.
Read the following text chunk from an interview. Extract initial 'open codes'.
Return ONLY a comma-separated list of short codes. Do not include introductory text."""

AXIAL_CODING_PROMPT = """You are an expert qualitative researcher.
Review the following list of open codes extracted from a set of interviews. Group them into 'axial codes' by identifying relationships. 
Map them using a "Condition -> Action/Interaction -> Consequence" format.
Output a structured markdown summary of these relationships."""

SELECTIVE_CODING_PROMPT = """You are an expert qualitative researcher.
Review the following axial relationships. Synthesize them into a single 'Selective Code' or Core Category that explains the central phenomenon of the entire dataset.
Provide the Core Category name, followed by a brief paragraph explaining the grounded theory."""