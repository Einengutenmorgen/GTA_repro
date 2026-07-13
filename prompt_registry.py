# prompt_registry.py
"""
Parameter-driven prompt selection across GT traditions.

Replaces the manual comment-toggle of imports in gta_pipeline.py. Each
tradition module exposes the SAME three variable names — OPEN_CODING_PROMPT,
AXIAL_CODING_PROMPT, SELECTIVE_CODING_PROMPT — so downstream code is unchanged;
only the source module differs.

  tradition="straussian" -> prompts.py          (open / axial / selective)
  tradition="charmaz"    -> prompts_charmaz.py  (initial / focused / theoretical,
                                                 exported under the same names)

Usage
-----
    from prompt_registry import get_prompts
    P = get_prompts("straussian")
    call_llm(P.open, chunk_text)
    call_llm(P.axial, combined_codes)

`.open/.axial/.selective` are stable accessors; the underlying constant names
stay identical for backward compatibility with any code still importing them
directly.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

TRADITIONS = {
    "straussian": "prompts",
    "charmaz": "prompts_charmaz",
}
DEFAULT_TRADITION = "straussian"


@dataclass(frozen=True)
class PromptSet:
    tradition: str
    module_name: str
    open: str
    axial: str
    selective: str


def get_prompts(tradition: str = DEFAULT_TRADITION) -> PromptSet:
    """Return the three prompts for a tradition. Raises on unknown tradition."""
    key = tradition.strip().lower()
    if key not in TRADITIONS:
        raise ValueError(
            f"unknown tradition {tradition!r}; expected one of {sorted(TRADITIONS)}"
        )
    mod = importlib.import_module(TRADITIONS[key])
    try:
        return PromptSet(
            tradition=key,
            module_name=TRADITIONS[key],
            open=mod.OPEN_CODING_PROMPT,
            axial=mod.AXIAL_CODING_PROMPT,
            selective=mod.SELECTIVE_CODING_PROMPT,
        )
    except AttributeError as e:
        raise AttributeError(
            f"{TRADITIONS[key]}.py is missing a required prompt constant: {e}"
        )
