# prompt_registry.py
"""
Parameter-driven prompt selection across GT traditions and datasets.

Replaces the manual comment-toggle of imports in gta_pipeline.py. Each
tradition module exposes the SAME three variable names — OPEN_CODING_PROMPT,
AXIAL_CODING_PROMPT, SELECTIVE_CODING_PROMPT — so downstream code is unchanged;
only the source module differs.

  tradition="straussian" -> prompts.py          (open / axial / selective)
  tradition="charmaz"    -> prompts_charmaz.py  (initial / focused / theoretical,
                                                 exported under the same names)

Independently, `dataset` selects which STUDY CONTEXT block is injected
into whichever tradition's prompts were selected above:

  dataset="silan"   -> (default) the tradition module's own STUDY CONTEXT,
                        completely unmodified -- byte-identical to calling
                        get_prompts(tradition) before `dataset` existed.
  dataset="semeval" -> the SemEval-2025 Task 10 STUDY CONTEXT block from
                        study_contexts.py is spliced in in place of the
                        tradition module's STUDY CONTEXT block; every other
                        line (coding rules, JSON output contract, field
                        names) is left byte-identical to the Silan version.

tradition and dataset are independent axes: dataset never changes which
module is picked for a tradition, and tradition never changes which
STUDY CONTEXT block is used for a dataset.

Usage
-----
    from prompt_registry import get_prompts
    P = get_prompts("straussian")                      # Silan (default), unchanged
    P = get_prompts("straussian", dataset="semeval")    # same tradition, SemEval focus
    call_llm(P.open, chunk_text)
    call_llm(P.axial, combined_codes)

`.open/.axial/.selective` are stable accessors; the underlying constant names
stay identical for backward compatibility with any code still importing them
directly.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

from study_contexts import DATASET_CONTEXTS, swap_study_context, swap_integration_closers

TRADITIONS = {
    "straussian": "prompts",
    "charmaz": "prompts_charmaz",
}
DEFAULT_TRADITION = "straussian"

# "silan" is the original dataset baked into prompts.py / prompts_charmaz.py.
# It is handled as a no-op (module prompts pass straight through) rather than
# being listed in DATASET_CONTEXTS, since there is no block to swap in for it.
DATASETS = ("silan", *DATASET_CONTEXTS.keys())
DEFAULT_DATASET = "silan"


@dataclass(frozen=True)
class PromptSet:
    tradition: str
    module_name: str
    open: str
    axial: str
    selective: str
    dataset: str = DEFAULT_DATASET


def get_prompts(
    tradition: str = DEFAULT_TRADITION, dataset: str = DEFAULT_DATASET
) -> PromptSet:
    """Return the three prompts for a tradition, focused on a dataset.

    Raises on unknown tradition or unknown dataset. `dataset` defaults to
    "silan", which preserves the exact prior behavior of this function
    (prompts.py / prompts_charmaz.py are returned unmodified) for full
    backward compatibility with existing callers that only pass `tradition`.
    """
    key = tradition.strip().lower()
    if key not in TRADITIONS:
        raise ValueError(
            f"unknown tradition {tradition!r}; expected one of {sorted(TRADITIONS)}"
        )
    ds_key = dataset.strip().lower()
    if ds_key not in DATASETS:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}"
        )

    mod = importlib.import_module(TRADITIONS[key])
    try:
        open_prompt = mod.OPEN_CODING_PROMPT
        axial_prompt = mod.AXIAL_CODING_PROMPT
        selective_prompt = mod.SELECTIVE_CODING_PROMPT
    except AttributeError as e:
        raise AttributeError(
            f"{TRADITIONS[key]}.py is missing a required prompt constant: {e}"
        )

    if ds_key != "silan":
        new_context = DATASET_CONTEXTS[ds_key]
        open_prompt = swap_study_context(open_prompt, new_context)
        axial_prompt = swap_study_context(axial_prompt, new_context)
        selective_prompt = swap_study_context(selective_prompt, new_context)

    # Some tradition prompts also name the Silan phenomenon in a closing
    # instruction sentence OUTSIDE the STUDY CONTEXT block (currently only
    # prompts_charmaz.py's THEORETICAL_CODING_PROMPT / .selective) -- swap
    # those too. No-op for dataset=="silan" and for any prompt that doesn't
    # contain one of the known fragments (i.e. every Straussian prompt, and
    # Charmaz's initial/focused prompts).
    open_prompt = swap_integration_closers(open_prompt, ds_key)
    axial_prompt = swap_integration_closers(axial_prompt, ds_key)
    selective_prompt = swap_integration_closers(selective_prompt, ds_key)

    return PromptSet(
        tradition=key,
        module_name=TRADITIONS[key],
        open=open_prompt,
        axial=axial_prompt,
        selective=selective_prompt,
        dataset=ds_key,
    )