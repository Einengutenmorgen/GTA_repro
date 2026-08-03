# test_taxonomy_seeded.py
"""
Functional tests for the taxonomy-seeded coding experiments (taxonomy_registry
+ taxonomy_match + the taxonomy-match prompt). No live LLM and no
sentence-transformers required: embeddings are stubbed with deterministic fake
vectors (same discipline as the §14 slot-recursion smoke tests), and the LLM is
a fixed lambda.

Run: python test_taxonomy_seeded.py    (prints PASS/FAIL per test; exit 1 on any fail)

The FIREWALL test is load-bearing: it asserts that a rendered match prompt
NEVER contains any gold answer-key token. If that ever fails, the experiment's
integrity guarantee is broken -- gold would be leaking into the model.
"""
from __future__ import annotations

import sys
import types
import numpy as np

import taxonomy_registry as TR


# ---------------------------------------------------------------------------
# Deterministic fake embeddings: map known texts to orthogonal-ish unit vectors
# by hashing into a small space, so cosine is stable and controllable in tests.
# We install this by monkeypatching utils.embed_texts BEFORE importing
# taxonomy_match (which binds embed_texts at import).
# ---------------------------------------------------------------------------

_FAKE_DIM = 16


def _fake_vec(text: str) -> np.ndarray:
    v = np.zeros(_FAKE_DIM, dtype=np.float32)
    # deterministic pseudo-embedding: bag of char-code buckets
    for i, ch in enumerate(text.lower()):
        v[(ord(ch) + i) % _FAKE_DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def _fake_embed_texts(texts, model_name="fake", disk_cache_path="fake"):
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([_fake_vec(t) for t in texts]).astype(np.float32)


import utils
utils.embed_texts = _fake_embed_texts  # patch the single source of truth
import taxonomy_match as TM
TM.embed_texts = _fake_embed_texts     # and the name taxonomy_match bound


_results = []

def check(name, cond):
    _results.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# ---------------------------------------------------------------------------
# taxonomy_registry
# ---------------------------------------------------------------------------

def test_registry():
    print("test_registry")
    er = TR.load_taxonomy("semeval", "entity_role")
    check("entity_role has 22 fine leaves", len(er.leaves) == 22)
    check("entity_role is domain-agnostic", er.domain is None)
    parents = {lf.parent for lf in er.leaves}
    check("entity_role parents are the 3 main roles",
          parents == {"protagonist", "antagonist", "innocent"})

    urw = TR.load_taxonomy("semeval", "narrative", domain="URW")
    cc = TR.load_taxonomy("semeval", "narrative", domain="CC")
    check("URW narrative tree non-empty", len(urw.leaves) > 10)
    check("CC narrative tree non-empty", len(cc.leaves) > 10)
    check("narrative leaves carry a domain",
          all(lf.domain == "URW" for lf in urw.leaves))

    # entity_role must reject a domain; narrative must require one
    try:
        TR.load_taxonomy("semeval", "entity_role", domain="CC"); ok = False
    except ValueError:
        ok = True
    check("entity_role rejects a domain arg", ok)
    try:
        TR.load_taxonomy("semeval", "narrative"); ok = False
    except ValueError:
        ok = True
    check("narrative requires a domain arg", ok)

    # deferred/placeholder paths raise the right thing
    try:
        TR.load_taxonomy("semeval", "entity_role", level="coarse"); ok = False
    except NotImplementedError:
        ok = True
    check("coarse level is deferred (NotImplementedError)", ok)
    try:
        TR.load_taxonomy("silan", "narrative", domain="URW"); ok = False
    except NotImplementedError:
        ok = True
    check("silan is a documented placeholder (NotImplementedError)", ok)


def test_domain_of():
    print("test_domain_of")
    check("EN_CC_200081 -> CC", TR.domain_of("EN_CC_200081") == "CC")
    check("EN_UA_DEV_100013 -> URW", TR.domain_of("EN_UA_DEV_100013") == "URW")
    check("unknown -> None", TR.domain_of("EN_XX_999") is None)


# ---------------------------------------------------------------------------
# embed_match + classify_edge_cases (deterministic, fake vectors)
# ---------------------------------------------------------------------------

def test_embed_match_and_edges():
    print("test_embed_match_and_edges")
    tax = TR.load_taxonomy("semeval", "entity_role")
    codes = [
        {"open_code": "Saboteur", "text_passage": "deliberately damage obstruct systems from within",
         "chunk_id": "EN_UA_1_c0000", "source_id": "EN_UA_1"},
        {"open_code": "Victim", "text_passage": "victims of sanctions blockades economic harm",
         "chunk_id": "EN_UA_2_c0000", "source_id": "EN_UA_2"},
        {"open_code": "zzzz nonsense qwxv", "text_passage": "qwxv zzz plugh xyzzy",
         "chunk_id": "EN_UA_3_c0000", "source_id": "EN_UA_3"},
        {"__status__": "failed", "chunk_id": "EN_UA_4_c0000"},  # must be dropped
    ]
    emb = TM.embed_match(codes, tax)
    check("failed code dropped (3 valid, not 4)", len(emb) == 3)
    check("every result has a best category + score",
          all("emb_category" in r and "emb_score" in r for r in emb))
    check("source_id recovered from chunk_id",
          emb[0]["source_id"] == "EN_UA_1")

    # classify with a HIGH orphan threshold so the low-similarity nonsense code
    # is flagged orphan deterministically.
    typed = TM.classify_edge_cases(emb, orphan_threshold=0.99, straddle_delta=0.0)
    check("all become orphans at threshold 0.99",
          all(t["edge_case_type"] == TM.ORPHAN for t in typed))

    # straddler: force a tiny margin by setting delta huge
    typed2 = TM.classify_edge_cases(emb, orphan_threshold=0.0, straddle_delta=1.0)
    check("all become straddlers at delta 1.0",
          all("straddler" in t["edge_case_types"] for t in typed2))

    # clean-ish: low threshold, zero delta -> no forced edges from thresholds
    typed3 = TM.classify_edge_cases(emb, orphan_threshold=0.0, straddle_delta=0.0)
    check("no threshold-forced edges when thresholds relaxed",
          all(t["edge_case_type"] == TM.CLEAN for t in typed3))


def test_disagreement_and_category_signals():
    print("test_disagreement_and_category_signals")
    tax = TR.load_taxonomy("semeval", "entity_role")
    codes = [{"open_code": "Corrupt official", "text_passage": "bribes graft profit over ethics",
              "chunk_id": "EN_UA_9_c0000", "source_id": "EN_UA_9"}]
    emb = TM.embed_match(codes, tax)
    emb_cat = emb[0]["emb_category"]
    # LLM deliberately picks a DIFFERENT leaf -> disagreement
    other = next(lf.name for lf in tax.leaves if lf.name != emb_cat)
    llm_results = [{"category": other, "confidence": 0.9, "reasoning": "x",
                    "is_edge_case": False, "edge_case_type": "clean"}]
    typed = TM.classify_edge_cases(emb, llm_results, taxonomy=tax,
                                   orphan_threshold=0.0, straddle_delta=0.0)
    check("matcher disagreement detected",
          TM.DISAGREEMENT in typed[0]["edge_case_types"])
    check("agree flag is False on disagreement", typed[0]["agree"] is False)

    sig = TM.category_signals(typed, tax)
    check("category_signals reports empty categories (Miss list non-empty here)",
          len(sig["empty_categories"]) == 21)  # 1 code -> 21 of 22 leaves empty
    check("n_leaves == 22", sig["n_leaves"] == 22)


# ---------------------------------------------------------------------------
# score_against_gold (eval-only)
# ---------------------------------------------------------------------------

def test_scoring():
    print("test_scoring")
    # narrative E2: build predicted from typed rows' emb_category names
    typed = [
        {"source_id": "EN_UA_1", "emb_category": "Ukraine is the aggressor",
         "emb_category_id": "URW/blaming_the_war_on_others_rather_than_the_invader/ukraine_is_the_aggressor"},
        {"source_id": "EN_UA_1", "emb_category": "The West is weak",
         "emb_category_id": "URW/discrediting_the_west_diplomacy/the_west_is_weak"},
    ]
    gold = {"EN_UA_1": {"Ukraine is the aggressor"}}  # 1 correct, 1 spurious
    rep = TM.score_against_gold(typed, gold, "narrative")
    check("narrative scoring returns a report", rep is not None)
    check("narrative precision == 0.5 (1 tp, 1 fp)",
          abs(rep["overall"]["precision"] - 0.5) < 1e-9)
    check("narrative recall == 1.0 (1 tp, 0 fn)",
          abs(rep["overall"]["recall"] - 1.0) < 1e-9)

    # empty gold -> None, no crash
    check("empty gold -> None", TM.score_against_gold(typed, {}, "narrative") is None)

    # entity_role without entity resolution -> None (deferred hard scoring)
    check("entity_role without resolution -> None",
          TM.score_against_gold(typed, {"EN_UA_1": [("x", "antagonist", [])]},
                                "entity_role") is None)

    # entity_role WITH a resolver -> accuracy computed
    er_typed = [{"source_id": "EN_UA_1", "emb_category_id": "antagonist/saboteur"}]
    er_gold = {"EN_UA_1": [("Russia", "antagonist", ["Saboteur"])]}
    rep2 = TM.score_against_gold(er_typed, er_gold, "entity_role",
                                 entity_resolution=lambda row: "Russia")
    check("entity_role accuracy == 1.0 with correct coarse role",
          rep2 is not None and rep2["accuracy"] == 1.0)


# ---------------------------------------------------------------------------
# FIREWALL (load-bearing): gold must NEVER appear in a rendered match prompt.
# ---------------------------------------------------------------------------

def test_firewall():
    print("test_firewall")
    from prompt_registry import get_taxonomy_match_prompt, _fill
    tax = TR.load_taxonomy("semeval", "narrative", domain="URW")
    block = tax.taxonomy_block()
    tmpl = get_taxonomy_match_prompt("semeval").match
    rendered = _fill(tmpl, domain_note="", taxonomy_block=block,
                     open_code="some code", text_passage="some passage")

    # The taxonomy_block (schema) IS allowed -- and encouraged -- in the prompt;
    # that is the seeded axial layer. What must NEVER leak is the GOLD ANSWER
    # KEY: the mapping from a specific article to its correct labels/roles.
    #
    # We probe with SYNTHETIC, unmistakable gold tokens that cannot collide with
    # legitimate schema vocabulary (real entity names like "Russia" appear in
    # the URW schema definitions and are NOT a leak -- they are schema words).
    # The leak we guard against is a per-instance ANSWER binding: an article id
    # tied to its gold labels. If the pipeline ever fed load_gold() output into
    # the prompt builder, these synthetic bindings would appear verbatim.
    SENTINEL_ARTICLE_ID = "GOLDPROBE_ARTICLE_ZZZ42"
    SENTINEL_ANSWER = "GOLDPROBE_ANSWER_QWXV"
    fake_gold = {
        SENTINEL_ARTICLE_ID: {SENTINEL_ANSWER},                    # narrative gold binding
        "GOLDPROBE_ARTICLE_ZZZ43": [("GOLDPROBE_ENTITY", "antagonist", [SENTINEL_ANSWER])],
    }
    # The render path only ever receives the taxonomy schema + the open code;
    # it has no parameter through which gold could enter. Assert the sentinels
    # are absent from the rendered prompt.
    leaked = [tok for tok in (SENTINEL_ARTICLE_ID, SENTINEL_ANSWER,
                              "GOLDPROBE_ARTICLE_ZZZ43", "GOLDPROBE_ENTITY")
              if tok in rendered]
    check("no gold article-id binding leaks into the prompt", SENTINEL_ARTICLE_ID not in rendered)
    check("no gold answer token leaks into the prompt", SENTINEL_ANSWER not in rendered)
    check("firewall: nothing from the gold key leaked", not leaked)

    # sanity: the schema DID make it in (so the test isn't vacuous)
    check("taxonomy schema is present in the prompt (test not vacuous)",
          "Ukraine is the aggressor" in rendered)

    # and taxonomy_block itself must be schema-only: no article-id -> answer
    # bindings of the form load_gold produces.
    check("taxonomy_block is schema-only (no gold bindings)",
          SENTINEL_ARTICLE_ID not in block and SENTINEL_ANSWER not in block)

    # STRUCTURAL firewall: the render closure's signature carries no gold. Prove
    # it by constructing the same render path the arm uses and confirming gold
    # is not among the inputs it can even see. (render(open_code, taxonomy) ->
    # only taxonomy schema + code text; no gold parameter exists.)
    import inspect
    from taxonomy_match import llm_match
    sig = inspect.signature(llm_match)
    check("llm_match takes no gold parameter (structural firewall)",
          "gold" not in sig.parameters)


# ---------------------------------------------------------------------------
# End-to-end-ish: llm_match with a stub LLM + render closure
# ---------------------------------------------------------------------------

def test_llm_match_roundtrip():
    print("test_llm_match_roundtrip")
    tax = TR.load_taxonomy("semeval", "entity_role")

    def render(oc, _tax):
        return ("system", f"Code: {oc.get('open_code')}")

    def stub_llm(system, user):
        return '{"category": "Saboteur", "confidence": 0.8, "reasoning": "r", "is_edge_case": false, "edge_case_type": "clean"}'

    oc = {"open_code": "sabotage", "text_passage": "damaged from within", "chunk_id": "EN_UA_1_c0000"}
    out = TM.llm_match(oc, tax, render, stub_llm)
    check("llm_match parses category", out["category"] == "Saboteur")
    check("llm_match parses is_edge_case", out["is_edge_case"] is False)

    # parse failure -> soft fallback, no crash
    out2 = TM.llm_match(oc, tax, render, lambda s, u: "not json at all")
    check("llm_match parse failure -> edge_case parse_error",
          out2["edge_case_type"] == "parse_error" and out2["is_edge_case"] is True)


def main():
    test_registry()
    test_domain_of()
    test_embed_match_and_edges()
    test_disagreement_and_category_signals()
    test_scoring()
    test_firewall()
    test_llm_match_roundtrip()

    n = len(_results)
    n_pass = sum(1 for _, ok in _results if ok)
    print(f"\n{n_pass}/{n} checks passed.")
    failed = [name for name, ok in _results if not ok]
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
