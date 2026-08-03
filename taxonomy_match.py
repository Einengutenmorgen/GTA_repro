# taxonomy_match.py
"""
Match emergent open codes against a SEEDED taxonomy, then type the edge cases.

This is the core of the taxonomy-seeded coding experiments (see
GTA_taxonomy_seeded_experiment.md). It runs AFTER open coding and INSTEAD of
emergent axial coding: the axial layer is a fixed taxonomy (taxonomy_registry),
and the analytic task is "how well do the open codes map onto it, and where do
they refuse to fit?"

Two matchers, deliberately kept separate and then compared:

  embed_match : deterministic. Embeds each taxonomy leaf's (definition+anchors)
      once and each open code's (label+text_passage) once, via the SAME project
      embedding path (utils.embed_texts) the alignment and question-similarity
      layers use -- so every embedding shares one model + disk cache. Cosine
      score per (code, leaf); best leaf + score. The score is itself the
      edge-case detector (orphan = below threshold; straddler = top-2 within δ).

  llm_match   : interpretive. Hands the LLM one open code + the taxonomy schema
      and asks for {category, confidence, reasoning, is_edge_case,
      edge_case_type} as constrained JSON. Catches semantic fit embeddings miss
      and narrates WHY something is an edge case.

Where the two disagree is itself a strong edge-case signal (and a validity
cross-check: high cosine + LLM-rejects is exactly the Gilardi "agreement !=
validity" trap).

classify_edge_cases applies the typology rules deterministically (pure function,
unit-testable with fake vectors). score_against_gold is the ONLY place the gold
answer key is touched (eval-only).

Dependencies: numpy; utils.embed_texts (lazy sentence-transformers). The LLM
binding and the prompt are injected by the caller (main.py), so this module has
no hard dependency on llm_client / prompt_registry and stays unit-testable.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import DEFAULT_MODEL, DEFAULT_EMBED_CACHE, embed_texts
from taxonomy_registry import Taxonomy, TaxonomyLeaf


# ---------------------------------------------------------------------------
# Open-code normalization
# ---------------------------------------------------------------------------

def iter_valid_open_codes(open_codes: Sequence[dict]) -> List[dict]:
    """Filter run_open_coding output down to real codes (drop __status__ parse
    failures / empties), preserving order. Each kept dict has open_code,
    text_passage, chunk_id (chunk_id stamped by run_open_coding)."""
    out = []
    for oc in open_codes:
        if not isinstance(oc, dict):
            continue
        if "__status__" in oc:
            continue
        if not str(oc.get("open_code", "")).strip():
            continue
        out.append(oc)
    return out


def _code_query_text(oc: dict) -> str:
    """Embedding query text for one open code: label + its evidence span.
    Mirrors how embed_match indexes leaves (name+definition+anchors)."""
    label = str(oc.get("open_code", "")).strip()
    passage = str(oc.get("text_passage", "")).strip()
    return f"{label} \n {passage}".strip()


# ---------------------------------------------------------------------------
# Matcher 1 -- deterministic embedding match
# ---------------------------------------------------------------------------

def embed_match(
    open_codes: Sequence[dict],
    taxonomy: Taxonomy,
    top_k: int = 3,
    model_name: str = DEFAULT_MODEL,
    embed_cache_path: str = DEFAULT_EMBED_CACHE,
) -> List[dict]:
    """Cosine-match each open code to taxonomy leaves.

    Returns one dict per valid open code:
      {chunk_id, source_id, open_code, text_passage,
       ranked: [(category_id, name, score), ...]   # len<=top_k, desc by score
       emb_category, emb_category_id, emb_score,    # best (ranked[0]) flattened
       emb_margin}                                  # score gap ranked[0]-ranked[1]

    Embeddings are L2-normalized (utils.embed_texts), so cosine == dot product.
    Deterministic given a fixed model + cache -> reproducible edge-case counts.
    """
    codes = iter_valid_open_codes(open_codes)
    leaves = taxonomy.leaves
    if not codes or not leaves:
        return []

    leaf_texts = [lf.index_text() for lf in leaves]
    code_texts = [_code_query_text(oc) for oc in codes]

    leaf_vecs = embed_texts(leaf_texts, model_name, embed_cache_path)   # (L, d)
    code_vecs = embed_texts(code_texts, model_name, embed_cache_path)   # (C, d)

    # (C, L) cosine similarity
    sims = code_vecs @ leaf_vecs.T

    results: List[dict] = []
    for ci, oc in enumerate(codes):
        row = sims[ci]
        order = np.argsort(-row)[:top_k]
        ranked = [
            (leaves[j].category_id, leaves[j].name, float(row[j]))
            for j in order
        ]
        best = ranked[0]
        margin = float(ranked[0][2] - ranked[1][2]) if len(ranked) > 1 else float(ranked[0][2])
        results.append({
            "chunk_id": oc.get("chunk_id"),
            "source_id": _source_of(oc),
            "open_code": oc.get("open_code"),
            "text_passage": oc.get("text_passage"),
            "ranked": ranked,
            "emb_category": best[1],
            "emb_category_id": best[0],
            "emb_score": best[2],
            "emb_margin": margin,
        })
    return results


def _source_of(oc: dict) -> Optional[str]:
    """Best-effort article/source id for an open code. run_open_coding stamps
    chunk_id like '<source_id>_c0000'; recover source_id from it."""
    if oc.get("source_id"):
        return oc["source_id"]
    cid = oc.get("chunk_id") or ""
    # strip a trailing _c#### chunk suffix if present
    if "_c" in cid:
        return cid.rsplit("_c", 1)[0]
    return cid or None


# ---------------------------------------------------------------------------
# Matcher 2 -- interpretive LLM match
# ---------------------------------------------------------------------------

def llm_match(
    open_code: dict,
    taxonomy: Taxonomy,
    render_prompt: Callable[[dict, Taxonomy], Tuple[str, str]],
    llm: Callable[[str, str], str],
) -> dict:
    """Ask the LLM to assign ONE open code to a taxonomy leaf.

    render_prompt(open_code, taxonomy) -> (system_prompt, user_text). Injected
    by the caller (main.py builds it from prompt_registry) so this module never
    imports the prompt layer -- keeps it unit-testable and keeps the firewall
    check localized to the prompt builder.

    llm(system, user) -> raw string.

    Returns {category, category_id?, confidence, reasoning, is_edge_case,
    edge_case_type} -- a parse-failure yields a soft fallback with
    is_edge_case=True/edge_case_type='parse_error' rather than crashing.
    """
    system, user = render_prompt(open_code, taxonomy)
    raw = llm(system, user)
    parsed = _safe_json(raw)
    if parsed is None:
        return {
            "category": None, "category_id": None, "confidence": None,
            "reasoning": None, "is_edge_case": True,
            "edge_case_type": "parse_error", "__raw__": raw,
        }
    # normalize the fields we rely on downstream
    return {
        "category": parsed.get("category"),
        "category_id": parsed.get("category_id"),
        "confidence": parsed.get("confidence"),
        "reasoning": parsed.get("reasoning"),
        "is_edge_case": bool(parsed.get("is_edge_case", False)),
        "edge_case_type": parsed.get("edge_case_type"),
    }


def _safe_json(raw: str):
    if not raw or not raw.strip():
        return None
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Edge-case typing (pure, deterministic)
# ---------------------------------------------------------------------------

# Edge-case type constants (also the values written into the assignment table).
ORPHAN = "orphan"                 # best category below threshold / LLM says NONE
STRADDLER = "straddler"           # top-2 category scores within delta
DISAGREEMENT = "matcher_disagreement"  # embedding vs LLM assign different leaves
CLEAN = "clean"                   # fits one leaf comfortably; not an edge case


def classify_edge_cases(
    emb_results: Sequence[dict],
    llm_results: Optional[Sequence[dict]] = None,
    taxonomy: Optional[Taxonomy] = None,
    orphan_threshold: float = 0.30,
    straddle_delta: float = 0.05,
) -> List[dict]:
    """Type each open code's match and fold in category-level signals.

    Per-code edge types (a code can carry several, collected in edge_case_types):
      - orphan     : emb_score < orphan_threshold (or LLM category is NONE/null)
      - straddler  : emb_margin < straddle_delta (top-2 nearly tied)
      - matcher_disagreement : emb best leaf != LLM category (when llm_results given)
    Category-level signals are returned separately by category_signals().

    llm_results, if given, must be aligned 1:1 with emb_results (same order).
    taxonomy is only needed to map an LLM category NAME back to an id for the
    disagreement check; pass it when llm_results is given.

    Returns a NEW list of dicts (emb_result fields + llm_* fields + edge fields);
    inputs are not mutated. Pure and deterministic -> unit-testable with fake
    vectors and no model.
    """
    name_to_id = {}
    if taxonomy is not None:
        name_to_id = {lf.name.strip().lower(): lf.category_id for lf in taxonomy.leaves}

    out: List[dict] = []
    for i, emb in enumerate(emb_results):
        row = dict(emb)  # copy
        types: List[str] = []

        # --- orphan (embedding) ---
        if emb.get("emb_score", 0.0) < orphan_threshold:
            types.append(ORPHAN)

        # --- straddler ---
        if emb.get("emb_margin", 1.0) < straddle_delta:
            types.append(STRADDLER)

        # --- LLM fields + disagreement ---
        if llm_results is not None:
            lm = llm_results[i] if i < len(llm_results) else {}
            row["llm_category"] = lm.get("category")
            row["llm_confidence"] = lm.get("confidence")
            row["llm_is_edge_case"] = lm.get("is_edge_case")
            row["llm_edge_case_type"] = lm.get("edge_case_type")
            row["llm_reasoning"] = lm.get("reasoning")

            llm_cat = lm.get("category")
            if llm_cat is None or str(llm_cat).strip().upper() in ("NONE", "NULL", ""):
                if ORPHAN not in types:
                    types.append(ORPHAN)
            else:
                # map llm category name -> id, compare to embedding best id
                llm_id = lm.get("category_id") or name_to_id.get(str(llm_cat).strip().lower())
                if llm_id is not None and llm_id != emb.get("emb_category_id"):
                    types.append(DISAGREEMENT)
                row["llm_category_id"] = llm_id

            # let the LLM's own edge flag contribute a type if it named one
            if lm.get("is_edge_case") and lm.get("edge_case_type"):
                t = str(lm["edge_case_type"]).strip().lower().replace(" ", "_")
                if t and t not in types:
                    types.append(t)

        row["edge_case_types"] = types
        row["is_edge_case"] = bool(types)
        # single most-salient type for compact reporting (order = severity)
        row["edge_case_type"] = types[0] if types else CLEAN
        row["agree"] = (DISAGREEMENT not in types) if llm_results is not None else None
        out.append(row)
    return out


def category_signals(
    typed: Sequence[dict],
    taxonomy: Taxonomy,
    over_populated_factor: float = 3.0,
) -> dict:
    """Category-level edge signals over a full typed assignment set.

    Returns {
      per_category : {category_id: {name, n_codes, code_ids: [...]}},
      empty_categories : [category_id, ...],   # a leaf no open code maps to (Miss)
      over_populated   : [category_id, ...],   # >= factor * mean non-empty load (Over-split)
      n_codes, n_leaves
    }
    Uses the EMBEDDING best-category assignment as the mapping (deterministic).
    """
    per: Dict[str, dict] = {
        lf.category_id: {"name": lf.name, "n_codes": 0, "code_ids": []}
        for lf in taxonomy.leaves
    }
    for row in typed:
        cid = row.get("emb_category_id")
        if cid in per:
            per[cid]["n_codes"] += 1
            per[cid]["code_ids"].append(row.get("open_code"))

    counts = [v["n_codes"] for v in per.values()]
    non_empty = [c for c in counts if c > 0]
    empty = [cid for cid, v in per.items() if v["n_codes"] == 0]
    mean_load = (sum(non_empty) / len(non_empty)) if non_empty else 0.0
    over = [cid for cid, v in per.items()
            if mean_load > 0 and v["n_codes"] >= over_populated_factor * mean_load]

    return {
        "per_category": per,
        "empty_categories": empty,       # Miss list
        "over_populated": over,          # Over-split list
        "n_codes": len(typed),
        "n_leaves": len(taxonomy.leaves),
        "mean_load": mean_load,
    }


# ---------------------------------------------------------------------------
# Scoring against GOLD -- EVAL-ONLY. The gold arg comes from
# taxonomy_registry.load_gold and must NEVER have touched a prompt.
# ---------------------------------------------------------------------------

def score_against_gold(
    typed: Sequence[dict],
    gold: dict,
    seed: str,
    entity_resolution: Optional[Callable[[dict], Optional[str]]] = None,
) -> Optional[dict]:
    """Hard-score the seeded matching against the SemEval gold key.

    seed="narrative": E2. gold = {source_id: set(subnarrative_names)}. We
        aggregate the per-code EMBEDDING assignments up to the article level
        (the set of leaf names any of an article's codes mapped to) and compute
        multi-label precision / recall / F1 against the gold sub-narrative set,
        micro-averaged across articles, AND per domain (from the leaf domain).
        Returns None (with no crash) if `gold` is empty.

    seed="entity_role": E1. Hard `main_role` accuracy needs to know WHICH gold
        entity each open code's assignment refers to -- i.e. entity resolution,
        which the pipeline does not do yet. It is therefore GATED behind
        `entity_resolution`: a callable row->entity_mention. If not provided,
        return None and log a note (soft edge-typing still stands; this just
        skips the hard number). When provided, compute coarse main_role accuracy
        + a 3x3 confusion matrix over {protagonist, antagonist, innocent}.

    Returns a report dict or None. NEVER mutates inputs; NEVER emits gold into
    any prompt (it only reads gold to compute metrics).
    """
    sd = seed.strip().lower()
    if not gold:
        print("  -> score_against_gold: no gold provided; skipping hard scoring "
              "(soft edge-typing unaffected).")
        return None

    if sd == "narrative":
        return _score_narrative(typed, gold)

    if sd == "entity_role":
        if entity_resolution is None:
            print("  -> score_against_gold(entity_role): entity resolution not "
                  "provided; skipping coarse main_role accuracy (E1 hard scoring "
                  "is an optional, deferred add-on -- soft edge-typing stands).")
            return None
        return _score_entity_role(typed, gold, entity_resolution)

    raise ValueError(f"unknown seed {seed!r}")


def _score_narrative(typed: Sequence[dict], gold: Dict[str, set]) -> dict:
    # predicted labels per article = set of leaf NAMES its codes mapped to
    pred: Dict[str, set] = {}
    domain_of_article: Dict[str, Optional[str]] = {}
    for row in typed:
        sid = row.get("source_id")
        if sid is None:
            continue
        pred.setdefault(sid, set()).add(row.get("emb_category"))
        # infer article domain from the assigned leaf id prefix (URW/.. or CC/..)
        cid = row.get("emb_category_id") or ""
        dom = cid.split("/", 1)[0] if "/" in cid else None
        domain_of_article.setdefault(sid, dom)

    def prf(articles) -> dict:
        tp = fp = fn = 0
        for sid in articles:
            p = pred.get(sid, set())
            g = gold.get(sid, set())
            tp += len(p & g)
            fp += len(p - g)
            fn += len(g - p)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn, "n_articles": len(articles)}

    scored_articles = [s for s in pred if s in gold]
    overall = prf(scored_articles)
    by_domain = {}
    for dom in ("URW", "CC"):
        arts = [s for s in scored_articles if domain_of_article.get(s) == dom]
        if arts:
            by_domain[dom] = prf(arts)

    return {
        "seed": "narrative",
        "metric": "multi_label_prf_micro",
        "overall": overall,
        "by_domain": by_domain,
        "n_articles_scored": len(scored_articles),
        "n_articles_no_gold": len([s for s in pred if s not in gold]),
    }


_MAIN_ROLES = ("protagonist", "antagonist", "innocent")


def _coarse_of(category_id: Optional[str]) -> Optional[str]:
    """Map a fine entity-role leaf id ('antagonist/saboteur') to its main role."""
    if not category_id:
        return None
    head = category_id.split("/", 1)[0].strip().lower()
    return head if head in _MAIN_ROLES else None


def _score_entity_role(typed, gold, entity_resolution) -> dict:
    # Build a per-(source, entity) gold main_role lookup.
    gold_lut: Dict[Tuple[str, str], str] = {}
    for sid, rows in gold.items():
        for (entity, main_role, _fine) in rows:
            gold_lut[(sid, entity.strip().lower())] = main_role.strip().lower()

    confusion = {g: {p: 0 for p in _MAIN_ROLES} for g in _MAIN_ROLES}
    n_scored = n_skipped = 0
    for row in typed:
        entity = entity_resolution(row)
        sid = row.get("source_id")
        if entity is None or sid is None:
            n_skipped += 1
            continue
        gold_role = gold_lut.get((sid, entity.strip().lower()))
        pred_role = _coarse_of(row.get("emb_category_id"))
        if gold_role in _MAIN_ROLES and pred_role in _MAIN_ROLES:
            confusion[gold_role][pred_role] += 1
            n_scored += 1
        else:
            n_skipped += 1

    correct = sum(confusion[r][r] for r in _MAIN_ROLES)
    accuracy = correct / n_scored if n_scored else None
    return {
        "seed": "entity_role",
        "metric": "coarse_main_role_accuracy",
        "accuracy": accuracy,
        "confusion": confusion,     # confusion[gold][pred]
        "n_scored": n_scored,
        "n_skipped": n_skipped,
    }


# ---------------------------------------------------------------------------
# Assignment table (the pre-populated manual-typing sheet) as flat rows
# ---------------------------------------------------------------------------

def assignment_rows(typed: Sequence[dict]) -> List[dict]:
    """Flatten typed assignments into CSV-ready rows (the coding sheet). Keeps
    an empty 'coder2' column for later double-coding (study §5.4)."""
    rows = []
    for r in typed:
        rows.append({
            "chunk_id": r.get("chunk_id"),
            "source_id": r.get("source_id"),
            "open_code": r.get("open_code"),
            "text_passage": r.get("text_passage"),
            "emb_category": r.get("emb_category"),
            "emb_category_id": r.get("emb_category_id"),
            "emb_score": round(r.get("emb_score", 0.0), 4),
            "emb_margin": round(r.get("emb_margin", 0.0), 4),
            "llm_category": r.get("llm_category"),
            "llm_confidence": r.get("llm_confidence"),
            "agree": r.get("agree"),
            "edge_case_type": r.get("edge_case_type"),
            "edge_case_types": "|".join(r.get("edge_case_types", [])),
            "coder2": "",   # left empty for a second human coder
        })
    return rows
