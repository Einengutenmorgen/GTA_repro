# alignment.py
"""
Embedding-based codebook alignment: LLM-generated inductive codes vs. an
expert human codebook (RQ2 automatic-metrics layer).

Architecture
------------
Two strictly separated stages:

  1. compute_similarity(...)  -> SimilarityCache     [EXPENSIVE, runs once]
     Embeds both code sets, builds the full human x LLM cosine matrix.
     Embeddings are additionally cached on disk keyed by (model, text hash),
     so even a full recompute is cheap on rerun.

  2. align(cache, threshold)  -> AlignmentResult     [PURE, CHEAP]
     Takes the precomputed cache + a threshold and produces all metrics.
     Threshold tuning / sweeping NEVER re-embeds.

Metrics terminology
-------------------
All agreement figures produced here are a **kappa-analogue** built on
embedding cosine similarity with optimal one-to-one assignment. They are
NOT Cohen's kappa (different construct: no chance-agreement correction,
no categorical identity requirement) and must never be reported as such.

Two matchings, kept strictly separate
-------------------------------------
A. One-to-one (Hungarian)  -> headline coverage / precision.
B. Many-to-many (threshold graph) -> structural diagnostic only
   (fan-out = over-splitting signal, fan-in = merging/flattening signal).
   Many-to-many pairs never feed coverage/precision.

Fan-out / fan-in counts are far more threshold-sensitive than the
one-to-one match; they are therefore only ever reported with the
threshold that produced them attached (carried on AlignmentResult and
printed by every inspect/sweep helper).

Scope (v1): single unit, open level. `level` and `weight_field` are
threaded as seams for axial/selective alignment and salience-weighted
coverage (weight human codes by n_participants) but are NOT implemented.

Dependencies: sentence-transformers, scipy, numpy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBED_CACHE = ".embedding_cache.pkl"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SimilarityCache:
    """Serializable intermediate between embedding and matching.

    Holds everything align() needs; picklable via save()/load() so a user
    computes once and iterates on thresholds across sessions.
    """
    matrix: np.ndarray                 # (n_human, n_llm) cosine similarities
    human_codes: List[dict]            # ordered, deduped input dicts
    llm_codes: List[dict]              # ordered, deduped input dicts
    human_texts: List[str]             # exact embedded text per human code
    llm_texts: List[str]               # exact embedded text per LLM code
    model_name: str
    enrich_with_definition: bool
    definition_fallbacks: List[str]    # LLM code strings embedded WITHOUT a
                                       # definition despite enrichment being on
                                       # (data-quality signal, not silent)
    level: str = "open"
    unit: Optional[str] = None

    @property
    def n_human(self) -> int:
        return len(self.human_codes)

    @property
    def n_llm(self) -> int:
        return len(self.llm_codes)

    @property
    def n_definition_fallbacks(self) -> int:
        return len(self.definition_fallbacks)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "SimilarityCache":
        with open(path, "rb") as f:
            cache = pickle.load(f)
        if not isinstance(cache, SimilarityCache):
            raise TypeError(f"{path} does not contain a SimilarityCache")
        return cache

    def summary(self) -> str:
        lines = [
            f"SimilarityCache: {self.n_human} human x {self.n_llm} LLM codes "
            f"(unit={self.unit!r}, level={self.level!r})",
            f"  model={self.model_name!r}, enrich_with_definition={self.enrich_with_definition}",
        ]
        if self.enrich_with_definition:
            lines.append(
                f"  definition fallbacks (LLM codes embedded without a definition): "
                f"{self.n_definition_fallbacks}"
            )
            if self.definition_fallbacks:
                lines.append("    " + ", ".join(repr(c) for c in self.definition_fallbacks))
        return "\n".join(lines)


@dataclass
class AlignmentResult:
    """All metrics for one (cache, threshold) pair. Pure function output.

    NOTE: every number here is threshold-dependent; `threshold` travels with
    the result so fan-out/fan-in figures are never quoted without it.
    Coverage/precision are a kappa-ANALOGUE, not Cohen's kappa.
    """
    threshold: float
    model_name: str
    n_human: int
    n_llm: int

    # --- A. one-to-one (Hungarian) block: the headline metrics -------------
    matched: List[dict]                # {"human_code", "llm_code", "similarity"}
    unmatched_human: List[str]         # candidate Misses
    unmatched_llm: List[str]           # candidate hallucination / phantom
    coverage: Optional[float]          # matched / n_human (None if n_human == 0)
    precision: Optional[float]         # matched / n_llm   (None if n_llm == 0)

    # --- B. many-to-many (threshold graph) block: diagnostic only ----------
    # Does NOT feed coverage/precision.
    fanout_per_human_code: Dict[str, List[dict]]  # human code -> [{"llm_code", "similarity"}, ...]
    fanin_per_llm_code: Dict[str, List[dict]]     # llm code -> [{"human_code", "similarity"}, ...]
    max_fanout: int
    mean_fanout: float                 # mean over ALL human codes (zeros included)
    n_oversplit: int                   # human codes with fan-out >= 2
    max_fanin: int
    mean_fanin: float                  # mean over ALL LLM codes (zeros included)
    n_merged: int                      # LLM codes with fan-in >= 2

    # seams (threaded, not implemented)
    level: str = "open"
    weight_field: Optional[str] = None


# ---------------------------------------------------------------------------
# Stage 1: embedding (the only expensive call)
# ---------------------------------------------------------------------------

def _dedupe(codes: Sequence[dict], side: str) -> List[dict]:
    """Drop duplicate `code` strings within one set (keep first), warn."""
    seen: Dict[str, int] = {}
    out: List[dict] = []
    dropped: List[str] = []
    for item in codes:
        key = str(item.get("code", "")).strip()
        if key in seen:
            dropped.append(key)
            continue
        seen[key] = 1
        out.append(item)
    if dropped:
        warnings.warn(
            f"{side}: dropped {len(dropped)} duplicate code string(s) "
            f"(kept first occurrence): {sorted(set(dropped))}"
        )
    return out


def _build_texts(
    human_codes: List[dict],
    llm_codes: List[dict],
    enrich_with_definition: bool,
) -> Tuple[List[str], List[str], List[str]]:
    """Return (human_texts, llm_texts, llm_definition_fallbacks).

    Embedding text is a pure function of the inputs + the enrichment flag,
    so it is constant across all runs off a given cache.
    """
    def _join(code: str, definition: Optional[str]) -> str:
        code = str(code).strip()
        if enrich_with_definition and definition:
            return f"{code}: {str(definition).strip()}"
        return code

    human_texts = [_join(h["code"], h.get("definition")) for h in human_codes]

    llm_texts: List[str] = []
    fallbacks: List[str] = []
    for l in llm_codes:
        definition = l.get("definition")
        if enrich_with_definition and not definition:
            # Fall back to the bare code; record it as a data-quality signal.
            fallbacks.append(str(l["code"]).strip())
        llm_texts.append(_join(l["code"], definition))
    return human_texts, llm_texts, fallbacks


def _load_disk_cache(path: str) -> Dict[Tuple[str, str], np.ndarray]:
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:  # corrupt cache is not fatal, just recompute
            warnings.warn(f"Could not read embedding cache {path!r} ({e}); recomputing.")
    return {}


def _save_disk_cache(path: str, cache: Dict[Tuple[str, str], np.ndarray]) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    os.replace(tmp, path)


def _text_key(model_name: str, text: str) -> Tuple[str, str]:
    return (model_name, hashlib.sha256(text.encode("utf-8")).hexdigest())


def _embed_texts(
    texts: List[str],
    model_name: str,
    disk_cache_path: str,
) -> np.ndarray:
    """Embed texts, L2-normalized, with a persistent per-text disk cache."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    disk = _load_disk_cache(disk_cache_path)
    missing_idx = [i for i, t in enumerate(texts)
                   if _text_key(model_name, t) not in disk]

    if missing_idx:
        # Import lazily so cache-only reruns never touch the model.
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        new_vecs = model.encode(
            [texts[i] for i in missing_idx],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        for i, vec in zip(missing_idx, new_vecs):
            disk[_text_key(model_name, texts[i])] = vec
        _save_disk_cache(disk_cache_path, disk)

    vecs = np.stack([disk[_text_key(model_name, t)] for t in texts]).astype(np.float32)
    # Re-normalize defensively (cached vectors are already normalized).
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def compute_similarity(
    human_codes: Sequence[dict],
    llm_codes: Sequence[dict],
    model_name: str = DEFAULT_MODEL,
    enrich_with_definition: bool = True,
    embed_cache_path: str = DEFAULT_EMBED_CACHE,
    level: str = "open",
) -> SimilarityCache:
    """Embed both code sets once and build the full cosine matrix.

    The ONLY expensive call in this module. Everything downstream
    (align, sweep, inspect_*) operates on the returned cache.

    Parameters
    ----------
    human_codes, llm_codes : lists of dicts in the common schema,
        pre-filtered to one unit and one level. Matching is on `code` text.
    enrich_with_definition : append `definition` to the embedded text.
        If an LLM code lacks a definition, its bare `code` is embedded
        instead and the fallback is recorded on the cache (count + codes).
    level : provenance only in v1; axial/selective are a later seam.
    """
    human = _dedupe(list(human_codes), "human_codes")
    llm = _dedupe(list(llm_codes), "llm_codes")

    human_texts, llm_texts, fallbacks = _build_texts(human, llm, enrich_with_definition)

    h_vecs = _embed_texts(human_texts, model_name, embed_cache_path)
    l_vecs = _embed_texts(llm_texts, model_name, embed_cache_path)

    if len(human) == 0 or len(llm) == 0:
        matrix = np.zeros((len(human), len(llm)), dtype=np.float32)
    else:
        matrix = h_vecs @ l_vecs.T  # cosine similarity (vectors are unit-norm)

    unit = None
    for item in list(human) + list(llm):
        if item.get("unit"):
            unit = item["unit"]
            break

    return SimilarityCache(
        matrix=matrix,
        human_codes=human,
        llm_codes=llm,
        human_texts=human_texts,
        llm_texts=llm_texts,
        model_name=model_name,
        enrich_with_definition=enrich_with_definition,
        definition_fallbacks=fallbacks,
        level=level,
        unit=unit,
    )


# ---------------------------------------------------------------------------
# Stage 2: matching (pure, cheap, no embedding, no I/O)
# ---------------------------------------------------------------------------

def align(
    cache: SimilarityCache,
    threshold: float,
    weight_field: Optional[str] = None,
) -> AlignmentResult:
    """Produce all metrics for one threshold off a precomputed cache.

    Pure function: (cache, threshold) -> AlignmentResult. No embedding,
    no file I/O. Fast enough to call in a tight sweep loop.

    weight_field is a SEAM for salience-weighted coverage (e.g. weight
    human codes by `n_participants`); it is threaded but not implemented.
    """
    if weight_field is not None:
        raise NotImplementedError(
            "salience-weighted coverage is a v2 seam; weight_field must be None"
        )

    n_h, n_l = cache.n_human, cache.n_llm
    h_names = [str(h["code"]).strip() for h in cache.human_codes]
    l_names = [str(l["code"]).strip() for l in cache.llm_codes]
    M = cache.matrix

    # ---- A. one-to-one (Hungarian) ----------------------------------------
    matched: List[dict] = []
    matched_h: set = set()
    matched_l: set = set()

    if n_h > 0 and n_l > 0:
        rows, cols = linear_sum_assignment(-M)  # maximize similarity; handles rectangular
        for r, c in zip(rows, cols):
            sim = float(M[r, c])
            if sim >= threshold:
                matched.append({
                    "human_code": h_names[r],
                    "llm_code": l_names[c],
                    "similarity": sim,
                })
                matched_h.add(r)
                matched_l.add(c)
        # Pairs below threshold split back into both unmatched pools implicitly.

    matched.sort(key=lambda p: p["similarity"], reverse=True)
    unmatched_human = [h_names[i] for i in range(n_h) if i not in matched_h]
    unmatched_llm = [l_names[j] for j in range(n_l) if j not in matched_l]

    coverage = (len(matched) / n_h) if n_h > 0 else None
    precision = (len(matched) / n_l) if n_l > 0 else None

    # ---- B. many-to-many (threshold graph, diagnostic only) ---------------
    fanout: Dict[str, List[dict]] = {}
    fanin: Dict[str, List[dict]] = {}
    fanout_counts = np.zeros(n_h, dtype=int)
    fanin_counts = np.zeros(n_l, dtype=int)

    if n_h > 0 and n_l > 0:
        above = M >= threshold
        fanout_counts = above.sum(axis=1)
        fanin_counts = above.sum(axis=0)
        for i in range(n_h):
            hits = np.flatnonzero(above[i])
            if hits.size:
                fanout[h_names[i]] = sorted(
                    ({"llm_code": l_names[j], "similarity": float(M[i, j])} for j in hits),
                    key=lambda d: d["similarity"], reverse=True,
                )
        for j in range(n_l):
            hits = np.flatnonzero(above[:, j])
            if hits.size:
                fanin[l_names[j]] = sorted(
                    ({"human_code": h_names[i], "similarity": float(M[i, j])} for i in hits),
                    key=lambda d: d["similarity"], reverse=True,
                )

    return AlignmentResult(
        threshold=float(threshold),
        model_name=cache.model_name,
        n_human=n_h,
        n_llm=n_l,
        matched=matched,
        unmatched_human=unmatched_human,
        unmatched_llm=unmatched_llm,
        coverage=coverage,
        precision=precision,
        fanout_per_human_code=fanout,
        fanin_per_llm_code=fanin,
        max_fanout=int(fanout_counts.max()) if n_h > 0 else 0,
        mean_fanout=float(fanout_counts.mean()) if n_h > 0 else 0.0,
        n_oversplit=int((fanout_counts >= 2).sum()),
        max_fanin=int(fanin_counts.max()) if n_l > 0 else 0,
        mean_fanin=float(fanin_counts.mean()) if n_l > 0 else 0.0,
        n_merged=int((fanin_counts >= 2).sum()),
        level=cache.level,
        weight_field=weight_field,
    )


# ---------------------------------------------------------------------------
# Sweep + inspection helpers (the primary UX)
# ---------------------------------------------------------------------------

def sweep(cache: SimilarityCache, thresholds: Sequence[float]) -> List[dict]:
    """One align() per threshold; returns one row per threshold.

    Fan-out/fan-in columns carry the row's threshold with them by
    construction — never quote them without it.
    """
    rows = []
    for t in thresholds:
        r = align(cache, t)
        rows.append({
            "threshold": round(float(t), 4),
            "coverage": None if r.coverage is None else round(r.coverage, 4),
            "precision": None if r.precision is None else round(r.precision, 4),
            "n_matched": len(r.matched),
            "mean_fanout": round(r.mean_fanout, 3),
            "max_fanout": r.max_fanout,
            "n_oversplit": r.n_oversplit,
            "mean_fanin": round(r.mean_fanin, 3),
            "max_fanin": r.max_fanin,
            "n_merged": r.n_merged,
        })
    return rows


def format_sweep_table(rows: List[dict]) -> str:
    """Plain-text table for terminal inspection."""
    if not rows:
        return "(empty sweep)"
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.rjust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join(
        "  ".join(str(r[c]).rjust(widths[c]) for c in cols) for r in rows
    )
    return "\n".join([header, sep, body])


def inspect_matches(cache: SimilarityCache, threshold: float, sort: str = "asc") -> List[dict]:
    """All one-to-one matched pairs at `threshold`, sorted by similarity.

    Default ascending: the low end is where bad matches hide and where
    the threshold gets set by eyeball.
    """
    r = align(cache, threshold)
    pairs = sorted(r.matched, key=lambda p: p["similarity"], reverse=(sort == "desc"))
    print(f"[one-to-one matches @ threshold={threshold} | model={cache.model_name}] "
          f"n={len(pairs)}, coverage={r.coverage}, precision={r.precision}")
    for p in pairs:
        print(f"  {p['similarity']:.3f}  {p['human_code']!r}  <->  {p['llm_code']!r}")
    return pairs


def inspect_fanout(cache: SimilarityCache, threshold: float, min_fanout: int = 2) -> Dict[str, List[dict]]:
    """Human codes with >= min_fanout above-threshold LLM codes.

    High fan-out reads directly as an over-splitting signal — always
    interpreted at the printed threshold, never in isolation.
    """
    r = align(cache, threshold)
    hits = {h: lst for h, lst in r.fanout_per_human_code.items() if len(lst) >= min_fanout}
    print(f"[fan-out >= {min_fanout} @ threshold={threshold} | model={cache.model_name}] "
          f"{len(hits)} human code(s)")
    for h, lst in sorted(hits.items(), key=lambda kv: len(kv[1]), reverse=True):
        print(f"  {h!r}  (fan-out {len(lst)})")
        for d in lst:
            print(f"      {d['similarity']:.3f}  {d['llm_code']!r}")
    return hits


# ---------------------------------------------------------------------------
# Thin CLI: load two JSON inputs, compute cache once, print a sweep, save.
# ---------------------------------------------------------------------------

def _load_codes(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of code dicts")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Embedding-based codebook alignment (kappa-analogue; NOT Cohen's kappa)."
    )
    ap.add_argument("human_json", help="human codebook codes (JSON array, common schema)")
    ap.add_argument("llm_json", help="LLM codes (JSON array, common schema)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-enrich", action="store_true",
                    help="embed bare code text without appending definitions")
    ap.add_argument("--cache-out", default="similarity_cache.pkl",
                    help="where to save the SimilarityCache pickle")
    ap.add_argument("--embed-cache", default=DEFAULT_EMBED_CACHE,
                    help="per-text embedding disk cache path")
    ap.add_argument("--t-min", type=float, default=0.30)
    ap.add_argument("--t-max", type=float, default=0.80)
    ap.add_argument("--t-step", type=float, default=0.05)
    args = ap.parse_args()

    human = _load_codes(args.human_json)
    llm = _load_codes(args.llm_json)

    cache = compute_similarity(
        human, llm,
        model_name=args.model,
        enrich_with_definition=not args.no_enrich,
        embed_cache_path=args.embed_cache,
    )
    print(cache.summary())
    print()

    thresholds = np.round(np.arange(args.t_min, args.t_max + 1e-9, args.t_step), 4)
    print(format_sweep_table(sweep(cache, thresholds)))
    print("\nNote: all figures are an embedding-based kappa-analogue, not Cohen's kappa.")
    print("Fan-out/fan-in are strongly threshold-dependent; quote them only with their threshold.")

    cache.save(args.cache_out)
    print(f"\nSimilarityCache saved -> {args.cache_out}")
    print("Re-tune later without re-embedding:")
    print(f"  from alignment import SimilarityCache, align, sweep")
    print(f"  cache = SimilarityCache.load({args.cache_out!r})")
    print(f"  align(cache, 0.55)")


if __name__ == "__main__":
    main()
