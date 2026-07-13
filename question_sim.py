# question_sim.py
"""
Question-to-question similarity index for the Straussian empty-slot escalation
ladder (step 2: retrieve the most similar Q&A blocks from OTHER participants
when a paradigm slot stays empty after re-passing the same participant).

Embed once (expensive), look up many (cheap) — same split as
alignment.SimilarityCache. Embedding goes through utils.embed_texts, so the
question vectors share the one project-wide model and disk cache.

Grain
-----
Built from qa_units (one interviewer question per unit), NOT from coding chunks,
so retrieval grain is independent of pairs_per_chunk. Only the QUESTION text is
embedded; answers and the INTERVIEWER:/PARTICIPANT: scaffold are excluded.

Retrieval: per-query top-k, merged
----------------------------------
A category with an empty slot is queried by MANY questions (the questions behind
its supporting open codes). Rather than averaging them into one centroid, each
query question is ranked independently off its own similarity row, and the
per-query top-k lists are merged (dedup keeps a retrieved question's HIGHEST
similarity across the queries that found it). This mirrors a human annotator
casting per-thread nets rather than an averaged blur, and it makes retrieval
auditable: every retrieved block traces to the specific query question that
pulled it in.

Widening the net across iterations
-----------------------------------
Results come back in a stable ranked order, so a loop can raise k each round
and take a set-difference on chunk_id to get only the NEWLY retrieved blocks —
never re-feeding the LLM what an earlier, smaller-k round already showed it.

Asymmetry
---------
The matrix is pure NxN and symmetric (participant-agnostic, reusable). The
step-2 use is asymmetric: retrieval excludes the querying participant's own
units (step 1 already consumed that participant's whole interview). The mask is
applied at QUERY time, never baked into the matrix.

Dependencies: numpy; utils.embed_texts (lazy sentence-transformers).
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from utils import DEFAULT_MODEL, DEFAULT_EMBED_CACHE, embed_texts


@dataclass
class QuestionSimCache:
    """Embed-once question index; retrieval is cheap lookups off `matrix` rows.

    matrix     : (N, N) cosine similarity, symmetric, unit-norm embeddings.
    chunk_ids  : row/col i -> originating qa-unit chunk_id.
    source_ids : row/col i -> participant id (for same-participant masking).
    questions  : row/col i -> exact embedded question text.
    units      : row/col i -> analysis unit (country), if tagged.
    """
    matrix: np.ndarray
    chunk_ids: List[str]
    source_ids: List[str]
    questions: List[str]
    units: List[Optional[str]]
    model_name: str

    @property
    def n(self) -> int:
        return len(self.chunk_ids)

    def _row(self, chunk_id: str) -> int:
        try:
            return self.chunk_ids.index(chunk_id)
        except ValueError:
            raise KeyError(f"chunk_id {chunk_id!r} not in question index")

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "QuestionSimCache":
        with open(path, "rb") as f:
            c = pickle.load(f)
        if not isinstance(c, QuestionSimCache):
            raise TypeError(f"{path} does not contain a QuestionSimCache")
        return c

    def summary(self) -> str:
        n_part = len(set(self.source_ids))
        return (f"QuestionSimCache: {self.n} questions from {n_part} participant(s), "
                f"model={self.model_name!r}")

    # -- retrieval (cheap, pure) --------------------------------------------

    def top_k_other_participant(
        self,
        query_chunk_ids: Sequence[str],
        k: int = 5,
        exclude_source_id: Optional[str] = None,
    ) -> List[dict]:
        """Per-query top-k, merged and capped, excluding the querying participant.

        For each query chunk_id, rank all questions by that query's own
        similarity row, take its top-k eligible hits, then merge across queries
        keeping each retrieved question's HIGHEST similarity (and recording
        which query pulled it in). Return the merged list, sorted by similarity
        desc, capped at k.

        query_chunk_ids   : qa-unit ids whose questions define the query
            (e.g. the questions behind a category's supporting open codes).
        k                 : per-query top-k DEPTH (not a global cap). Each query
            thread keeps its own top-k; the merged union is returned in full so
            no thread is starved. Raise it across iterations to widen the net.
        exclude_source_id : participant to mask out. Defaults to the query's
            source when all query units share one participant; pass explicitly
            for multi-participant queries.

        Returns [{chunk_id, source_id, question, similarity, matched_query}, ...].
        """
        if self.n == 0 or not query_chunk_ids:
            return []

        rows = [self._row(cid) for cid in query_chunk_ids]

        if exclude_source_id is None:
            srcs = {self.source_ids[r] for r in rows}
            exclude_source_id = next(iter(srcs)) if len(srcs) == 1 else None

        query_set = set(query_chunk_ids)
        # merged: chunk_id -> best hit dict seen so far
        merged: Dict[str, dict] = {}

        for qcid, r in zip(query_chunk_ids, rows):
            sim_row = self.matrix[r]                 # (N,) this query's similarities
            order = np.argsort(-sim_row)
            taken = 0
            for j in order:
                cid = self.chunk_ids[j]
                if cid in query_set:
                    continue
                if exclude_source_id is not None and self.source_ids[j] == exclude_source_id:
                    continue
                sim = float(sim_row[j])
                prev = merged.get(cid)
                if prev is None or sim > prev["similarity"]:
                    merged[cid] = {
                        "chunk_id": cid,
                        "source_id": self.source_ids[j],
                        "question": self.questions[j],
                        "similarity": sim,
                        "matched_query": qcid,
                    }
                taken += 1
                if taken >= k:
                    break

        # k is per-query DEPTH (each thread keeps its own top-k above); the
        # merged union is returned in full so no thread is starved by a global
        # cap. Output size is bounded by k * len(query_chunk_ids) before dedup.
        return sorted(merged.values(),
                      key=lambda d: (d["similarity"], d["chunk_id"]),
                      reverse=True)


def build_question_sim_cache(
    qa_units: Sequence[dict],
    model_name: str = DEFAULT_MODEL,
    embed_cache_path: str = DEFAULT_EMBED_CACHE,
) -> QuestionSimCache:
    """Embed all questions once, build the NxN cosine matrix.

    qa_units : dicts with at least {chunk_id, source_id, question} (unit
        optional). Feed iter_qa_units() output concatenated across all PDFs in
        the analysis unit.
    """
    # de-dupe on chunk_id (stable) while preserving order
    seen = set()
    rows = []
    for u in qa_units:
        cid = u["chunk_id"]
        if cid in seen:
            continue
        seen.add(cid)
        rows.append(u)

    chunk_ids = [u["chunk_id"] for u in rows]
    source_ids = [u["source_id"] for u in rows]
    questions = [str(u["question"]).strip() for u in rows]
    units = [u.get("unit") for u in rows]

    vecs = embed_texts(questions, model_name, embed_cache_path)  # (N, d), unit-norm
    if len(questions) == 0:
        matrix = np.zeros((0, 0), dtype=np.float32)
    else:
        matrix = (vecs @ vecs.T).astype(np.float32)

    return QuestionSimCache(
        matrix=matrix,
        chunk_ids=chunk_ids,
        source_ids=source_ids,
        questions=questions,
        units=units,
        model_name=model_name,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a question-to-question similarity index")
    ap.add_argument("qa_units_json", help="JSON array of qa-unit dicts (chunk_id, source_id, question)")
    ap.add_argument("--out", default="question_sim_cache.pkl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--embed-cache", default=DEFAULT_EMBED_CACHE)
    ap.add_argument("--probe", help="chunk_id to probe: print its top-5 cross-participant matches")
    args = ap.parse_args()

    units = json.load(open(args.qa_units_json, encoding="utf-8"))
    cache = build_question_sim_cache(units, model_name=args.model, embed_cache_path=args.embed_cache)
    print(cache.summary())
    cache.save(args.out)
    print(f"saved -> {args.out}")

    if args.probe:
        print(f"\nProbe {args.probe!r}:")
        print(f"  query question: {cache.questions[cache._row(args.probe)]!r}")
        for hit in cache.top_k_other_participant([args.probe], k=5):
            print(f"  {hit['similarity']:.3f}  [{hit['source_id']}]  {hit['question'][:70]!r}")
