# GTA Viewer

A local, zero-dependency inspector for GTA pipeline runs. It scans `data/` for
run output directories, works out which arm produced each one, and renders the
pipeline as a chronological spine you can click through.

Personal research tool: binds to `127.0.0.1`, read-only, no auth, no build step.

## Run it

```bash
python viewer/serve.py            # → http://127.0.0.1:8765
python viewer/serve.py --port 9000 --data data/RelationshipQuality
```

Requires nothing beyond the Python standard library. Hit **⟳ Rescan** in the UI
after a pipeline run finishes to pick up new output directories — no restart.

## What it shows

**Pipeline** — every stage in execution order down a vertical spine, with each
stage's headline counts and its output artifacts. A stage that *loops* becomes a
grid: rows are the steps within one pass, columns are successive passes, so the
same step across iterations lines up horizontally. Click any cell for its payload.

- Charmaz reflection loop → one column per theoretical-sampling iteration
- Straussian slot ladder → one column per category, rows are escalation rungs

**Codes & text** — pick a chunk, read the source text with every coded passage
highlighted inline, colored by the category the code was eventually aggregated
into. Codes whose passage cannot be found verbatim are reported separately
rather than silently dropped — a paraphrase rate you probably want to know.

**Lineage** — the aggregation step, audited four ways:

| view | question it answers |
|---|---|
| Categories | which codes back each category; which paradigm slots are empty |
| Orphans | codes that no category claims — dropped at aggregation, grouped by source |
| Overlaps | codes claimed by 2+ categories, plus *phantom references* — supporting labels that match no actual code |
| Seed → final flow | inferred merges, splits, renames and dissolutions between the seed and final category sets, from shared supporting codes |

**Change tree** (Charmaz) — absorbed fraction against the saturation threshold per
iteration, and every declared category operation (added / merged / split /
renamed) with its rationale.

**Slot ladder** (Straussian) — per-category rung-by-rung escalation, with the
before/after state of the condition / action / consequence slots.

**Taxonomy match** (seeded arm) — the assignment table with embedding score and
margin beside the LLM's pick, filterable by edge-case type (orphan, straddler,
matcher disagreement), plus taxonomy leaf load with empty (missed) and
over-populated (over-split) leaves flagged.

**SemEval corpus** — dev-set articles per language with gold entity spans
highlighted from their character offsets, colored by main role, alongside gold
narratives and subtask-3 explanations. Offsets are verified against the text and
drift is flagged rather than hidden.

## Notes

- Every view is linkable: the URL hash carries run, tab, language and article.
- Older run directories predate `chunk_index.json` and the current code schema.
  The viewer normalizes what it can and says plainly what it cannot show.
- `--data` can point anywhere; the server refuses paths outside that root.
