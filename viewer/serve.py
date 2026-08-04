#!/usr/bin/env python3
"""
GTA Viewer -- a local, zero-dependency inspector for GTA pipeline runs.

    python viewer/serve.py            # http://127.0.0.1:8765
    python viewer/serve.py --port 9000 --data data

It scans the data/ tree for run output directories, auto-detects which arm
produced each one (straussian / charmaz / taxonomy-seeded), builds a
chronological step graph for it, and serves everything to a static SPA.

Personal-use tool: binds to localhost, no auth, no write access.
Stdlib only -- no pip install needed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# --------------------------------------------------------------------------
# Artifact catalogue -- filename -> human meaning. Drives both detection and
# the artifact chips shown on each step card.
# --------------------------------------------------------------------------

ARTIFACT_META = {
    "chunk_index.json":                    ("Chunk index", "The full backing store: every chunk with its source, question and answer."),
    "question_sim_cache.pkl":              ("Question-similarity cache", "Embedding cache backing the escalation ladder's retrieval rungs."),
    "output_open_codes.json":              ("Open codes", "Emergent codes, one row per code, stamped with its originating chunk."),
    "output_initial_codes.json":           ("Initial codes", "Charmaz initial (line-by-line) codes with in-vivo flags."),
    "output_axial_codes.json":             ("Axial categories", "Categories with condition / action / consequence slots."),
    "output_axial_codes_resolved.json":    ("Axial categories (resolved)", "After the empty-slot escalation ladder ran."),
    "output_axial_codes_seeded.json":      ("Seeded axial layer", "Fixed taxonomy injected in place of emergent axial coding."),
    "output_slot_traces.json":             ("Slot escalation traces", "Per-category rung-by-rung log of how empty slots were filled."),
    "output_focused_codes.json":           ("Focused categories (seed pass)", "First-pass Charmaz focused coding."),
    "output_focused_codes_final.json":     ("Focused categories (final)", "After the reflection loop converged or exhausted the corpus."),
    "output_initial_memos.json":           ("Initial memos", "Charmaz step 4 memos over the initial codes."),
    "output_advanced_memos_seed.json":     ("Advanced memos (seed)", "Category-level memos naming properties and thin areas."),
    "output_advanced_memos_final.json":    ("Advanced memos (final)", "Advanced memos after the reflection loop."),
    "output_charmaz_change_tree.json":     ("Change tree", "Every reflection-loop iteration: what was absorbed, added, merged, split."),
    "output_taxonomy_assignments.json":    ("Taxonomy assignments", "Every open code matched to a taxonomy leaf by embedding + LLM."),
    "output_taxonomy_assignments.csv":     ("Taxonomy assignments (CSV)", "Same table, ready for manual double-coding."),
    "output_taxonomy_category_signals.json": ("Category signals", "Empty (missed) and over-populated (over-split) taxonomy leaves."),
    "output_taxonomy_gold_score.json":     ("Gold score", "Eval-only scoring against the SemEval gold key."),
    "output_final_theory.md":              ("Final theory", "The integrated theoretical account."),
    "output_axial_codes.md":               ("Axial categories (markdown)", "Legacy markdown axial output."),
}

CHANGE_TYPE_ORDER = ["added", "merged", "split", "renamed", "broadened", "narrowed", "dropped"]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _read_text(path, limit=None):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit) if limit else fh.read()
    except Exception:
        return None


def _valid_codes(codes):
    """Open/initial codes minus the failure envelopes the pipeline writes."""
    if not isinstance(codes, list):
        return []
    return [c for c in codes if isinstance(c, dict) and "__status__" not in c]


def _failed_codes(codes):
    if not isinstance(codes, list):
        return []
    return [c for c in codes if isinstance(c, dict) and c.get("__status__") == "failed"]


# --------------------------------------------------------------------------
# Run discovery
# --------------------------------------------------------------------------

def discover_runs(data_root):
    """Every directory under data_root holding at least one pipeline artifact."""
    runs = []
    for dirpath, dirnames, filenames in os.walk(data_root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        hits = [f for f in filenames if f in ARTIFACT_META]
        if not hits:
            continue
        # a directory that only carries a chunk index isn't a run
        if set(hits) <= {"chunk_index.json", "question_sim_cache.pkl"}:
            continue
        rel = os.path.relpath(dirpath, data_root).replace(os.sep, "/")
        runs.append(_run_summary(data_root, rel, sorted(hits)))
    runs.sort(key=lambda r: (r["sort_key"], r["id"]), reverse=True)
    return runs


def _detect_arm(files):
    f = set(files)
    if {"output_taxonomy_assignments.json"} & f or "output_axial_codes_seeded.json" in f:
        return "seeded"
    if {"output_charmaz_change_tree.json", "output_initial_codes.json",
        "output_focused_codes.json"} & f:
        return "charmaz"
    return "straussian"


TIMECODE_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")


def _run_summary(data_root, rel, files):
    path = os.path.join(data_root, rel)
    name = os.path.basename(rel)
    arm = _detect_arm(files)

    m = TIMECODE_RE.search(name)
    if m:
        stamp = "%s-%s-%s %s:%s:%s" % m.groups()
        sort_key = "".join(m.groups())
    else:
        stamp = None
        try:
            sort_key = str(int(os.path.getmtime(path)))
        except OSError:
            sort_key = "0"

    lower = name.lower()
    dataset = "semeval" if "semeval" in lower else "silan"
    model = "proprietary" if "proprietary" in lower else ("local" if "local" in lower else None)
    version = None
    vm = re.match(r"[vV]\.?(\d+(?:\.\d+)?)", name)
    if vm:
        version = "v" + vm.group(1)

    # cheap headline stats -- read only what is small or countable
    stats = {}
    for fn, key in (("output_open_codes.json", "codes"),
                    ("output_initial_codes.json", "codes")):
        if fn in files:
            data = _load_json(os.path.join(path, fn))
            stats[key] = len(_valid_codes(data))
            failed = len(_failed_codes(data))
            if failed:
                stats["failed_chunks"] = failed
    for fn, key in (("output_axial_codes_resolved.json", "categories"),
                    ("output_axial_codes.json", "categories"),
                    ("output_focused_codes_final.json", "categories"),
                    ("output_focused_codes.json", "categories"),
                    ("output_axial_codes_seeded.json", "categories")):
        if fn in files and key not in stats:
            data = _load_json(os.path.join(path, fn))
            if isinstance(data, list):
                stats[key] = len(data)
    if "chunk_index.json" in files:
        idx = _load_json(os.path.join(path, "chunk_index.json"))
        if isinstance(idx, dict):
            stats["chunks"] = len(idx)
            stats["sources"] = len({v.get("source_id") for v in idx.values()
                                    if isinstance(v, dict)})
    if "output_charmaz_change_tree.json" in files:
        tree = _load_json(os.path.join(path, "output_charmaz_change_tree.json")) or {}
        stats["iterations"] = tree.get("n_iterations")
        stats["saturated"] = tree.get("saturation_reached")
    if "output_taxonomy_assignments.json" in files:
        rows = _load_json(os.path.join(path, "output_taxonomy_assignments.json")) or []
        stats["assignments"] = len(rows)
        stats["edge_cases"] = sum(1 for r in rows
                                  if r.get("edge_case_type") not in (None, "clean"))

    return {
        "id": rel,
        "name": name,
        "collection": rel.split("/")[0],
        "arm": arm,
        "dataset": dataset,
        "model": model,
        "version": version,
        "timestamp": stamp,
        "sort_key": sort_key,
        "files": files,
        "stats": stats,
    }


# --------------------------------------------------------------------------
# Step-graph construction
#
# A run is rendered as a vertical chronological spine. A stage that loops
# carries `loop.columns` -- one column per iteration -- so the same step of
# successive iterations lines up horizontally.
# --------------------------------------------------------------------------

def _stage(id_, title, subtitle, phase, **kw):
    st = {
        "id": id_, "title": title, "subtitle": subtitle, "phase": phase,
        "kind": "step", "status": "ok", "stats": {}, "artifacts": [],
        "detail": None, "note": None,
    }
    st.update(kw)
    return st


def _artifact_chip(path, fn):
    label, desc = ARTIFACT_META.get(fn, (fn, ""))
    full = os.path.join(path, fn)
    return {
        "file": fn, "label": label, "desc": desc,
        "size": os.path.getsize(full) if os.path.exists(full) else 0,
        "exists": os.path.exists(full),
    }


def build_manifest(data_root, rel):
    path = os.path.join(data_root, rel)
    if not os.path.isdir(path):
        return None
    files = sorted(f for f in os.listdir(path) if f in ARTIFACT_META)
    summary = _run_summary(data_root, rel, files)
    arm = summary["arm"]

    chip = lambda fn: _artifact_chip(path, fn)
    have = set(files)
    stages = []
    phase = 0

    # ---- Phase 0: chunking (shared by every arm) --------------------------
    if "chunk_index.json" in have:
        idx = _load_json(os.path.join(path, "chunk_index.json")) or {}
        sources = {}
        for cid, c in idx.items():
            if not isinstance(c, dict):
                continue
            sources.setdefault(c.get("source_id") or "?", 0)
            sources[c.get("source_id") or "?"] += 1
        stages.append(_stage(
            "chunking", "Chunking", "Phase 0 · corpus → chunks", phase,
            stats={"chunks": len(idx), "sources": len(sources)},
            artifacts=[chip("chunk_index.json")],
            detail="chunks",
        ))
    else:
        stages.append(_stage(
            "chunking", "Chunking", "Phase 0 · corpus → chunks", phase,
            status="missing", note="No chunk_index.json in this run — "
                                   "predates per-run chunk persistence.",
        ))
    phase += 1

    if "question_sim_cache.pkl" in have:
        stages.append(_stage(
            "qsim", "Question-similarity index", "Phase 0b · retrieval backing store", phase,
            artifacts=[chip("question_sim_cache.pkl")],
            note="Binary cache — size only, not inspectable here.",
        ))
        phase += 1

    # ---- Phase 1: open / initial coding -----------------------------------
    code_file = ("output_initial_codes.json" if "output_initial_codes.json" in have
                 else "output_open_codes.json" if "output_open_codes.json" in have else None)
    if code_file:
        codes = _load_json(os.path.join(path, code_file)) or []
        valid, failed = _valid_codes(codes), _failed_codes(codes)
        in_vivo = sum(1 for c in valid if c.get("in_vivo"))
        st = {"codes": len(valid), "chunks covered": len({c.get("chunk_id") for c in valid})}
        if in_vivo:
            st["in-vivo"] = in_vivo
        if failed:
            st["failed chunks"] = len(failed)
        stages.append(_stage(
            "coding", "Initial coding" if arm == "charmaz" else "Open coding",
            "Phase 1 · chunks → codes", phase,
            stats=st, artifacts=[chip(code_file)], detail="codes",
            status="warn" if failed else "ok",
            note=("%d chunk(s) produced no parseable codes." % len(failed)) if failed else None,
        ))
        phase += 1

    # ---- Arm-specific middle ---------------------------------------------
    if arm == "charmaz":
        phase = _charmaz_stages(path, have, stages, phase, chip)
    elif arm == "seeded":
        phase = _seeded_stages(path, have, stages, phase, chip)
    else:
        phase = _straussian_stages(path, have, stages, phase, chip)

    # ---- Terminal: theory -------------------------------------------------
    if "output_final_theory.md" in have:
        txt = _read_text(os.path.join(path, "output_final_theory.md")) or ""
        stages.append(_stage(
            "theory",
            "Memo sorting → theoretical account" if arm == "charmaz" else "Selective coding",
            "Final phase · categories → theory", phase,
            stats={"words": len(txt.split())},
            artifacts=[chip("output_final_theory.md")], detail="theory",
        ))
        phase += 1

    summary["stages"] = stages
    summary["artifacts"] = [chip(f) for f in files]
    return summary


def _straussian_stages(path, have, stages, phase, chip):
    if "output_axial_codes.json" in have:
        ax = _load_json(os.path.join(path, "output_axial_codes.json")) or []
        empty = 0
        supp = 0
        if isinstance(ax, list):
            for c in ax:
                if not isinstance(c, dict):
                    continue
                supp += len(c.get("supporting_open_codes") or [])
                empty += sum(1 for s in ("condition", "action_interaction", "consequence")
                             if not (c.get(s) or "").strip())
        stages.append(_stage(
            "axial", "Axial coding", "Phase 2 · codes → categories", phase,
            stats={"categories": len(ax) if isinstance(ax, list) else 0,
                   "supporting links": supp, "empty slots": empty},
            artifacts=[chip("output_axial_codes.json")], detail="axial",
        ))
        phase += 1

    if "output_slot_traces.json" in have:
        traces = _load_json(os.path.join(path, "output_slot_traces.json")) or []
        # Loop columns = categories that actually escalated; rows = rungs.
        cols = []
        for t in traces if isinstance(traces, list) else []:
            trace = t.get("trace") or []
            rungs = [r for r in trace if r.get("rung") not in (None, "none")]
            cols.append({
                "label": t.get("axial_category", "?"),
                "index": t.get("index"),
                "escalated": bool(rungs),
                "substeps": [{
                    "id": "rung-%s-%s" % (t.get("index"), i),
                    "title": r.get("rung", "?"),
                    "note": r.get("note"),
                    "payload": r,
                } for i, r in enumerate(trace)],
            })
        n_esc = sum(1 for c in cols if c["escalated"])
        arts = [chip("output_slot_traces.json")]
        if "output_axial_codes_resolved.json" in have:
            arts.append(chip("output_axial_codes_resolved.json"))
        stages.append(_stage(
            "ladder", "Empty-slot escalation ladder",
            "Phase 2b · loop · one column per category", phase,
            kind="loop",
            stats={"categories": len(cols), "escalated": n_esc},
            artifacts=arts, detail="ladder",
            loop={"axis_label": "category", "columns": cols},
            note=None if n_esc else "No category had an empty slot — the ladder short-circuited.",
        ))
        phase += 1
    elif "output_axial_codes_resolved.json" in have:
        stages.append(_stage(
            "ladder", "Empty-slot escalation ladder", "Phase 2b", phase,
            artifacts=[chip("output_axial_codes_resolved.json")], detail="axial",
        ))
        phase += 1
    return phase


def _charmaz_stages(path, have, stages, phase, chip):
    if "output_initial_memos.json" in have:
        memos = (_load_json(os.path.join(path, "output_initial_memos.json")) or {}).get("memos", [])
        stages.append(_stage(
            "initial_memos", "Initial memo-writing", "Step 4 · codes → memos", phase,
            stats={"memos": len(memos)},
            artifacts=[chip("output_initial_memos.json")], detail="memos",
        ))
        phase += 1

    if "output_focused_codes.json" in have:
        foc = _load_json(os.path.join(path, "output_focused_codes.json")) or []
        supp = sum(len(c.get("supporting_open_codes") or []) for c in foc if isinstance(c, dict))
        stages.append(_stage(
            "focused", "Focused coding (seed pass)", "Step 5 · codes → categories", phase,
            stats={"categories": len(foc), "supporting links": supp},
            artifacts=[chip("output_focused_codes.json")], detail="axial",
        ))
        phase += 1

    if "output_advanced_memos_seed.json" in have:
        m = (_load_json(os.path.join(path, "output_advanced_memos_seed.json")) or {}).get("memos", [])
        thin = sum(len(x.get("thin_areas") or []) for x in m if isinstance(x, dict))
        stages.append(_stage(
            "adv_memos_seed", "Advanced memo-writing (seed)", "Step 6 · categories → memos", phase,
            stats={"memos": len(m), "thin areas named": thin},
            artifacts=[chip("output_advanced_memos_seed.json")], detail="memos",
        ))
        phase += 1

    if "output_charmaz_change_tree.json" in have:
        tree = _load_json(os.path.join(path, "output_charmaz_change_tree.json")) or {}
        cols = []
        for it in tree.get("change_tree", []) or []:
            changes = it.get("changes_made") or []
            by_type = {}
            for ch in changes:
                by_type[ch.get("type", "?")] = by_type.get(ch.get("type", "?"), 0) + 1
            cols.append({
                "label": "Iteration %s" % it.get("iteration_n"),
                "index": it.get("iteration_n"),
                "decision": it.get("decision"),
                "absorbed_fraction": it.get("absorbed_fraction"),
                "threshold": it.get("threshold"),
                "change_counts": by_type,
                "substeps": [
                    {"id": "slice-%s" % it.get("iteration_n"), "title": "Theoretical sample",
                     "stat": "%d source(s), %d unit(s)" % (len(it.get("slice_sources") or []),
                                                           it.get("n_units") or 0),
                     "payload": {"slice_sources": it.get("slice_sources"),
                                 "n_units": it.get("n_units")}},
                    {"id": "applic-%s" % it.get("iteration_n"), "title": "Applicability check",
                     "stat": "%.0f%% absorbed (thr %.0f%%)" % (100 * (it.get("absorbed_fraction") or 0),
                                                               100 * (it.get("threshold") or 0)),
                     "status": "ok" if it.get("decision") == "saturated" else "warn",
                     "payload": {"n_fit": it.get("n_fit"),
                                 "n_unabsorbed_units": it.get("n_unabsorbed_units"),
                                 "decision": it.get("decision"),
                                 "applicability_raw": it.get("applicability_raw")}},
                    {"id": "newcodes-%s" % it.get("iteration_n"), "title": "New initial codes",
                     "stat": "%d code(s)" % len(it.get("new_initial_codes") or []),
                     "payload": {"new_initial_codes": it.get("new_initial_codes")}},
                    {"id": "changes-%s" % it.get("iteration_n"), "title": "Category changes",
                     "stat": ", ".join("%d %s" % (v, k) for k, v in by_type.items()) or "no change",
                     "payload": {"changes_made": changes}},
                    {"id": "recat-%s" % it.get("iteration_n"), "title": "Resulting categories",
                     "stat": "%d categor(ies)" % len(it.get("resulting_categories") or []),
                     "payload": {"resulting_categories": it.get("resulting_categories")}},
                    {"id": "advmemo-%s" % it.get("iteration_n"), "title": "Advanced memo",
                     "stat": "%d memo(s)" % len(((it.get("advanced_memo") or {}) or {}).get("memos", []) or []),
                     "payload": {"advanced_memo": it.get("advanced_memo")}},
                ],
            })
        stages.append(_stage(
            "loop", "Reflection loop (theoretical sampling)",
            "Steps 7–8 · loop · one column per iteration", phase,
            kind="loop",
            stats={"iterations": tree.get("n_iterations"),
                   "slice size": tree.get("slice_size"),
                   "saturation": "reached" if tree.get("saturation_reached") else "not reached",
                   "stop reason": tree.get("stop_reason")},
            artifacts=[chip("output_charmaz_change_tree.json")], detail="changetree",
            status="ok" if tree.get("saturation_reached") else "warn",
            loop={"axis_label": "iteration", "columns": cols},
        ))
        phase += 1

    tail = [f for f in ("output_focused_codes_final.json", "output_advanced_memos_final.json")
            if f in have]
    if tail:
        foc = _load_json(os.path.join(path, "output_focused_codes_final.json")) or []
        stages.append(_stage(
            "focused_final", "Focused categories (final)",
            "Post-loop · consolidated category set", phase,
            stats={"categories": len(foc) if isinstance(foc, list) else 0},
            artifacts=[chip(f) for f in tail], detail="axial",
        ))
        phase += 1
    return phase


def _seeded_stages(path, have, stages, phase, chip):
    if "output_axial_codes_seeded.json" in have:
        seeded = _load_json(os.path.join(path, "output_axial_codes_seeded.json")) or []
        stages.append(_stage(
            "seed", "Seeded axial layer injected",
            "Phase 2 · fixed taxonomy replaces emergent axial coding", phase,
            stats={"leaves": len(seeded) if isinstance(seeded, list) else 0},
            artifacts=[chip("output_axial_codes_seeded.json")], detail="axial",
        ))
        phase += 1

    if "output_taxonomy_assignments.json" in have:
        rows = _load_json(os.path.join(path, "output_taxonomy_assignments.json")) or []
        types = {}
        for r in rows:
            for t in (r.get("edge_case_types") or "").split("|"):
                if t:
                    types[t] = types.get(t, 0) + 1
        agree = sum(1 for r in rows if r.get("agree") is True)
        arts = [chip("output_taxonomy_assignments.json")]
        if "output_taxonomy_assignments.csv" in have:
            arts.append(chip("output_taxonomy_assignments.csv"))
        stages.append(_stage(
            "match", "Taxonomy matching (embedding + LLM)",
            "Phase 2b · codes → taxonomy leaves", phase,
            stats={"assignments": len(rows),
                   "matcher agreement": "%d/%d" % (agree, len(rows)) if rows else "—",
                   "edge cases": sum(1 for r in rows if r.get("edge_case_type") not in (None, "clean"))},
            artifacts=arts, detail="assignments",
        ))
        phase += 1

        stages.append(_stage(
            "edges", "Edge-case typing", "Phase 2c · orphans, straddlers, disagreements", phase,
            stats={k: v for k, v in sorted(types.items(), key=lambda kv: -kv[1])},
            artifacts=[chip("output_taxonomy_category_signals.json")]
                      if "output_taxonomy_category_signals.json" in have else [],
            detail="signals",
            status="warn" if types else "ok",
        ))
        phase += 1

    if "output_taxonomy_gold_score.json" in have:
        stages.append(_stage(
            "gold", "Gold scoring", "Phase 3 · eval-only, never entered a prompt", phase,
            artifacts=[chip("output_taxonomy_gold_score.json")], detail="gold",
        ))
        phase += 1
    return phase


# --------------------------------------------------------------------------
# SemEval corpus (raw articles + gold annotations)
# --------------------------------------------------------------------------

def semeval_roots(data_root):
    """Find every <split>/ dir that has raw-documents/ under it."""
    out = []
    for dirpath, dirnames, _ in os.walk(data_root):
        if "raw-documents" in dirnames:
            out.append(os.path.relpath(dirpath, data_root).replace(os.sep, "/"))
            dirnames[:] = [d for d in dirnames if d != "raw-documents"]
    return sorted(out)


def _parse_subtask1(path):
    """article -> [ {entity, start, end, main_role, fine_roles[]} ]"""
    by_article = {}
    for line in (_read_text(path) or "").splitlines():
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 5:
            continue
        art, ent, start, end, main = cols[:5]
        try:
            start, end = int(start), int(end)
        except ValueError:
            continue
        by_article.setdefault(art, []).append({
            "entity": ent, "start": start, "end": end,
            "main_role": main, "fine_roles": [c for c in cols[5:] if c.strip()],
        })
    for v in by_article.values():
        v.sort(key=lambda s: (s["start"], s["end"]))
    return by_article


def _parse_subtask2(path):
    """article -> {narratives:[], subnarratives:[]}"""
    out = {}
    for line in (_read_text(path) or "").splitlines():
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 2:
            continue
        art = cols[0]
        narr = [s for s in cols[1].split(";") if s.strip()] if len(cols) > 1 else []
        sub = [s for s in cols[2].split(";") if s.strip()] if len(cols) > 2 else []
        out[art] = {"narratives": narr, "subnarratives": sub}
    return out


def _parse_subtask3(path):
    """article -> [ {narrative, subnarrative, explanation} ]"""
    out = {}
    for line in (_read_text(path) or "").splitlines():
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 2:
            continue
        out.setdefault(cols[0], []).append({
            "narrative": cols[1] if len(cols) > 1 else "",
            "subnarrative": cols[2] if len(cols) > 2 else "",
            "explanation": cols[3] if len(cols) > 3 else "",
        })
    return out


def _domain_of(article_id, gold2):
    """URW / CC, from the filename token when present, else from the gold
    narrative prefixes (non-English filenames don't encode the domain)."""
    up = article_id.upper()
    if "_CC_" in up:
        return "CC"
    if "_UA_" in up or "_URW_" in up:
        return "URW"
    for n in (gold2 or {}).get("narratives", []):
        head = n.split(":", 1)[0].strip().upper()
        if head in ("CC", "URW"):
            return head
    return "?"


def semeval_index(data_root):
    roots = []
    for root in semeval_roots(data_root):
        raw = os.path.join(data_root, root, "raw-documents")
        labels = os.path.join(data_root, root, "labels")
        langs = []
        for lang in sorted(os.listdir(raw)):
            ldir = os.path.join(raw, lang)
            if not os.path.isdir(ldir):
                continue
            arts = sorted(f for f in os.listdir(ldir) if f.endswith(".txt"))
            gold1 = _parse_subtask1(os.path.join(labels, lang, "subtask-1-annotations.txt"))
            gold2 = _parse_subtask2(os.path.join(labels, lang, "subtask-2-annotations.txt"))
            langs.append({
                "lang": lang,
                "n_articles": len(arts),
                "n_annotated": len([a for a in arts if a in gold1 or a in gold2]),
                "n_spans": sum(len(gold1.get(a, [])) for a in arts),
                "articles": [{
                    "id": a,
                    "domain": _domain_of(a, gold2.get(a, {})),
                    "n_spans": len(gold1.get(a, [])),
                    "n_narratives": len(gold2.get(a, {}).get("narratives", [])),
                } for a in arts],
            })
        roots.append({"root": root, "langs": langs})
    return roots


def semeval_article(data_root, root, lang, article_id):
    raw = os.path.join(data_root, root, "raw-documents", lang, article_id)
    if not os.path.isfile(raw):
        return None
    text = _read_text(raw) or ""
    ldir = os.path.join(data_root, root, "labels", lang)
    spans = _parse_subtask1(os.path.join(ldir, "subtask-1-annotations.txt")).get(article_id, [])
    narr = _parse_subtask2(os.path.join(ldir, "subtask-2-annotations.txt")).get(article_id, {})
    expl = _parse_subtask3(os.path.join(ldir, "subtask-3-annotations.txt")).get(article_id, [])
    # verify offsets against the text so drift is visible rather than silent
    for s in spans:
        actual = text[s["start"]:s["end"] + 1]
        s["offset_ok"] = actual.strip() == s["entity"].strip()
        s["actual"] = actual
    return {"id": article_id, "lang": lang, "root": root, "text": text,
            "spans": spans, "narratives": narr, "explanations": expl}


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

class Handler(SimpleHTTPRequestHandler):
    data_root = None

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, fmt, *args):
        if "--verbose" in sys.argv:
            super().log_message(fmt, *args)

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _safe(self, rel):
        """Resolve a data-relative path, refusing anything outside data_root."""
        full = os.path.abspath(os.path.join(self.data_root, rel))
        if not full.startswith(os.path.abspath(self.data_root) + os.sep):
            return None
        return full

    def do_GET(self):
        parsed = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        route = parsed.path

        if not route.startswith("/api/"):
            if route == "/":
                self.path = "/index.html"
            return super().do_GET()

        try:
            if route == "/api/runs":
                return self._json({
                    "data_root": os.path.abspath(self.data_root),
                    "repo": REPO,
                    "runs": discover_runs(self.data_root),
                    "semeval": semeval_index(self.data_root),
                })

            if route == "/api/run":
                man = build_manifest(self.data_root, unquote(q.get("id", "")))
                return self._json(man or {"error": "run not found"}, 200 if man else 404)

            if route == "/api/artifact":
                run, name = unquote(q.get("run", "")), q.get("name", "")
                if name not in ARTIFACT_META:
                    return self._json({"error": "unknown artifact"}, 400)
                full = self._safe(os.path.join(run, name))
                if not full or not os.path.isfile(full):
                    return self._json({"error": "not found"}, 404)
                if name.endswith(".json"):
                    return self._json({"name": name, "kind": "json",
                                       "data": _load_json(full)})
                if name.endswith(".pkl"):
                    return self._json({"name": name, "kind": "binary",
                                       "size": os.path.getsize(full)})
                return self._json({"name": name,
                                   "kind": "csv" if name.endswith(".csv") else "text",
                                   "text": _read_text(full)})

            if route == "/api/semeval/article":
                art = semeval_article(self.data_root, unquote(q.get("root", "")),
                                      q.get("lang", ""), q.get("id", ""))
                return self._json(art or {"error": "not found"}, 200 if art else 404)

            return self._json({"error": "unknown route"}, 404)
        except Exception as exc:  # never take the server down on a bad run dir
            import traceback
            traceback.print_exc()
            return self._json({"error": str(exc)}, 500)


def main():
    ap = argparse.ArgumentParser(description="GTA pipeline viewer (local).")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data", default=os.path.join(REPO, "data"),
                    help="root of the data tree to scan (default: <repo>/data)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    Handler.data_root = os.path.abspath(args.data)
    if not os.path.isdir(Handler.data_root):
        sys.exit("data directory not found: %s" % Handler.data_root)

    runs = discover_runs(Handler.data_root)
    print("GTA viewer")
    print("  data   : %s" % Handler.data_root)
    print("  runs   : %d found" % len(runs))
    for r in runs[:12]:
        print("           [%-10s] %s" % (r["arm"], r["id"]))
    if len(runs) > 12:
        print("           … and %d more" % (len(runs) - 12))
    print("  serving: http://%s:%d" % (args.host, args.port))
    try:
        HTTPServer((args.host, args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
