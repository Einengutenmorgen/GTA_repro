# article_chunking.py
"""
Whole-article "chunking" for the SemEval-2025 Task 10 dataset focus.

chunking.py is turn-aware Q&A chunking purpose-built for Silan-Ciruelas
interview transcripts (PDFs with Speaker N [HH:MM:SS] turns). SemEval-2025
Task 10 articles have no such structure: per the SemEval STUDY CONTEXT block
(study_contexts.SEMEVAL_STUDY_CONTEXT), the unit of analysis is the WHOLE
ARTICLE. So this module does the simplest thing that is still correct: one
article file becomes exactly one chunk, no sub-splitting.

Each chunk dict uses the same keys chunking.chunk_transcript produces
(chunk_id, source_id, unit, q_index, text) so it is a drop-in input for
gta_pipeline.run_open_coding / main.py's chunk_index persistence -- nothing
downstream needs to know which loader produced a chunk.

Matches the official SemEval-2025 Task 10 release layout:

    <root>/{train,dev,test}/raw-documents/<LANG>/<article_id>.txt   (article text)
    <root>/{train,dev}/labels/<LANG>/subtask-{1,2,3}-annotations.txt (GOLD LABELS)

Each raw-documents/<LANG>/*.txt file is UTF-8, title on line 1, blank line,
then body from line 3 on. This loader reads the whole file (title included)
as the chunk's text.

SAFETY (load-bearing): the labels/ files contain the actual gold-standard
role annotations (main_role, fine-grained_roles per entity mention) -- the
exact answer space this study measures inductive-coding recovery against.
labels/ sits as a SIBLING of raw-documents/ under the same split directory,
so a caller pointing target_dir at a split root (e.g. ".../train") instead
of the raw-documents subtree (".../train/raw-documents/EN") would, with a
naive recursive glob, sweep the annotation files in as if they were articles
and feed gold labels straight to the model. extract_and_chunk_articles
therefore hard-excludes (a) any path with a "labels" directory component and
(b) any filename containing "annotation", regardless of what directory it's
pointed at -- defense in depth on top of pointing it at the right folder in
the first place. Excluded files are reported (count + paths) so a
misconfigured target_dir is visible rather than silently "successful."
"""
import os
import glob
from typing import Dict, List, Optional, Tuple

ARTICLE_GLOB = "*.txt"

# Directory names that must never be descended into / read from, no matter
# where target_dir points. Case-insensitive.
EXCLUDED_DIR_NAMES = {"labels"}

# Filenames containing any of these substrings are excluded even if they
# slip through the directory check (e.g. a labels file copied alongside
# articles). Case-insensitive.
EXCLUDED_FILENAME_SUBSTRINGS = ("annotation", "-labels", "_labels")


def source_id_from_path(article_path: str) -> str:
    """Stable article id = filename stem (no dir, no extension)."""
    return os.path.splitext(os.path.basename(article_path))[0]


def _read_article(article_path: str) -> str:
    """Return the article's raw text. Swap this out if your corpus isn't
    one-plain-text-file-per-article (e.g. extract a 'text' field from JSON)."""
    with open(article_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _is_excluded(path: str, target_dir: str) -> bool:
    rel = os.path.relpath(path, target_dir)
    parts = rel.split(os.sep)
    dir_parts = parts[:-1]
    filename = parts[-1].lower()
    if any(p.lower() in EXCLUDED_DIR_NAMES for p in dir_parts):
        return True
    if any(sub in filename for sub in EXCLUDED_FILENAME_SUBSTRINGS):
        return True
    return False


def extract_and_chunk_articles(
    target_dir: str, unit: Optional[str] = None
) -> Tuple[List[Dict], Dict]:
    """Read every article file under target_dir; one article == one chunk.

    Hard-excludes anything under a "labels" directory or matching a
    labels/annotations filename pattern (see module docstring) -- this is
    the gold-standard role scheme and must never reach the model. Returns
    (chunks, chunk_index) -- the same shape main.extract_and_chunk_interviews
    returns, so it plugs into main.py and gta_pipeline.run_open_coding
    without any changes on their end.
    """
    all_paths = sorted(
        glob.glob(os.path.join(target_dir, "**", ARTICLE_GLOB), recursive=True)
    )
    excluded = [p for p in all_paths if _is_excluded(p, target_dir)]
    included = [p for p in all_paths if p not in set(excluded)]

    if excluded:
        print(
            f"  -> SAFETY: excluded {len(excluded)} labels/annotation file(s) "
            f"under {target_dir} (never fed to the model):"
        )
        for p in excluded:
            print(f"       - {os.path.relpath(p, target_dir)}")

    chunks: List[Dict] = []
    for path in included:
        text = _read_article(path)
        if not text:
            continue
        source_id = source_id_from_path(path)
        chunks.append(
            {
                "chunk_id": f"{source_id}_c0000",
                "source_id": source_id,
                "unit": unit,
                "q_index": 0,
                "text": text,
            }
        )

    chunk_index = {c["chunk_id"]: c for c in chunks}
    if len(chunk_index) != len(chunks):
        raise ValueError(
            f"Duplicate chunk_id detected in {target_dir}: "
            f"{len(chunks)} chunks but {len(chunk_index)} unique ids. "
            "Two source articles likely share a filename stem."
        )
    return chunks, chunk_index


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Whole-article chunker (SemEval focus)")
    ap.add_argument("target_dir")
    args = ap.parse_args()
    chunks, _ = extract_and_chunk_articles(args.target_dir)
    print(f"{len(chunks)} article chunks")
    for c in chunks[:3]:
        print(f"\n--- {c['chunk_id']} (source={c['source_id']}) ---")
        print(c["text"][:300])