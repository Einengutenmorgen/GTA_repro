# chunking.py
"""
Turn-aware Q&A chunking for Silan-Ciruelas transcripts (Condition B).

Replaces the naive character-window slicing in main.py. Parses
`Speaker N [HH:MM:SS]` turns, then groups each interviewer question with the
participant answer(s) that follow it into one semantic unit.

Speaker 1 = interviewer, Speaker 2 = participant (verified on USA transcripts).

pairs_per_chunk : how many consecutive Q&A pairs to pack into one chunk.
    1  = finest grain (one question-answer per chunk). ~50 chunks per
         interview; may reproduce the open-code fragmentation seen with tiny
         chunks in early runs. Good for the explore-stage effect check.
    >1 = batch several pairs, coarser grain, fewer chunks.

Nothing is dropped: preamble, closing, and every turn are kept. Pure chunking.

Structured output (v1.3+)
-------------------------
`chunk_transcript` now returns a list of dicts, one per chunk, instead of bare
strings. Each dict carries the identity needed for the Straussian empty-slot
escalation ladder (re-pass a participant's interview; retrieve similar Q&A):

    {
      "chunk_id":  "<source_id>_c0007",   # stable, unique within a run
      "source_id": "<pdf stem>",           # participant identity
      "unit":      "USA",                  # country / analysis unit (or None)
      "q_index":   7,                      # ordinal of the FIRST Q&A unit in chunk
      "question":  "<interviewer text>",   # questions only (no answer, no scaffold)
      "answer":    "<participant text>",
      "text":      "INTERVIEWER: ...\nPARTICIPANT: ...",  # exact LLM input, unchanged
    }

`text` is byte-identical to what the previous string-returning version fed the
model, so open coding's prompt input does not change. When pairs_per_chunk > 1,
`question`/`answer` hold the FIRST unit in the batch and `questions`/`answers`
hold all of them (see below), so single-question retrieval stays well-defined
regardless of coding chunk size.

For question-to-question similarity (step 2.5) build the index from qa_units
directly (one question per unit), NOT from chunks, so the retrieval grain is
decoupled from pairs_per_chunk. Use iter_qa_units() for that.

Dependencies: pdfplumber (better turn-preserving extraction than PyPDF2).
"""
import os
import re
from typing import List, Tuple, Optional, Dict

TURN_RE = re.compile(r'(Speaker\s+([12]))\s*\[(\d{2}:\d{2}:\d{2})\]\s*')


def source_id_from_path(pdf_path: str) -> str:
    """Stable participant id = PDF file stem (no dir, no extension)."""
    return os.path.splitext(os.path.basename(pdf_path))[0]


def extract_text(pdf_path: str) -> str:
    import pdfplumber
    txt = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                txt.append(t)
    return "\n".join(txt)


def parse_turns(text: str) -> List[Tuple[str, str]]:
    """Return [(speaker_num, body), ...] in order. speaker_num is '1' or '2'."""
    parts = TURN_RE.split(text)
    turns = []
    i = 1
    while i + 3 < len(parts):
        spk_num = parts[i + 1]      # capture group 2 = the digit
        body = parts[i + 3].strip()
        if body:
            turns.append((spk_num, body))
        i += 4
    return turns


def qa_units(turns, min_answer_chars: int = 25) -> List[Tuple[str, str]]:
    """Group interviewer question(s) + following participant answer(s) into units.

    Consecutive S1 turns merge into the question; consecutive S2 turns merge
    into the answer. A unit is emitted only when a question has a non-trivial
    answer, so interviewer-only filler ('Yeah.', 'All right.') never forms a
    standalone chunk.

    Returns a list of (question, answer) string pairs in transcript order.
    """
    units: List[Tuple[str, str]] = []
    q, a = [], []

    def flush():
        if q and a and sum(len(x) for x in a) >= min_answer_chars:
            units.append((" ".join(q).strip(), " ".join(a).strip()))

    for spk, body in turns:
        if spk == '1':
            if a:               # answer complete -> close previous unit
                flush()
                q, a = [], []
            q.append(body)
        elif spk == '2':
            a.append(body)
    flush()
    return units


def _unit_to_text(q: str, a: str) -> str:
    """The exact string the LLM sees for one Q&A unit (unchanged from v1.2)."""
    return f"INTERVIEWER: {q}\nPARTICIPANT: {a}"


def iter_qa_units(pdf_path: str,
                  min_answer_chars: int = 2,
                  unit: Optional[str] = None) -> List[Dict]:
    """One dict per Q&A unit (always one question each), for the Q-Q sim index.

    This is the retrieval grain: independent of pairs_per_chunk. Build the
    question-similarity cache from the concatenation of this across all PDFs.
    """
    source_id = source_id_from_path(pdf_path)
    units = qa_units(parse_turns(extract_text(pdf_path)),
                     min_answer_chars=min_answer_chars)
    out = []
    for idx, (q, a) in enumerate(units):
        out.append({
            "chunk_id": f"{source_id}_c{idx:04d}",
            "source_id": source_id,
            "unit": unit,
            "q_index": idx,
            "question": q,
            "answer": a,
            "text": _unit_to_text(q, a),
        })
    return out


def chunk_transcript(pdf_path: str,
                     pairs_per_chunk: int = 1,
                     min_answer_chars: int = 25,
                     unit: Optional[str] = None) -> List[Dict]:
    """Full pipeline: PDF -> list of structured chunk dicts. Nothing dropped.

    When pairs_per_chunk == 1 (default) each chunk is exactly one Q&A unit and
    `question`/`answer` are that unit. When > 1, the chunk batches several
    units: `text` concatenates them (exactly as before), `question`/`answer`
    hold the FIRST unit, and `questions`/`answers` list all units in the batch.
    """
    source_id = source_id_from_path(pdf_path)
    units = qa_units(parse_turns(extract_text(pdf_path)),
                     min_answer_chars=min_answer_chars)

    chunks: List[Dict] = []
    for start in range(0, len(units), pairs_per_chunk):
        batch = units[start:start + pairs_per_chunk]
        questions = [q for q, _ in batch]
        answers = [a for _, a in batch]
        text = "\n\n".join(_unit_to_text(q, a) for q, a in batch)

        chunk = {
            "chunk_id": f"{source_id}_c{start:04d}",
            "source_id": source_id,
            "unit": unit,
            "q_index": start,
            "question": questions[0],
            "answer": answers[0],
            "text": text,
        }
        if pairs_per_chunk > 1:
            chunk["questions"] = questions
            chunk["answers"] = answers
        chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Q&A-aware transcript chunker")
    ap.add_argument("pdf")
    ap.add_argument("--pairs-per-chunk", type=int, default=1)
    args = ap.parse_args()
    chunks = chunk_transcript(args.pdf, pairs_per_chunk=args.pairs_per_chunk)
    print(f"{len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"\n--- {c['chunk_id']} (source={c['source_id']}, q_index={c['q_index']}) ---")
        print(c["text"][:300])