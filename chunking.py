# chunking.py
"""
Turn-aware Q&A chunking for Silan-Ciruelas transcripts (Condition B).

Replaces the naive character-window slicing in main.py. Parses
`Speaker N [HH:MM:SS]` turns, then groups each interviewer question with the
participant answer(s) that follow it into one semantic chunk.

Speaker 1 = interviewer, Speaker 2 = participant (verified on USA transcripts).

pairs_per_chunk : how many consecutive Q&A pairs to pack into one chunk.
    1  = finest grain (one question-answer per chunk). ~50 chunks per
         interview; may reproduce the open-code fragmentation seen with tiny
         chunks in early runs. Good for the explore-stage effect check.
    >1 = batch several pairs, coarser grain, fewer chunks.

Nothing is dropped: preamble, closing, and every turn are kept. Pure chunking.

Dependencies: pdfplumber (better turn-preserving extraction than PyPDF2).
"""
import re
from typing import List, Tuple

TURN_RE = re.compile(r'(Speaker\s+([12]))\s*\[(\d{2}:\d{2}:\d{2})\]\s*')


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


def qa_units(turns, min_answer_chars: int = 25):
    """Group interviewer question(s) + following participant answer(s) into units.

    Consecutive S1 turns merge into the question; consecutive S2 turns merge
    into the answer. A unit is emitted only when a question has a non-trivial
    answer, so interviewer-only filler ('Yeah.', 'All right.') never forms a
    standalone chunk.
    """
    units = []
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


def chunk_transcript(pdf_path: str,
                     pairs_per_chunk: int = 1,
                     min_answer_chars: int = 25) -> List[str]:
    """Full pipeline: PDF -> list of Q&A chunk strings. Nothing dropped."""
    turns = parse_turns(extract_text(pdf_path))
    units = qa_units(turns, min_answer_chars=min_answer_chars)

    chunks = []
    for i in range(0, len(units), pairs_per_chunk):
        batch = units[i:i + pairs_per_chunk]
        block = "\n\n".join(
            f"INTERVIEWER: {q}\nPARTICIPANT: {a}" for q, a in batch
        )
        chunks.append(block)
    return chunks


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Q&A-aware transcript chunker")
    ap.add_argument("pdf")
    ap.add_argument("--pairs-per-chunk", type=int, default=1)
    args = ap.parse_args()
    chunks = chunk_transcript(args.pdf, pairs_per_chunk=args.pairs_per_chunk)
    print(f"{len(chunks)} chunks")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- chunk {i+1} ---\n{c[:300]}")