#!/usr/bin/env python3
"""
Minimal codebook extractor: Silan-Ciruelas relationship-quality xlsx ->
the common {code, definition, ...} schema that alignment.py consumes.

The xlsx has one theme-level sheet per country named "<CC>-T" with a fixed
6-column layout (headers vary only in singular/plural + whitespace, so we
read BY POSITION, not by header text):

    col 0  Theme            (blank / "None" => same theme as the row above)
    col 1  Code             <- becomes `code`
    col 2  Code definition  <- becomes `definition`
    col 3  Example Quote    <- kept as `example`
    col 4  # passages coded <- kept as `n_passages`
    col 5  # participants   <- kept as `n_participants` (salience weight)

The bare "<CC>" sheets are per-participant raw data, NOT the codebook, and
are ignored. `Themes` (country->shared-theme map) and `Theme Codebook`
(theme-level definitions) are left for the axial/selective layer later.

Usage
-----
  # one country, open level (the alignment unit)
  python codebook_extractors.py Silan-Ciruelas_PublicCodebook.xlsx \
      --country USA --out human_open_USA.json

  # all five countries into one file (each row carries its own `unit`)
  python codebook_extractors.py Silan-Ciruelas_PublicCodebook.xlsx \
      --out human_open_ALL.json
"""
import argparse
import json
import sys

import openpyxl

# sheet suffix "-T" is the theme/code-level codebook per country
SHEET_BY_COUNTRY = {
    "USA": "US-T",
    "PHL": "PH-T",
    "TUR": "TR-T",
    "BRA": "BR-T",
    "FRA": "FR-T",
}

# column positions (0-indexed) in every "<CC>-T" sheet
THEME, CODE, DEFN, QUOTE, N_PASS, N_PART = range(6)


def _clean(v):
    if v is None:
        return None
    s = str(v).strip()
    return s if s and s.lower() != "none" else None


def _to_int(v):
    v = _clean(v)
    if v is None:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None  # non-numeric passage/participant cells -> drop the weight, keep the code


def extract_country(wb, country: str) -> list:
    sheet = SHEET_BY_COUNTRY[country]
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))

    out = []
    current_theme = None
    for r in rows[1:]:                       # skip header row
        r = list(r) + [None] * (6 - len(r))  # pad short rows defensively
        theme = _clean(r[THEME])
        if theme is not None:                # a real theme starts a new group;
            current_theme = theme            # blank/"None" => inherit from above

        code = _clean(r[CODE])
        if code is None:
            continue                         # no code text => not a codebook row

        out.append({
            "code": code,
            "definition": _clean(r[DEFN]) or "",
            "example": _clean(r[QUOTE]) or "",
            "theme": current_theme,
            "n_passages": _to_int(r[N_PASS]),
            "n_participants": _to_int(r[N_PART]),
            "unit": country,
            "level": "open",
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--country", choices=sorted(SHEET_BY_COUNTRY),
                    help="one country; omit to extract all five")
    ap.add_argument("--out", default='codebook_transformed.json')
    a = ap.parse_args()

    wb = openpyxl.load_workbook(a.xlsx, data_only=True, read_only=True)
    countries = [a.country] if a.country else sorted(SHEET_BY_COUNTRY)

    codes = []
    for c in countries:
        got = extract_country(wb, c)
        codes.extend(got)
        print(f"{c}: {len(got)} codes", file=sys.stderr)

    json.dump(codes, open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"total {len(codes)} codes -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
