#!/usr/bin/env python
"""Past medical history (H source) from MIMIC-IV-Note discharge summaries.

For every MIMIC-IV cohort stay (01_build_flags.py), find the "Past Medical
History:" section of discharge summaries and flag the four conditions.

  <cond>_H        index admission's summary, published past_history terms
                  (word-bounded) plus the free-text abbreviations in
                  configs/external.yaml (primary H)
  <cond>_Hstrict  index admission's summary, published terms only, matched as
                  substrings exactly as in eICU
  <cond>_Hprior   any summary of an earlier admission stored before the index
                  ICU admission (same rule as H)
  <cond>_Hlatest  the most recent such earlier summary only

Matching is per item: the section is cut into lines and ';' pieces, then ','
pieces. An item counts when it has an include match, no exclude match, no
negation word before the match, and is not followed by a qualifier-only
exclude piece (as in "diabetes, type 1"). Family-history items are skipped.

Outputs (row-level, external drive):
  mimic_history_flags.parquet  one row per cohort stay
  pmh_validation_sample.csv    200 index sections from the reference subset with
                               the flags and empty label columns, for hand review
Aggregate output: results/history_extraction_summary.csv

Usage (from mimic_external/):
  uv run --no-project --with pandas --with pyarrow --with pyyaml --with duckdb \
      python scripts/08_extract_history.py --config configs/external.yaml
  Add --partial to test on an incompletely downloaded file (writes *.partial outputs).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import zlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

CONDITIONS = ["diabetes", "hypertension", "chf", "af"]


def build_rules(ph: dict, nh: dict) -> dict:
    """Compiled include/exclude regexes per condition for H and Hstrict."""
    rules = {"H": {}, "Hstrict": {}}
    for c in CONDITIONS:
        inc = ph[c]["past_history"].get("include", []) or []
        exc = ph[c]["past_history"].get("exclude", []) or []
        bounded = lambda terms: [rf"\b{re.escape(t)}\b" for t in terms]
        h_inc = bounded(inc) + nh["extra_include"].get(c, [])
        h_exc = bounded(exc) + nh["extra_exclude"].get(c, [])
        rules["H"][c] = (re.compile("|".join(h_inc), re.I), re.compile("|".join(h_exc), re.I) if h_exc else None)
        rules["Hstrict"][c] = (re.compile("|".join(re.escape(t) for t in inc), re.I),
                               re.compile("|".join(re.escape(t) for t in exc), re.I) if exc else None)
    return rules


def find_section(text: str, start: re.Pattern, end: re.Pattern, maxc: int) -> str | None:
    if not isinstance(text, str):
        return None
    m = start.search(text)
    if not m:
        return None
    rest = text[m.end():]
    e = end.search(rest)
    return (rest[:e.start()] if e else rest)[:maxc]


def flag_section(sec: str | None, rules: dict, neg: re.Pattern, skip: re.Pattern) -> dict:
    found = {c: False for c in rules}
    if not sec:
        return found
    for line in re.split(r"[\n;]", sec):
        if not line.strip() or skip.match(line):
            continue
        pieces = line.split(",")
        for c, (inc, exc) in rules.items():
            if found[c]:
                continue
            for j, p in enumerate(pieces):
                im = inc.search(p)
                if not im or (exc and exc.search(p)) or neg.search(p[:im.start()]):
                    continue
                nxt = pieces[j + 1] if j + 1 < len(pieces) else ""
                if exc and exc.search(nxt) and not inc.search(nxt):
                    continue
                found[c] = True
                break
    return found


def read_notes(path: str, subjects: set, partial: bool):
    """Yield chunks of discharge notes for cohort subjects; stop cleanly at a truncated end if partial."""
    it = pd.read_csv(path, chunksize=10000,
                     usecols=["note_id", "subject_id", "hadm_id", "note_type", "charttime", "storetime", "text"])
    n = 0
    while True:
        try:
            df = next(it)
        except StopIteration:
            break
        except (EOFError, zlib.error, pd.errors.ParserError, OSError) as ex:
            if not partial:
                raise
            print(f"  stopped at truncated end after {n:,} notes ({type(ex).__name__})")
            break
        n += len(df)
        yield df[df.subject_id.isin(subjects)]
    print(f"  read {n:,} notes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--partial", action="store_true", help="test on an incomplete download")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    nh = cfg["notes_history"]
    ph = yaml.safe_load(open(base / cfg["paths"]["phenotypes"]))
    rules = build_rules(ph, nh)
    start, end = re.compile(nh["section_start"]), re.compile(nh["section_end"])
    neg, skip = re.compile(nh["negation"], re.I), re.compile(nh["skip_item"], re.I)
    tag = ".partial" if args.partial else ""

    note_path = Path(nh["path"])
    sums = note_path.parent.parent / "SHA256SUMS.txt"
    if not args.partial and sums.exists():
        want = [l.split()[0] for l in sums.read_text().splitlines() if l.strip().endswith("note/discharge.csv.gz")]
        h = hashlib.sha256()
        with open(note_path, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        assert want and h.hexdigest() == want[0], "discharge.csv.gz does not match SHA256SUMS.txt"
        print("  SHA256 checked")

    M = cfg["paths"]["mimic"]
    coh = pd.read_parquet(rowlevel / "mimic_flags.parquet", columns=["subject_id", "hadm_id", "stay_id", "anchor_year_group"])
    times = duckdb.execute(f"""SELECT i.stay_id, i.intime, a.admittime FROM '{M}/icu/icustays.parquet' i
        JOIN '{M}/hosp/admissions.parquet' a USING (hadm_id)""").df()
    adm = duckdb.execute(f"SELECT hadm_id, dischtime FROM '{M}/hosp/admissions.parquet'").df()
    coh = coh.merge(times, on="stay_id", how="left")
    split = pd.read_parquet(rowlevel / "mimic_split.parquet")[["stay_id", "reference"]]
    coh = coh.merge(split, on="stay_id", how="left")

    rows = []
    for df in read_notes(str(note_path), set(coh.subject_id), args.partial):
        for r in df.itertuples(index=False):
            sec = find_section(r.text, start, end, nh["max_section_chars"])
            fh = flag_section(sec, rules["H"], neg, skip)
            fs = flag_section(sec, rules["Hstrict"], neg, skip)
            rows.append(dict(note_id=r.note_id, subject_id=r.subject_id, hadm_id=r.hadm_id, note_type=r.note_type,
                             charttime=r.charttime, storetime=r.storetime, has_section=sec is not None,
                             section=sec if sec is not None else "",
                             **{f"{c}_H": fh[c] for c in CONDITIONS}, **{f"{c}_Hstrict": fs[c] for c in CONDITIONS}))
    notes = pd.DataFrame(rows)
    for col in ["charttime", "storetime"]:
        notes[col] = pd.to_datetime(notes[col])
    notes = notes.merge(adm, on="hadm_id", how="left")
    print(f"  {len(notes):,} notes for cohort subjects, {notes.has_section.mean():.1%} with a history section")

    hcols = [f"{c}_H" for c in CONDITIONS]
    scols = [f"{c}_Hstrict" for c in CONDITIONS]
    # index admission
    idx = coh[["stay_id", "hadm_id"]].merge(notes, on="hadm_id", how="inner")
    idx_any = idx.groupby("stay_id").agg(has_index_note=("note_id", "size"), has_index_section=("has_section", "max"),
                                         **{c: (c, "max") for c in hcols + scols})
    # earlier admissions: different hadm, discharged before the index admission, note stored before ICU admission
    pr = coh[["stay_id", "subject_id", "hadm_id", "intime", "admittime"]].merge(
        notes.drop(columns=["section"]), on="subject_id", suffixes=("", "_note"))
    avail = pr.storetime.fillna(pr.charttime)
    pr = pr[(pr.hadm_id_note != pr.hadm_id) & (pr.dischtime < pr.admittime) & (avail < pr.intime)]
    prior_any = pr.groupby("stay_id").agg(has_prior_note=("note_id", "size"), has_prior_section=("has_section", "max"),
                                          **{c.replace("_H", "_Hprior"): (c, "max") for c in hcols})
    latest = pr.sort_values("charttime").groupby("stay_id").tail(1).set_index("stay_id")[hcols]
    latest.columns = [c.replace("_H", "_Hlatest") for c in hcols]

    out = coh[["stay_id", "subject_id", "hadm_id", "anchor_year_group", "reference"]].set_index("stay_id")
    out = out.join(idx_any).join(prior_any).join(latest)
    flag_cols = [c for c in out.columns if c.endswith(("_H", "_Hstrict", "_Hprior", "_Hlatest"))
                 or c.startswith("has_")]
    out[flag_cols] = out[flag_cols].fillna(0).astype(bool)
    out = out.reset_index()
    out.to_parquet(rowlevel / f"mimic_history_flags{tag}.parquet", index=False)

    # validation sample: index sections of reference-subset stays
    samp = idx[idx.has_section].merge(out[["stay_id", "reference"]], on="stay_id")
    samp = samp[samp.reference].sample(n=min(200, int(samp.reference.sum())), random_state=20261105)
    samp = samp[["stay_id", "note_id", "section"] + hcols + scols]
    for c in CONDITIONS:
        samp[f"label_{c}"] = ""
    samp.to_csv(rowlevel / f"pmh_validation_sample{tag}.csv", index=False)

    # aggregate summary
    def summarize(g: pd.DataFrame, label: str) -> dict:
        d = dict(group=label, stays=len(g), index_note=g.has_index_note.mean(), index_section=g.has_index_section.mean(),
                 prior_note=g.has_prior_note.mean(), prior_section=g.has_prior_section.mean())
        for c in CONDITIONS:
            for s in ["H", "Hstrict", "Hprior", "Hlatest"]:
                d[f"{c}_{s}"] = g[f"{c}_{s}"].mean()
        return d
    summ = [summarize(out, "all")]
    summ += [summarize(g, f"reference={r}") for r, g in out.groupby("reference")]
    summ += [summarize(g, f"years {y}") for y, g in out.groupby("anchor_year_group")]
    e = pd.read_parquet(rowlevel / "eicu_flags.parquet")
    summ.append(dict(group="eICU pastHistory (for comparison)", stays=len(e),
                     **{f"{c}_H": e[f"{c}_H"].mean() for c in CONDITIONS}))
    summ = pd.DataFrame(summ)
    summ.round(4).to_csv(res / f"history_extraction_summary{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).T.to_string())


if __name__ == "__main__":
    main()
