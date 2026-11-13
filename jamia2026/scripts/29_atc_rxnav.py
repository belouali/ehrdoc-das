#!/usr/bin/env python
"""A_condition_scan: medication name lists for the eight added conditions from WHO ATC classes,
read from NLM RxNav (user's choice 2026-11-07, "Use NLM service"), and their match to the eICU
drug vocabulary. No outcome is read.

Only ATC codes and RxNorm identifiers are sent to NLM. eICU drug names never leave the machine:
the class members come down first and are matched locally.

Stage 1, fetch (public data, cached in results/atc_rxnav_members.csv):
  1. each listed code of level 1-4 is checked with rxclass/class/byId; its members come from
     rxclass/classMembers (relaSource ATC). A level-5 code is read from the members of its
     level-4 parent, keeping members whose SourceId equals the code. A code with no class or no
     member is reported as missing.
  2. members listed as precise ingredients (PIN) are resolved to their ingredient (IN)
  3. brand names (BN) of each ingredient come from rxcui/{id}/related?tty=BN
  4. exclude_atc codes remove their ingredients from the condition
Stage 2, match (local): a drugname counts for a condition if it contains an ingredient or brand
name as a whole word (case-insensitive). Run on the cohort's medication (M) and admissionDrug (A)
rows, the table chosen per condition in phenotypes_scan.yaml.

Outputs: results/atc_rxnav_members.csv, results/atc_code_check.csv, results/scan_med_coverage.csv
(aggregate); rowlevel/eicu_flags_scan_med.parquet (row level, external drive).

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with requests \
      python scripts/29_atc_rxnav.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests
import yaml

API = "https://rxnav.nlm.nih.gov/REST"
MIN_RECORDS = 100  # per-hospital evidence count used for the coverage columns, as script 28


def get(session, path, **params):
    for attempt in range(4):
        r = session.get(f"{API}/{path}", params=params, timeout=30)
        if r.status_code == 200:
            time.sleep(0.06)  # NLM asks for at most 20 requests per second
            return r.json()
        time.sleep(1 + attempt)
    r.raise_for_status()


def members(session, class_id):
    d = get(session, "rxclass/classMembers.json", classId=class_id, relaSource="ATC")
    rows = []
    for m in d.get("drugMemberGroup", {}).get("drugMember", []):
        a = {x["attrName"]: x["attrValue"] for x in m.get("nodeAttr", [])}
        c = m["minConcept"]
        rows.append(dict(rxcui=c["rxcui"], name=c["name"], tty=c["tty"], atc5=a.get("SourceId", "")))
    return rows


def fetch(cfg_scan):
    s = requests.Session()
    version = get(s, "version.json")["version"]
    rows, checks, cache = [], [], {}
    for table in ("in_stay", "home"):
        for cond, rule in cfg_scan[table].items():
            for code in rule["atc"] + rule.get("exclude_atc", []):
                role = "exclude" if code in rule.get("exclude_atc", []) else "include"
                if len(code) == 7:  # level 5: read from the level-4 parent
                    parent = code[:5]
                    if parent not in cache:
                        cache[parent] = members(s, parent)
                    got = [m for m in cache[parent] if m["atc5"] == code]
                    exists = bool(got)
                else:
                    cls = get(s, "rxclass/class/byId.json", classId=code)
                    exists = bool(cls.get("rxclassMinConceptList"))
                    got = members(s, code) if exists else []
                checks.append(dict(condition=cond, code=code, role=role, exists=exists, members=len(got)))
                for m in got:
                    rows.append(dict(table=table, condition=cond, code=code, role=role, **m))
    mem = pd.DataFrame(rows)
    # precise ingredients to ingredients
    ing = {}
    for rx in mem.loc[mem.tty == "PIN", "rxcui"].unique():
        d = get(s, f"rxcui/{rx}/related.json", tty="IN")
        cs = [c for g in d["relatedGroup"].get("conceptGroup", []) for c in g.get("conceptProperties", [])]
        ing[rx] = cs[0] if cs else None
    mem["in_rxcui"] = [ing[r]["rxcui"] if t == "PIN" and ing.get(r) else r for r, t in zip(mem.rxcui, mem.tty)]
    mem["in_name"] = [ing[r]["name"] if t == "PIN" and ing.get(r) else n
                      for r, t, n in zip(mem.rxcui, mem.tty, mem.name)]
    brands = {}
    for rx in mem.in_rxcui.unique():
        d = get(s, f"rxcui/{rx}/related.json", tty="BN")
        cs = [c for g in d["relatedGroup"].get("conceptGroup", []) for c in g.get("conceptProperties", [])]
        brands[rx] = sorted({c["name"] for c in cs})
    mem["brands"] = ["|".join(brands[r]) for r in mem.in_rxcui]
    mem["rxnorm_version"] = version
    return mem, pd.DataFrame(checks), version


def name_lists(mem):
    """Per condition: ingredient and brand names, after exclusions."""
    out = {}
    for cond, g in mem.groupby("condition"):
        drop = set(g.loc[g.role == "exclude", "in_rxcui"])
        keep = g[(g.role == "include") & ~g.in_rxcui.isin(drop)]
        names = set(keep.in_name.str.lower()) | set(keep.name.str.lower())
        for b in keep.brands:
            names |= {x.lower() for x in b.split("|") if x}
        out[cond] = sorted(names)
    return out


def word_mask(values: pd.Series, names) -> pd.Series:
    pat = re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                     + r")(?![a-z0-9])")
    return pd.Series([bool(pat.search(str(v).lower())) for v in values], index=values.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    res = base / cfg["paths"]["results"]
    rowlevel = Path(cfg["paths"]["rowlevel"])
    E = cfg["paths"]["eicu"]
    scan = yaml.safe_load(open(base / "configs" / "phenotypes_scan.yaml"))

    mem_path, chk_path = res / "atc_rxnav_members.csv", res / "atc_code_check.csv"
    if args.refetch or not mem_path.exists():
        mem, chk, version = fetch(scan)
        mem.to_csv(mem_path, index=False)
        chk.to_csv(chk_path, index=False)
    mem = pd.read_csv(mem_path, dtype=str, keep_default_na=False)
    chk = pd.read_csv(chk_path)
    print(f"RxNorm {mem.rxnorm_version.iloc[0]}; codes missing: "
          + (", ".join(chk.loc[~chk.exists | (chk.members == 0), "code"]) or "none"))
    lists = name_lists(mem)

    ref = pd.read_parquet(rowlevel / "eicu_flags.parquet", columns=["patientunitstayid", "hospitalid"])
    con = duckdb.connect()
    con.register("coh", ref)
    flags = ref.copy()
    rows = []
    for table, src, fname in (("in_stay", "M", "medication.parquet"), ("home", "A", "admissionDrug.parquet")):
        con.execute(f"""CREATE OR REPLACE TEMP TABLE src AS SELECT t.patientunitstayid id, t.drugname v
                        FROM '{E}/{fname}' t JOIN coh USING (patientunitstayid) WHERE t.drugname IS NOT NULL""")
        vocab = con.execute("SELECT v, count(*) n FROM src GROUP BY v").df()
        for cond in scan[table]:
            hit = vocab[word_mask(vocab.v, lists[cond])]
            con.register("hit", hit[["v"]])
            ids = set(con.execute("SELECT DISTINCT id FROM src JOIN hit USING (v)").df()["id"])
            con.unregister("hit")
            col = f"{cond}_{src}"
            flags[col] = flags.patientunitstayid.isin(ids)
            per_h = flags.groupby("hospitalid")[col].sum()
            g = mem[mem.condition == cond]
            rows.append(dict(condition=cond, source=src, atc_codes=len(scan[table][cond]["atc"]),
                             ingredients=g.loc[g.role == "include", "in_rxcui"].nunique(),
                             names_searched=len(lists[cond]), eicu_names_matched=len(hit),
                             rows_matched=int(hit.n.sum()), stays=int(flags[col].sum()),
                             share_stays=flags[col].mean(), hospitals_any=int((per_h > 0).sum()),
                             hospitals_ge100=int((per_h >= MIN_RECORDS).sum())))
    flags.to_parquet(rowlevel / "eicu_flags_scan_med.parquet", index=False)
    out = pd.DataFrame(rows)
    out.round(4).to_csv(res / "scan_med_coverage.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
