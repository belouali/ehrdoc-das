#!/usr/bin/env python
"""A_condition_scan, medication specificity check for chronic pulmonary disease (requested by the
user 2026-11-07). Outcome-free. Among eICU stays given ipratropium (ATC R03BB01) but no long-acting
agent, what share has chronic pulmonary disease recorded in the diagnosis (D) or history (H) source?
Compared with stays given a long-acting agent, and with all stays.

Two D/H definitions:
  charlson  Quan ICD (icd9code column) or pastHistory picklist, as scripts/11_charlson_features.py
            (COPD, asthma and restrictive disease)
  copd      the locked keyword rules in configs/phenotypes_ext.yaml (COPD only)

Output: results/ipratropium_check.csv (aggregate).

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      python scripts/30_ipratropium_check.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import warnings
from pathlib import Path

import duckdb
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
warnings.filterwarnings("ignore")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s11 = _load("s11", "11_charlson_features.py")
s29 = _load("s29", "29_atc_rxnav.py")
COMP = "chronic_pulmonary_disease"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E = cfg["paths"]["eicu"]
    con = duckdb.connect()

    coh = pd.read_parquet(rowlevel / "eicu_flags_ext.parquet", columns=["patientunitstayid", "copd_D", "copd_H"])
    con.register("ecoh", coh[["patientunitstayid"]])

    # Charlson D: Quan ICD on icd9code, as script 11
    raw = con.execute(f"""SELECT patientunitstayid, icd9code FROM '{E}/diagnosis.parquet'
                          WHERE icd9code IS NOT NULL AND patientunitstayid IN (SELECT patientunitstayid FROM ecoh)""").df()
    rows = []
    for sid, field in zip(raw.patientunitstayid, raw.icd9code):
        for code in str(field).split(","):
            code = code.strip().upper()
            if code:
                v9 = bool(s11.ICD9_SHAPE.match(code))
                c = code.replace(".", "")
                rows.append((sid, c if v9 else None, None if v9 else c))
    con.register("diag", pd.DataFrame(rows, columns=["hadm_id", "icd9_code", "icd10_code"]).drop_duplicates())
    coded = con.execute(f"SELECT hadm_id {s11.quan_case_block()} FROM diag GROUP BY hadm_id").df()
    d_ids = set(coded.loc[coded[COMP] > 0, "hadm_id"])
    # Charlson H: pastHistory picklist, as script 11
    ph = con.execute(f"""SELECT DISTINCT patientunitstayid, lower(pasthistorypath) p FROM '{E}/pastHistory.parquet'
                         WHERE patientunitstayid IN (SELECT patientunitstayid FROM ecoh)""").df()
    h_ids = set(ph.loc[ph.p.str.contains(s11.PASTHISTORY[COMP], regex=True), "patientunitstayid"])
    coh["charlson_D"] = coh.patientunitstayid.isin(d_ids)
    coh["charlson_H"] = coh.patientunitstayid.isin(h_ids)

    # medication groups from the RxNav members (results/atc_rxnav_members.csv)
    mem = pd.read_csv(res / "atc_rxnav_members.csv", dtype=str, keep_default_na=False)
    mem = mem[mem.condition == COMP]

    def names(g):
        out = set(g.in_name.str.lower()) | set(g.name.str.lower())
        for b in g.brands:
            out |= {x.lower() for x in b.split("|") if x}
        return out

    ipra = names(mem[mem.atc5 == "R03BB01"])
    longa = names(mem[(mem.role == "include") & ~mem.atc5.isin(["R03BB01", "R03BB02"])])
    src = con.execute(f"""SELECT t.patientunitstayid id, t.drugname v FROM '{E}/medication.parquet' t
                          JOIN ecoh USING (patientunitstayid) WHERE t.drugname IS NOT NULL""").df()
    vocab = pd.Series(src.v.unique())
    for col, nm in (("ipra", ipra), ("longa", longa)):
        hit = set(vocab[s29.word_mask(vocab, nm)])
        coh[col] = coh.patientunitstayid.isin(set(src.loc[src.v.isin(hit), "id"]))

    groups = {"all stays": coh.index == coh.index,
              "ipratropium, no long-acting": coh.ipra & ~coh.longa,
              "long-acting (any)": coh.longa,
              "neither": ~coh.ipra & ~coh.longa}
    out = []
    for gname, m in groups.items():
        g = coh[m]
        out.append(dict(group=gname, stays=len(g),
                        charlson_D_or_H=(g.charlson_D | g.charlson_H).mean(),
                        charlson_D=g.charlson_D.mean(), charlson_H=g.charlson_H.mean(),
                        copd_D_or_H=(g.copd_D | g.copd_H).mean()))
    out = pd.DataFrame(out)
    out.round(4).to_csv(res / "ipratropium_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
