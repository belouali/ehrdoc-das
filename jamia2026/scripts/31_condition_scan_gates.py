#!/usr/bin/env python
"""A_condition_scan, gates 1 and 2 (outcome-free). Settings: configs/screen_settings.yaml
(A_condition_scan) and configs/phenotypes_scan.yaml, both locked 2026-11-07 before computing.

Sources per condition:
  paper's four   D, H, M (and A) from rowlevel/eicu_flags.parquet (the paper's rules)
  Charlson       D = Quan ICD on the eICU icd9code column, H = pastHistory picklist (script 11 rules,
                 severity splits merged); M or A from rowlevel/eicu_flags_scan_med.parquet (script 29)
Constructs: DHM or DHA per sources.by_condition; DHMany (A or M merged) for the paper's four.

Gate 1: for each n in the grid, hospitals with >= 2n evidence-positive stays; two disjoint samples
of n stays per hospital, 20 draws; reliability = Spearman correlation across hospitals of the two
samples' angles to the pooled eICU profile (mean over draws). Each sample already holds n stays, so
the correlation is the reliability at n and the Spearman-Brown step to n is the identity.
n* = smallest grid n with reliability >= 0.80; no such n drops the condition.
Gate 1, revised (gate1_pairwise, locked 2026-11-07, used for gate 2): the same split, but each
sample gives the angle between every pair of hospitals over the 7 lobes; reliability = Spearman
correlation of the two sets of pair angles (mean over draws); an n counts only if the hospitals with
>= 2n stays include >= 32 of the 41 development targets. The first gate 1 stays reported.
Gate 1, full data (gate1_full, locked 2026-11-07, used for scoring): hospitals with >= 100
evidence-positive stays; each hospital's stays split in half (20 draws); Spearman correlation of the
two halves' pair angles, Spearman-Brown corrected to full size; passes at >= 0.80 if those hospitals
include >= 32 of the 41 development targets. Gate 2 then counts targets with >= 100 stays.
Gate 2: >= 32 of the 41 development targets (per task) with >= n* evidence-positive stays.

Outputs: results/condition_scan_gate1.csv, results/condition_scan_gates.csv (aggregate).

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with requests python scripts/31_condition_scan_gates.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s02 = _load("s02", "02_das_external.py")
s11 = _load("s11", "11_charlson_features.py")
TASKS = ["A3_los_gt7d", "A6_hospital_los_gt14d"]


def charlson_DH(con, E):
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
    coded = con.execute(f"SELECT hadm_id {s11.quan_case_block()} FROM diag GROUP BY hadm_id").df().set_index("hadm_id")
    ph = con.execute(f"""SELECT DISTINCT patientunitstayid, lower(pasthistorypath) p FROM '{E}/pastHistory.parquet'
                         WHERE patientunitstayid IN (SELECT patientunitstayid FROM ecoh)""").df()
    hist = {comp: set(ph.loc[ph.p.str.contains(rx, regex=True), "patientunitstayid"])
            for comp, rx in s11.PASTHISTORY.items()}
    return coded, hist


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
    st = yaml.safe_load(open(base / "configs" / "screen_settings.yaml"))["A_condition_scan"]
    assert st["status"] == "locked"
    assert yaml.safe_load(open(base / "configs" / "phenotypes_scan.yaml"))["status"] == "locked"
    g1, g2 = st["gate1_hospital_profile"], st["gate2_targets"]
    grid, draws, bar = g1["n_grid"], 20, 0.80
    min_targets = 32
    assert f">= {min_targets} of the 41" in g2["rule"]
    assert st["gate1_pairwise"]["status"] == "locked" and st["gate1_full"]["status"] == "locked"
    floor = 100
    assert f">= {floor} evidence-positive stays" in st["gate1_full"]["rule"]
    rng_full = np.random.default_rng(st["seed"] + 1)  # separate stream: earlier gates keep their draws

    e = pd.read_parquet(rowlevel / "eicu_flags.parquet")
    med = pd.read_parquet(rowlevel / "eicu_flags_scan_med.parquet").drop(columns="hospitalid")
    assert len(med) == len(e)
    e = e.merge(med, on="patientunitstayid", validate="1:1")
    con = duckdb.connect()
    con.register("ecoh", e[["patientunitstayid"]])
    coded, hist = charlson_DH(con, E)
    sid = e.patientunitstayid
    for cond, comps in st["conditions"]["charlson"].items():
        d = np.zeros(len(e), bool)
        h = np.zeros(len(e), bool)
        for c in comps:
            if c in coded:
                d |= sid.map(coded[c]).fillna(0).to_numpy() > 0
            h |= sid.isin(hist.get(c, set())).to_numpy()
        e[f"{cond}_D"], e[f"{cond}_H"] = d, h
    for cond in st["conditions"]["paper"]:
        e[f"{cond}_X"] = e[f"{cond}_A"] | e[f"{cond}_M"]

    arms = [(c, "DHM", ("D", "H", "M")) for c in st["sources"]["by_condition"]["DHM"]]
    arms += [(c, "DHA", ("D", "H", "A")) for c in st["sources"]["by_condition"]["DHA"]]
    arms += [(c, "DHMany", ("D", "H", "X")) for c in st["conditions"]["paper"]]
    dev = pd.read_csv(res / "task_ceiling_targets_construct.csv")[["task", "target"]].drop_duplicates()

    rng = np.random.default_rng(st["seed"])
    rel_rows, gate_rows = [], []
    for cond, cname, srcs in arms:
        codes = s02.lobe_codes(e, cond, srcs)
        keep = codes >= 0
        cnt = s02.hospital_counts(e, codes, 7)
        pooled = cnt.sum().to_numpy(float)
        ev = e.loc[keep, ["hospitalid"]].assign(lobe=codes[keep])
        by_h = {h: g.lobe.to_numpy() for h, g in ev.groupby("hospitalid")}
        dev_all = set(dev.target)
        n_star = n_pair = None
        for n in grid:
            hs = [h for h, l in by_h.items() if len(l) >= 2 * n]
            n_dev = len(set(hs) & dev_all)
            if len(hs) < 3:
                rel_rows.append(dict(condition=cond, construct=cname, n=n, hospitals=len(hs), dev_targets=n_dev,
                                     reliability=np.nan, reliability_pairwise=np.nan))
                continue
            rs, rp = [], []
            iu = np.triu_indices(len(hs), 1)
            for _ in range(draws):
                A = np.zeros((len(hs), 7))
                B = np.zeros((len(hs), 7))
                for i, h in enumerate(hs):
                    p = rng.permutation(by_h[h])
                    np.add.at(A[i], p[:n], 1)
                    np.add.at(B[i], p[n:2 * n], 1)
                a, b = s02.angles(A, pooled), s02.angles(B, pooled)
                rs.append(pd.Series(a).rank().corr(pd.Series(b).rank()))
                pa, pb = s02.pairwise_angles(A)[iu], s02.pairwise_angles(B)[iu]
                rp.append(pd.Series(pa).rank().corr(pd.Series(pb).rank()))
            r, r2 = float(np.mean(rs)), float(np.mean(rp))
            rel_rows.append(dict(condition=cond, construct=cname, n=n, hospitals=len(hs), dev_targets=n_dev,
                                 reliability=r, reliability_pairwise=r2))
            if n_star is None and r >= bar:
                n_star = n
            if n_pair is None and r2 >= bar and n_dev >= min_targets:
                n_pair = n
        n_ev = cnt.sum(axis=1)
        hs = [h for h, l in by_h.items() if len(l) >= floor]
        iu = np.triu_indices(len(hs), 1)
        rf = []
        for _ in range(draws if len(hs) >= 3 else 0):
            A = np.zeros((len(hs), 7))
            B = np.zeros((len(hs), 7))
            for i, h in enumerate(hs):
                p = rng_full.permutation(by_h[h])
                m = len(p) // 2
                np.add.at(A[i], p[:m], 1)
                np.add.at(B[i], p[m:2 * m], 1)
            pa, pb = s02.pairwise_angles(A)[iu], s02.pairwise_angles(B)[iu]
            rf.append(pd.Series(pa).rank().corr(pd.Series(pb).rank()))
        r_half = float(np.mean(rf)) if rf else np.nan
        r_full = 2 * r_half / (1 + r_half) if rf else np.nan
        full_dev = len(set(hs) & dev_all)
        full_pass = bool(rf) and r_full >= bar and full_dev >= min_targets
        row = dict(condition=cond, construct=cname, stays_with_evidence=int(keep.sum()),
                   share_D=e[f"{cond}_D"].mean(), share_H=e[f"{cond}_H"].mean(),
                   share_third=e[f"{cond}_{srcs[2]}"].mean(), n_star=n_star,
                   hospitals_with_profile=int((n_ev >= n_star).sum()) if n_star else 0,
                   gate1_pass=n_star is not None, n_star_pairwise=n_pair,
                   gate1_pairwise_pass=n_pair is not None, full_hospitals=len(hs), full_dev_targets=full_dev,
                   full_reliability_half=r_half, full_reliability=r_full, gate1_full_pass=full_pass)
        for task in TASKS:
            tg = dev.loc[dev.task == task, "target"]
            k = int((n_ev.reindex(tg).fillna(0) >= n_star).sum()) if n_star else 0
            row[f"{task}_dev_targets_with_profile"] = k
            row[f"{task}_gate2_pass"] = bool(n_star) and k >= min_targets
            k2 = int((n_ev.reindex(tg).fillna(0) >= n_pair).sum()) if n_pair else 0
            row[f"{task}_dev_targets_pairwise"] = k2
            row[f"{task}_gate2_pairwise_pass"] = bool(n_pair) and k2 >= min_targets
            k3 = int((n_ev.reindex(tg).fillna(0) >= floor).sum())
            row[f"{task}_dev_targets_full"] = k3
            row[f"{task}_gate2_full_pass"] = full_pass and k3 >= min_targets
        if cond in st["conditions"]["paper"] and cname == "DHM":
            row["paper_n100_dev_targets"] = {t: int((n_ev.reindex(dev.loc[dev.task == t, "target"]).fillna(0) >= 100).sum())
                                             for t in TASKS}
        gate_rows.append(row)
        print(f"{cond:28s} {cname:6s} n*={n_star} n*pair={n_pair} full r={r_full:.2f} ({len(hs)} hosp) "
              + "  ".join(f"{t[:2]} {row[f'{t}_dev_targets_full']}/41 {row[f'{t}_gate2_full_pass']}" for t in TASKS),
              flush=True)

    pd.DataFrame(rel_rows).round(4).to_csv(res / "condition_scan_gate1.csv", index=False)
    out = pd.DataFrame(gate_rows)
    out.round(4).to_csv(res / "condition_scan_gates.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rel_rows).pivot_table(index=["condition", "construct"], columns="n",
                                             values="reliability").round(2).to_string())
    print(pd.DataFrame(rel_rows).pivot_table(index=["condition", "construct"], columns="n",
                                             values="reliability_pairwise").round(2).to_string())


if __name__ == "__main__":
    main()
