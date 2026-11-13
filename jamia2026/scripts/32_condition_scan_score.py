#!/usr/bin/env python
"""A_condition_scan, scoring: nearest 10 by each angle that passed the full-data gate 1 and gate 2
(configs/screen_settings.yaml, A_condition_scan with gate1_full, locked 2026-11-07). Development
targets only; tasks A3 (ICU stay > 7 days) and A6 (hospital stay > 14 days); k = 10.

Targets, pool, models, size control and the S/E split of each target as scripts/22_task_ceiling.py.
Profiles: hospitals with >= 100 evidence-positive stays (the paper's minimum, gate1_full floor).
Arms:
  new candidates  das_copd (DHM), das_DHMany_<diabetes|hypertension|chf|af>, das_DHMany_joined
                  (mean DHMany angle over the four), das_scan_joined (mean DHM angle over the paper's
                  four and COPD)
  references      das_<paper condition> (DHM) and das_joined, rebuilt on the same targets and draws
Contrasts: gain over random 10; paired minus das_chf (vs_chf); DHMany minus DHM, same condition
(das_DHMany_joined minus das_joined). Pass: mean gain > 0 and lower 95% target-bootstrap bound > 0.
New candidates go to results/search_ledger.csv, block A_condition_scan.

Outputs (aggregate): results/condition_scan_targets.csv, results/condition_scan_summary.csv.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with joblib --with requests python scripts/32_condition_scan_score.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m1_model import C_M1, design_columns  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s02 = _load("s02", "02_das_external.py")
s19 = _load("s19", "19_ceiling.py")
s31 = _load("s31", "31_condition_scan_gates.py")

PAPER = s19.CONDITIONS
FLAGS = [f"{c}_{s}" for c in PAPER for s in ("D", "H", "M")]
K, N_RANDOM, MIN_POOL, MIN_EVENTS = 10, 200, 100, 20
TASKS = ["A3_los_gt7d", "A6_hospital_los_gt14d"]
COPD = "chronic_pulmonary_disease"


def profiles(con, E, rowlevel, st, floor):
    """Lobe-share vectors per hospital (>= floor evidence-positive stays) for every scored angle."""
    e = pd.read_parquet(rowlevel / "eicu_flags.parquet")
    med = pd.read_parquet(rowlevel / "eicu_flags_scan_med.parquet").drop(columns="hospitalid")
    e = e.merge(med, on="patientunitstayid", validate="1:1")
    con.register("ecoh", e[["patientunitstayid"]])
    coded, hist = s31.charlson_DH(con, E)
    sid = e.patientunitstayid
    comps = st["conditions"]["charlson"][COPD]
    e[f"{COPD}_D"] = np.any([sid.map(coded[c]).fillna(0).to_numpy() > 0 for c in comps if c in coded], axis=0)
    e[f"{COPD}_H"] = np.any([sid.isin(hist.get(c, set())).to_numpy() for c in comps], axis=0)
    for c in PAPER:
        e[f"{c}_X"] = e[f"{c}_A"] | e[f"{c}_M"]
    spec = {f"das_{c}": (c, ("D", "H", "M")) for c in PAPER}
    spec["das_copd"] = (COPD, ("D", "H", "M"))
    spec |= {f"das_DHMany_{c}": (c, ("D", "H", "X")) for c in PAPER}
    V = {}
    for arm, (cond, srcs) in spec.items():
        cnt = s02.hospital_counts(e, s02.lobe_codes(e, cond, srcs), 7)
        cnt = cnt[cnt.sum(axis=1) >= floor]
        V[arm] = cnt.div(cnt.sum(axis=1), axis=0)
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E = cfg["paths"]["eicu"]
    st = yaml.safe_load(open(base / "configs" / "screen_settings.yaml"))["A_condition_scan"]
    assert st["status"] == "locked" and st["gate1_full"]["status"] == "locked"
    floor = cfg["das"]["min_records_per_hospital"]
    assert floor == 100
    rng = np.random.default_rng(st["seed"])

    con = duckdb.connect()
    V = profiles(con, E, rowlevel, st, floor)
    joins = {"das_joined": [f"das_{c}" for c in PAPER],
             "das_DHMany_joined": [f"das_DHMany_{c}" for c in PAPER],
             "das_scan_joined": [f"das_{c}" for c in PAPER] + ["das_copd"]}

    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_m2_features.parquet")[["patientunitstayid", "charlson"]],
                on="patientunitstayid", how="left")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_flags.parquet")[["patientunitstayid"] + FLAGS],
                on="patientunitstayid", how="left")
    e[FLAGS] = e[FLAGS].fillna(0).astype(float)
    pt = con.execute(f"""SELECT patientunitstayid, unitdischargeoffset / 1440.0 AS los_d,
                                (hospitaldischargeoffset - hospitaladmitoffset) / 1440.0 AS hosp_los_d
                         FROM '{E}/patient.parquet'""").df()
    e = e.merge(pt, on="patientunitstayid", how="left")
    m1 = design_columns()
    tasks = {"A3_los_gt7d": (e.los_d > 7).astype(int), "A6_hospital_los_gt14d": (e.hosp_los_d > 14).astype(int)}

    split = pd.read_csv(res / "target_split.csv")
    dev_all = split[split.half == "development"].hospitalid.to_numpy()
    X = e[m1 + ["charlson"] + FLAGS].to_numpy(float)
    hid = e.hospitalid.to_numpy()
    rows_of = {h: np.flatnonzero(hid == h).astype(np.int32) for h in np.unique(hid)}
    trows = []
    for task, yy in tasks.items():
        y = np.asarray(yy, int)
        g = pd.DataFrame({"h": hid, "y": y}).groupby("h").y.agg(["size", "sum"])
        pool_all = np.array(sorted(g[(g["size"] >= MIN_POOL) & (g["sum"] > 0) & (g["sum"] < g["size"])].index))
        dev = [t for t in dev_all if t in g.index and g.loc[t, "sum"] >= MIN_EVENTS]
        if args.limit:
            dev = dev[: args.limit]
        print(f"{task}: {len(dev)} development targets, pool {len(pool_all)}", flush=True)
        for t in dev:
            r_t = rows_of[t]
            d_t = y[r_t]
            S = np.concatenate([rng.permutation(r_t[d_t == v])[: (d_t == v).sum() // 2] for v in (0, 1)])
            Ev = np.setdiff1d(r_t, S)
            pool = pool_all[pool_all != t]
            sets, ang = {}, {}
            for arm, Va in V.items():
                if t in Va.index:
                    cand = np.array([h for h in pool if h in Va.index])
                    a = pd.Series(s02.angles(Va.loc[cand].to_numpy(float), Va.loc[t].to_numpy(float)), index=cand)
                    ang[arm] = a
                    sets[arm] = a.sort_values(kind="stable").index[:K].to_numpy()
            for j, parts in joins.items():
                have = [ang[p] for p in parts if p in ang]
                if have:
                    sets[j] = pd.concat(have, axis=1).mean(axis=1, skipna=True) \
                        .sort_values(kind="stable").index[:K].to_numpy()
            rand = [rng.choice(pool, K, replace=False) for _ in range(N_RANDOM)]
            jobs, n_target = s19.size_jobs(sets, rand, rows_of, rng)
            preds = Parallel(n_jobs=args.jobs, batch_size=8)(
                delayed(s19.fit_scores)(X, y, rows, X[Ev], C_M1) for _, _, rows in jobs)
            r = pd.DataFrame([dict(arm=a, auc_E=s19.auc(y[Ev], p)) for (a, _, _), p in zip(jobs, preds)]) \
                .groupby("arm").auc_E.mean()
            rnd = r[r.index.str.startswith("rand_")]
            row = dict(task=task, target=t, n_train=n_target, random_mean_E=rnd.mean(), random_sd_E=rnd.std())
            for a in sets:
                row[f"{a}_E"] = r.loc[a]
            trows.append(row)
        print(f"  {task} done", flush=True)
    tg = pd.DataFrame(trows)
    sfx = "" if not args.limit else "_test"
    tg.round(5).to_csv(res / f"condition_scan_targets{sfx}.csv", index=False)

    brng = np.random.default_rng(st["seed"] + 2)
    new = ["das_copd"] + [f"das_DHMany_{c}" for c in PAPER] + ["das_DHMany_joined", "das_scan_joined"]
    refs = [f"das_{c}" for c in PAPER] + ["das_joined"]
    pair = {f"das_DHMany_{c}": f"das_{c}" for c in PAPER} | {"das_DHMany_joined": "das_joined"}
    srows = []

    def add(task, s, arm, a, b):
        ok = s[a].notna() & s[b].notna()
        dif = (s[a] - s[b])[ok]
        est, lo, hi = s19.boot_mean(dif, brng)
        srows.append(dict(task=task, arm=arm, targets=int(ok.sum()), gain_vs_random=est, lo=lo, hi=hi,
                          share_targets_positive=(dif > 0).mean(), passes=bool(est > 0 and lo > 0)))

    for task, s in tg.groupby("task"):
        for a in refs + new:
            if f"{a}_E" in s:
                add(task, s, a, f"{a}_E", "random_mean_E")
        for a in new:
            if f"{a}_E" in s:
                add(task, s, f"{a}_minus_chf", f"{a}_E", "das_chf_E")
        for a, b in pair.items():
            if f"{a}_E" in s:
                add(task, s, f"{a}_minus_DHM", f"{a}_E", f"{b}_E")
    sm = pd.DataFrame(srows)
    sm.round(5).to_csv(res / f"condition_scan_summary{sfx}.csv", index=False)
    pd.set_option("display.width", 200)
    print(sm.round(4).to_string(index=False))
    if args.limit:
        return
    led = pd.read_csv(res / "search_ledger.csv")
    c = sm[sm.arm.isin(new)].copy()
    c["candidate"] = "A_" + c.task + "_" + c.arm + "_nearest10"
    c["block"], c["date"] = "A_condition_scan", "2026-11-07"
    led = pd.concat([led[~led.candidate.isin(c.candidate)], c[led.columns]], ignore_index=True)
    led.to_csv(res / "search_ledger.csv", index=False)


if __name__ == "__main__":
    main()
