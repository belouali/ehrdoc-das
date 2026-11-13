#!/usr/bin/env python
"""Part A follow-up: the C1 ceiling and DAS candidates for the tasks that passed the gate.

Settings: configs/screen_settings.yaml (A_task_screen, then_for_passing_tasks) and
configs/ceiling_devconfirm_settings.yaml (C1, D1), approved 2026-11-06 before computing.
Development targets only; k = 10.

Tasks and features as scripts/21_task_gate.py (A3 stay > 7 days, A4 ICU readmission,
A5 comorbidity-only death). Each development target needs >= 20 events for the task.
Pool: every other hospital with >= 100 stays in the task cohort. Arms as script 19, plus
das_joined (mean DHM angle over the conditions both hospitals have). The candidates
(das_<condition>, das_joined, casemix; 6 per task, 18 in all) are added to
results/search_ledger.csv.

Outputs (aggregate): results/task_ceiling_targets.csv, results/task_ceiling_summary.csv.

A_task_screen_ext (approved 2026-11-07): --tasks A6_hospital_los_gt14d A7_discharge_facility
--tag _ext. Run first with --ceiling-only (ceiling arms and random sets only, no DAS or
case-mix set is built); the candidates run only for tasks whose ranking ceiling is >= 0.010,
and are logged in the ledger under block A_task_screen_ext.

A_construct_condition_ext part 1 (approved 2026-11-07): --tasks A3_los_gt7d A6_hospital_los_gt14d
--tag _construct --constructs DAM DHAM. Adds nearest 10 by each construct per condition and joined
(das_<construct>_<condition>), keeps the DHM arms for the paired contrast against the same condition,
drops case mix, seed 20261120. Ledger block A_construct_condition_ext.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn \
      --with joblib python scripts/22_task_ceiling.py --config configs/external.yaml
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
s07 = _load("s07", "07_casemix_nearest.py")
s19 = _load("s19", "19_ceiling.py")

CONDITIONS = s19.CONDITIONS
FLAGS = [f"{c}_{s}" for c in CONDITIONS for s in ("D", "H", "M")]
K, N_RANDOM, MIN_POOL, MIN_EVENTS = 10, 200, 100, 20
SEED = 20261110
FACILITY = ["Skilled Nursing Facility", "Nursing Home", "Rehabilitation"]
SOURCES = {"DHM": ("D", "H", "M"), "DAM": ("D", "A", "M"), "DHAM": ("D", "H", "A", "M")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tasks", nargs="+", default=["A3_los_gt7d", "A4_icu_readmission", "A5_comorbidity_only"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--ceiling-only", action="store_true")
    ap.add_argument("--constructs", nargs="+", default=[], choices=["DAM", "DHAM"])
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    E = cfg["paths"]["eicu"]
    rng = np.random.default_rng(20261120 if args.constructs else SEED if not args.tag else 20261116)

    e = pd.read_parquet(rowlevel / "eicu_m1_features.parquet")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_m2_features.parquet")[["patientunitstayid", "charlson"]],
                on="patientunitstayid", how="left")
    e = e.merge(pd.read_parquet(rowlevel / "eicu_flags.parquet")[["patientunitstayid"] + FLAGS],
                on="patientunitstayid", how="left")
    e[FLAGS] = e[FLAGS].fillna(0).astype(float)
    pt = duckdb.execute(f"""
        SELECT a.patientunitstayid, a.unitdischargeoffset / 1440.0 AS los_d, a.unitdischargestatus,
               (a.hospitaldischargeoffset - a.hospitaladmitoffset) / 1440.0 AS hosp_los_d,
               a.hospitaldischargestatus, a.hospitaldischargelocation,
               EXISTS (SELECT 1 FROM '{E}/patient.parquet' b
                       WHERE b.patienthealthsystemstayid = a.patienthealthsystemstayid
                         AND b.unitvisitnumber > a.unitvisitnumber) AS readmit
        FROM '{E}/patient.parquet' a""").df()
    e = e.merge(pt, on="patientunitstayid", how="left")
    m1 = design_columns()
    age_sex = ["agenum"] + [c for c in m1 if c.startswith("gender=")]
    alive = e[e.unitdischargestatus == "Alive"].reset_index(drop=True)
    tasks = {
        "A3_los_gt7d": (e, (e.los_d > 7).astype(int), m1),
        "A4_icu_readmission": (alive, alive.readmit.astype(int), m1 + ["los_d"]),
        "A5_comorbidity_only": (e, e.death.astype(int), age_sex),
        "A6_hospital_los_gt14d": (e, (e.hosp_los_d > 14).astype(int), m1),
    }
    surv = e[(e.hospitaldischargestatus == "Alive")
             & e.hospitaldischargelocation.isin(FACILITY + ["Home"])].reset_index(drop=True)
    tasks["A7_discharge_facility"] = (surv, surv.hospitaldischargelocation.isin(FACILITY).astype(int), m1)
    tasks = {k: tasks[k] for k in args.tasks}

    split = pd.read_csv(res / "target_split.csv")
    dev_all = split[split.half == "development"].hospitalid.to_numpy()
    flags = pd.read_parquet(rowlevel / "eicu_flags.parquet")
    VC = {}
    for cname in ["DHM"] + args.constructs:
        srcs = SOURCES[cname]
        VC[cname] = {}
        for cond in CONDITIONS:
            cnt = s02.hospital_counts(flags, s02.lobe_codes(flags, cond, srcs), 2 ** len(srcs) - 1)
            cnt = cnt[cnt.sum(axis=1) >= cfg["das"]["min_records_per_hospital"]]
            VC[cname][cond] = cnt.div(cnt.sum(axis=1), axis=0)
    no_a = sorted(set(flags.hospitalid) - set(flags.loc[flags.any_A.astype(bool), "hospitalid"]))
    if args.constructs:
        print(f"{len(no_a)} eICU hospitals have no home-medication record (their DAM/DHAM vectors lack A lobes)")
    feat = pd.read_parquet(rowlevel / "eicu_features.parquet")
    feat = feat[feat.death.notna()]
    eu = duckdb.execute(f"SELECT patientunitstayid, unittype FROM '{E}/patient.parquet'").df()
    feat = s07.add_units(feat.reset_index(drop=True), feat[["patientunitstayid"]].merge(eu, how="left").unittype,
                         s07.EICU_UNITS)
    hmeans = feat.groupby("hospitalid")[s07.CASEMIX].mean()
    Z = (hmeans - hmeans.mean()) / hmeans.std()

    trows = []
    for task, (d, yy, basecols) in tasks.items():
        X = d[basecols + ["charlson"] + FLAGS].to_numpy(float)
        y = np.asarray(yy, int)
        hid = d.hospitalid.to_numpy()
        rows_of = {h: np.flatnonzero(hid == h).astype(np.int32) for h in np.unique(hid)}
        g = pd.DataFrame({"h": hid, "y": y}).groupby("h").y.agg(["size", "sum"])
        pool_all = np.array(sorted(g[(g["size"] >= MIN_POOL) & (g["sum"] > 0) & (g["sum"] < g["size"])].index))
        dev = [t for t in dev_all if t in g.index and g.loc[t, "sum"] >= MIN_EVENTS]
        if args.limit:
            dev = dev[: args.limit]
        print(f"{task}: {len(dev)} development targets, pool {len(pool_all)}", flush=True)
        single = Parallel(n_jobs=args.jobs)(delayed(s19.fit_scores)(X, y, rows_of[h], X, s19.SINGLE_C)
                                            for h in pool_all)
        single = dict(zip(pool_all, single))
        for t in dev:
            r_t = rows_of[t]
            d_t = y[r_t]
            S = np.concatenate([rng.permutation(r_t[d_t == v])[: (d_t == v).sum() // 2] for v in (0, 1)])
            Ev = np.setdiff1d(r_t, S)
            pool = pool_all[pool_all != t]
            s_auc = pd.Series({h: s19.auc(y[S], single[h][S]) for h in pool}).sort_values(ascending=False,
                                                                                         kind="stable")
            sets = {"rank_best": s_auc.index[:K].to_numpy(), "rank_worst": s_auc.index[-K:].to_numpy()}
            for cname, V in VC.items():
                pre = "das_" if cname == "DHM" else f"das_{cname}_"
                ang = {}
                for cond in ([] if args.ceiling_only else CONDITIONS):
                    if t in V[cond].index:
                        cand = np.array([h for h in pool if h in V[cond].index])
                        a = pd.Series(s02.angles(V[cond].loc[cand].to_numpy(float),
                                                 V[cond].loc[t].to_numpy(float)), index=cand)
                        ang[cond] = a
                        sets[f"{pre}{cond}"] = a.sort_values(kind="stable").index[:K].to_numpy()
                if ang:
                    sets[f"{pre}joined"] = pd.concat(ang.values(), axis=1).mean(axis=1, skipna=True) \
                        .sort_values(kind="stable").index[:K].to_numpy()
            if t in Z.index and not args.ceiling_only and not args.constructs:
                cm = np.sqrt(((Z.reindex(pool) - Z.loc[t]) ** 2).mean(axis=1, skipna=True)).dropna()
                sets["casemix"] = cm.sort_values(kind="stable").index[:K].to_numpy()
            rand = [rng.choice(pool, K, replace=False) for _ in range(N_RANDOM)]
            jobs, n_target = s19.size_jobs(sets, rand, rows_of, rng)
            test = np.concatenate([S, Ev])
            preds = Parallel(n_jobs=args.jobs, batch_size=8)(
                delayed(s19.fit_scores)(X, y, rows, X[test], C_M1) for _, _, rows in jobs)
            ns = len(S)
            r = pd.DataFrame([dict(arm=a, auc_S=s19.auc(y[S], p[:ns]), auc_E=s19.auc(y[Ev], p[ns:]))
                              for (a, _, _), p in zip(jobs, preds)]).groupby("arm")[["auc_S", "auc_E"]].mean()
            rnd = r[r.index.str.startswith("rand_")]
            row = dict(task=task, target=t, n_train=n_target, target_has_A=t not in no_a,
                       random_mean_E=rnd.auc_E.mean(),
                       random_sd_E=rnd.auc_E.std(), set_best_E=rnd.auc_E.loc[rnd.auc_S.idxmax()],
                       set_worst_E=rnd.auc_E.loc[rnd.auc_S.idxmin()])
            for a in sets:
                row[f"{a}_E"] = r.loc[a, "auc_E"]
            trows.append(row)
        print(f"  {task} done", flush=True)
    tg = pd.DataFrame(trows)
    tag = args.tag + ("_ceiling" if args.ceiling_only else "") + ("" if not args.limit else "_test")
    tg.round(5).to_csv(res / f"task_ceiling_targets{tag}.csv", index=False)

    brng = np.random.default_rng(SEED + 1)
    new_arms = [f"das_{cn}_{c}" for cn in args.constructs for c in CONDITIONS + ["joined"]]
    arms = ["rank_best", "set_best", "rank_worst", "set_worst", "casemix", "das_joined"] + \
           [f"das_{c}" for c in CONDITIONS] + new_arms
    srows = []
    for task, s in tg.groupby("task"):
        ceil = s.rank_best_E - s.random_mean_E
        for a in new_arms:  # paired contrast: construct minus the paper's DHM, same condition, same targets
            ref = "das_" + a.split("_", 2)[2]
            if f"{a}_E" not in s or f"{ref}_E" not in s:
                continue
            for subset, m in (("all", s[f"{a}_E"].notna() & s[f"{ref}_E"].notna()),
                              ("target_has_A", s[f"{a}_E"].notna() & s[f"{ref}_E"].notna() & s.target_has_A)):
                dif = (s[f"{a}_E"] - s[f"{ref}_E"])[m]
                est, lo, hi = s19.boot_mean(dif, brng)
                srows.append(dict(task=task, arm=f"{a}_minus_DHM" + ("" if subset == "all" else "_has_A"),
                                  targets=int(m.sum()), gain_vs_random=est, lo=lo, hi=hi,
                                  share_targets_positive=(dif > 0).mean()))
        for a in arms:
            if f"{a}_E" not in s:
                continue
            ok = s[f"{a}_E"].notna()
            dif = (s[f"{a}_E"] - s.random_mean_E)[ok]
            est, lo, hi = s19.boot_mean(dif, brng)
            srows.append(dict(task=task, arm=a, targets=int(ok.sum()), gain_vs_random=est, lo=lo, hi=hi,
                              share_targets_positive=(dif > 0).mean(),
                              capture_of_rank_ceiling=est / ceil[ok].mean() if a != "rank_best" else 1.0))
        srows.append(dict(task=task, arm="random_sd_median", targets=len(s), gain_vs_random=s.random_sd_E.median()))
        srows.append(dict(task=task, arm="random_mean_median", targets=len(s),
                          gain_vs_random=s.random_mean_E.median()))
    sm = pd.DataFrame(srows)
    sm.round(5).to_csv(res / f"task_ceiling_summary{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(sm.round(4).to_string(index=False))
    if args.limit or args.ceiling_only:
        return
    led = pd.read_csv(res / "search_ledger.csv")
    c = sm[sm.arm.isin(new_arms if args.constructs else
                       ["casemix", "das_joined"] + [f"das_{x}" for x in CONDITIONS])].copy()
    c["candidate"] = "A_" + c.task + "_" + c.arm + "_nearest10"
    c["block"], c["date"] = ("A_construct_condition_ext", "2026-11-07") if args.constructs else \
        ("A_task_screen", "2026-11-06") if not args.tag else ("A_task_screen_ext", "2026-11-07")
    led = pd.concat([led[~led.candidate.isin(c.candidate)], c[led.columns]], ignore_index=True)
    led.to_csv(res / "search_ledger.csv", index=False)


if __name__ == "__main__":
    main()
