#!/usr/bin/env python
"""External DAS (stage 2): where does MIMIC-IV sit among eICU hospitals?

Steps:
  1. Split MIMIC-IV patients into a documentation-reference subset (25%) and a
     held-out evaluation subset, within anchor_year_group, without outcomes.
  2. Build lobe vectors per eICU hospital and for the MIMIC-IV reference.
  3. DAS from each eligible eICU hospital to MIMIC-IV; rank hospitals.
  4. Benchmark: compare MIMIC-IV's k-th nearest DAS with each eICU hospital's
     k-th nearest-neighbour DAS, and its distance to the eICU centroid with each
     hospital's distance to the centroid of the others.
  5. Bootstrap the MIMIC-IV reference patients and summarize nearest-k stability.
  6. Noise floor: split each eICU hospital's patients into random halves (20 times)
     and compare within-hospital with between-hospital angles, and the DAS-to-MIMIC
     ranking from one half with the ranking from the other.

Constructs (sources in lobe order; n sources give 2^n - 1 lobes):
  DAM         diagnosis, home medication, in-stay medication (7 lobes, stage 4 primary)
  DAM_spec    same with condition-specific medication classes
  DM          diagnosis and in-stay medication only (3 lobes)
  DM_spec     same with condition-specific medication classes
  DicdAM      eICU diagnosis from ICD codes (not diabetes)
  Constructs with past history (H: eICU pastHistory; MIMIC-IV notes, script 08),
  run only when mimic_history_flags.parquet exists:
  DHM         diagnosis, history, in-stay medication (the published construct, 7 lobes)
  DHA         diagnosis, history, home medication (7 lobes)
  DHAM        diagnosis, history, home and in-stay medication (15 lobes)
  DHM_strict  DHM with MIMIC-IV history matched on published terms only
  DHM_prior   DHM with MIMIC-IV history from earlier admissions' summaries
Reference variants: ref_2011_2019 (years with ED home medications) and ref_all_years.

Writes aggregate CSVs only to results/. With --history pointing at a *.partial
file, outputs are written with a .partial suffix.

Usage (from mimic_external/):
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with numpy \
      python scripts/02_das_external.py --config configs/external.yaml
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONDITIONS = ["diabetes", "hypertension", "chf", "af"]
CONSTRUCTS = {
    "DAM": ("D", "A", "M"),
    "DAM_spec": ("D", "Aspec", "Mspec"),
    "DM": ("D", "M"),
    "DM_spec": ("D", "Mspec"),
    "DicdAM": ("Dicd", "A", "M"),
    "DHM": ("D", "H", "M"),
    "DHA": ("D", "H", "A"),
    "DHAM": ("D", "H", "A", "M"),
    "DHM_strict": ("D", "Hstrict", "M"),
    "DHM_prior": ("D", "Hprior", "M"),
}
HISTORY_CONSTRUCTS = {"DHM", "DHA", "DHAM", "DHM_strict", "DHM_prior"}
BOTH_REFS = {"DAM", "DHM", "DHAM"}          # run on both reference variants
CAPTURE = {"DAM", "DM", "DHM", "DHA", "DHAM"}
SHORT = {"D": "D", "Dicd": "D", "H": "H", "Hstrict": "H", "Hprior": "H",
         "A": "A", "Aspec": "A", "M": "M", "Mspec": "M"}
N_SPLITS = 20


def lobe_codes(df: pd.DataFrame, cond: str, srcs: tuple) -> np.ndarray:
    """Lobe index per row in the manuscript order (all sources first, single sources last); -1 = no evidence.

    The binary code reads the sources left to right; lobe index = 2^n - 1 - code,
    which gives D∩X∩M, D∩X, D∩M, D only, X∩M, X only, M only for three sources.
    """
    code = np.zeros(len(df), dtype=int)
    for s in srcs:
        code = code * 2 + df[f"{cond}_{s}"].to_numpy(bool)
    return np.where(code > 0, (2 ** len(srcs) - 1) - code, -1)


def lobe_labels(srcs: tuple) -> list[str]:
    n, short = len(srcs), [SHORT[s] for s in srcs]
    out = []
    for i in range(2 ** n - 1):
        code = 2 ** n - 1 - i
        on = [short[j] for j in range(n) if (code >> (n - 1 - j)) & 1]
        out.append("∩".join(on) if len(on) > 1 else f"{on[0]} only")
    return out


def angles(vectors: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Angular separation in degrees between each row of vectors and ref."""
    num = vectors @ ref
    den = np.linalg.norm(vectors, axis=1) * np.linalg.norm(ref)
    cos = np.clip(np.divide(num, den, out=np.zeros_like(num), where=den > 0), -1, 1)
    return np.degrees(np.arccos(cos))


def pairwise_angles(v: np.ndarray) -> np.ndarray:
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(u @ u.T, -1, 1)))


def split_mimic(m: pd.DataFrame, frac: float, salt: str) -> pd.Series:
    """Reference membership by hashed subject_id, taken within each anchor_year_group."""
    h = m["subject_id"].map(lambda s: int(hashlib.md5(f"{salt}:{s}".encode()).hexdigest()[:12], 16))
    rank = h.groupby(m["anchor_year_group"]).rank(pct=True, method="first")
    return rank <= frac


def hospital_counts(e: pd.DataFrame, codes: np.ndarray, n_lobes: int) -> pd.DataFrame:
    keep = codes >= 0
    c = pd.crosstab(e.loc[keep, "hospitalid"], codes[keep])
    return c.reindex(columns=range(n_lobes), fill_value=0)


def noise_floor(e: pd.DataFrame, codes: np.ndarray, hids: np.ndarray, n_lobes: int,
                rvec: np.ndarray, rng: np.random.Generator, k: int = 10) -> dict:
    """Split-half check of how much of the between-hospital spread is sampling noise."""
    keep = (codes >= 0) & e.hospitalid.isin(hids).to_numpy()
    hidx = pd.Index(hids).get_indexer(e.hospitalid.to_numpy()[keep])
    lob = codes[keep]
    within, between, rho, overlap, sparse = [], [], [], [], []
    off = ~np.eye(len(hids), dtype=bool)
    for _ in range(N_SPLITS):
        half = rng.random(len(lob)) < 0.5
        A = np.zeros((len(hids), n_lobes))
        B = np.zeros((len(hids), n_lobes))
        np.add.at(A, (hidx[half], lob[half]), 1)
        np.add.at(B, (hidx[~half], lob[~half]), 1)
        sparse.append(np.mean(A < 5))
        A /= A.sum(axis=1, keepdims=True)
        B /= B.sum(axis=1, keepdims=True)
        UA = A / np.linalg.norm(A, axis=1, keepdims=True)
        UB = B / np.linalg.norm(B, axis=1, keepdims=True)
        X = np.degrees(np.arccos(np.clip(UA @ UB.T, -1, 1)))
        within.append(np.diag(X))
        between.append(X[off])
        dA, dB = angles(A, rvec), angles(B, rvec)
        rho.append(pd.Series(dA).rank().corr(pd.Series(dB).rank()))
        overlap.append(len(set(np.argsort(dA)[:k]) & set(np.argsort(dB)[:k])))
    w, b = np.concatenate(within), np.concatenate(between)
    return dict(n_lobes=n_lobes, within_half_median=np.median(w), between_half_median=np.median(b),
                between_to_within=np.median(b) / np.median(w),
                share_pairs_beyond_within95=np.mean(b > np.percentile(w, 95)),
                das_rank_split_half_rho=np.mean(rho), nearest10_split_half_overlap=np.mean(overlap),
                share_half_cells_below5=np.mean(sparse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/external.yaml")
    ap.add_argument("--history", default=None, help="MIMIC-IV history flags (default: rowlevel/mimic_history_flags.parquet)")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg_path.parent.parent
    rowlevel = Path(cfg["paths"]["rowlevel"])
    res = base / cfg["paths"]["results"]
    res.mkdir(parents=True, exist_ok=True)
    dcfg = cfg["das"]
    rng = np.random.default_rng(dcfg["seed"])
    nf_rng = np.random.default_rng(dcfg["seed"] + 2)
    min_n = dcfg["min_records_per_hospital"]
    ks = dcfg["k"]

    e = pd.read_parquet(rowlevel / "eicu_flags.parquet")
    m = pd.read_parquet(rowlevel / "mimic_flags.parquet")
    for cond in CONDITIONS:
        m[f"{cond}_Dicd"] = m[f"{cond}_D"]   # MIMIC-IV diagnoses are ICD codes already

    hist_path = Path(args.history) if args.history else rowlevel / "mimic_history_flags.parquet"
    tag = ".partial" if ".partial" in hist_path.name else ""
    constructs = dict(CONSTRUCTS)
    if hist_path.exists():
        h = pd.read_parquet(hist_path)
        hcols = [f"{c}_{s}" for c in CONDITIONS for s in ["H", "Hstrict", "Hprior"]]
        m = m.merge(h[["stay_id"] + hcols], on="stay_id", how="left")
        m[hcols] = m[hcols].fillna(False).astype(bool)
        for cond in CONDITIONS:   # eICU has one history source, pastHistory
            e[f"{cond}_Hstrict"] = e[f"{cond}_H"]
            e[f"{cond}_Hprior"] = e[f"{cond}_H"]
        print(f"History flags from {hist_path.name}")
    else:
        constructs = {k: v for k, v in constructs.items() if k not in HISTORY_CONSTRUCTS}
        print(f"No {hist_path.name}: history constructs skipped")

    m["reference"] = split_mimic(m, cfg["split"]["reference_fraction"], cfg["split"]["salt"])
    if not tag:
        m[["subject_id", "stay_id", "anchor_year_group", "reference"]].to_parquet(rowlevel / "mimic_split.parquet", index=False)
    years = cfg["split"]["reference_year_groups"]
    refs = {
        "ref_2011_2019": m[m.reference & m.anchor_year_group.isin(years)],
        "ref_all_years": m[m.reference],
    }
    split_tab = (m.groupby("anchor_year_group")["reference"].agg(["size", "sum"])
                 .rename(columns={"size": "stays", "sum": "reference"}).reset_index())
    split_tab.to_csv(res / f"mimic_split_by_year_group{tag}.csv", index=False)
    print(split_tab.to_string(index=False))

    capture, profiles, das_rows, summary, stab, freq, noise = [], [], [], [], [], [], []

    for cname, srcs in constructs.items():
        labels = lobe_labels(srcs)
        n_lobes = len(labels)
        for cond in CONDITIONS:
            if cname == "DicdAM" and cond == "diabetes":
                continue
            ecodes = lobe_codes(e, cond, srcs)
            counts = hospital_counts(e, ecodes, n_lobes)
            counts = counts[counts.sum(axis=1) >= min_n]
            hids = counts.index.to_numpy()
            V = counts.to_numpy(float)
            V = V / V.sum(axis=1, keepdims=True)

            # zero-source hospitals: a whole source empty among the condition's patients
            pos = e[ecodes >= 0]
            zero = pd.Series(False, index=hids)
            for s in srcs:
                share = pos.groupby("hospitalid")[f"{cond}_{s}"].mean().reindex(hids)
                zero |= (share == 0).to_numpy()
            samples = {"all_hospitals": np.ones(len(hids), bool), "excluding_zero_source": ~zero.to_numpy()}

            for rname, ref in refs.items():
                if cname not in BOTH_REFS and rname != "ref_2011_2019":
                    continue
                rcodes = lobe_codes(ref, cond, srcs)
                rcount = np.bincount(rcodes[rcodes >= 0], minlength=n_lobes).astype(float)
                rvec = rcount / rcount.sum()
                d = angles(V, rvec)

                if cname in CAPTURE:
                    sh = {f"share_{s}": ref.loc[rcodes >= 0, f"{cond}_{s}"].mean() for s in srcs}
                    capture.append(dict(construct=cname, reference=rname, dataset="MIMIC-IV", condition=cond,
                                        stays=len(ref), evidence_positive=int(rcount.sum()), **sh,
                                        **{f"lobe_{l}": v for l, v in zip(labels, rvec)}))
                    if rname == "ref_2011_2019":
                        epos = e[ecodes >= 0]
                        sh = {f"share_{s}": epos[f"{cond}_{s}"].mean() for s in srcs}
                        pooled = np.bincount(ecodes[ecodes >= 0], minlength=n_lobes) / (ecodes >= 0).sum()
                        capture.append(dict(construct=cname, reference="pooled", dataset="eICU", condition=cond,
                                            stays=len(e), evidence_positive=int((ecodes >= 0).sum()),
                                            eligible_hospitals=len(hids), zero_source_hospitals=int(zero.sum()), **sh,
                                            **{f"lobe_{l}": v for l, v in zip(labels, pooled)}))

                for i, hid in enumerate(hids):
                    das_rows.append(dict(construct=cname, reference=rname, condition=cond, hospitalid=int(hid),
                                         n_evidence=int(counts.iloc[i].sum()), das=d[i], zero_source=bool(zero.iloc[i])))

                for sname, keep in samples.items():
                    Vs, ds, hs = V[keep], d[keep], hids[keep]
                    if len(hs) < max(ks) + 1:
                        continue
                    order = np.argsort(ds)
                    P = pairwise_angles(Vs)
                    np.fill_diagonal(P, np.inf)
                    Psort = np.sort(P, axis=1)
                    loo = np.array([angles(Vs[[j]], np.delete(Vs, j, 0).mean(axis=0))[0] for j in range(len(hs))])
                    m_cent = angles(Vs.mean(axis=0, keepdims=True), rvec)[0]
                    row = dict(construct=cname, reference=rname, condition=cond, sample=sname,
                               n_hospitals=len(hs), das_min=ds.min(), das_median=np.median(ds), das_max=ds.max(),
                               nearest_hospital=int(hs[order[0]]), farthest_hospital=int(hs[order[-1]]),
                               eicu_pairwise_median=np.median(P[np.isfinite(P)]),
                               mimic_to_centroid=m_cent, eicu_to_centroid_median=np.median(loo),
                               mimic_centroid_percentile=100 * np.mean(loo < m_cent))
                    for k in ks:
                        kth_e = Psort[:, k - 1]
                        row[f"mimic_kth_nearest_k{k}"] = ds[order[k - 1]]
                        row[f"eicu_kth_nearest_median_k{k}"] = np.median(kth_e)
                        row[f"eicu_kth_nearest_max_k{k}"] = kth_e.max()
                        row[f"mimic_kth_percentile_k{k}"] = 100 * np.mean(kth_e < ds[order[k - 1]])
                    summary.append(row)
                    if cname == "DAM" or sname == "all_hospitals":
                        top = Vs[order[:10]].mean(axis=0)
                        profiles.append(dict(construct=cname, reference=rname, condition=cond, sample=sname,
                                             **{f"mimic_{l}": v for l, v in zip(labels, rvec)},
                                             **{f"nearest10_{l}": v for l, v in zip(labels, top)},
                                             **{f"eicu_median_{l}": v for l, v in zip(labels, np.median(Vs, axis=0))}))

                    if rname == "ref_2011_2019":
                        noise.append(dict(construct=cname, condition=cond, sample=sname, n_hospitals=len(hs),
                                          **noise_floor(e, ecodes, hs, n_lobes, rvec, nf_rng)))

                    # bootstrap the MIMIC-IV reference patients (multinomial over lobes incl. no evidence)
                    if rname != "ref_2011_2019" and cname not in BOTH_REFS:
                        continue
                    p_all = np.append(rcount, len(ref) - rcount.sum()) / len(ref)
                    draws = rng.multinomial(len(ref), p_all, size=dcfg["n_bootstrap"])[:, :n_lobes].astype(float)
                    draws /= draws.sum(axis=1, keepdims=True)
                    U = Vs / np.linalg.norm(Vs, axis=1, keepdims=True)
                    B = np.degrees(np.arccos(np.clip((draws / np.linalg.norm(draws, axis=1, keepdims=True)) @ U.T, -1, 1)))
                    branks = np.argsort(B, axis=1)
                    for k in ks:
                        point = set(order[:k])
                        overlap = np.array([len(point & set(r[:k])) for r in branks])
                        stab.append(dict(construct=cname, reference=rname, condition=cond, sample=sname, k=k,
                                         mean_overlap=overlap.mean(), min_overlap=int(overlap.min()),
                                         mean_jaccard=np.mean(overlap / (2 * k - overlap)),
                                         share_identical=np.mean(overlap == k),
                                         gap_k_to_k1=ds[order[k]] - ds[order[k - 1]]))
                    if cname in BOTH_REFS:
                        sel10 = np.zeros(len(hs))
                        for r in branks:
                            sel10[r[:10]] += 1
                        lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
                        for j in order[:20]:
                            freq.append(dict(construct=cname, reference=rname, condition=cond, sample=sname,
                                             hospitalid=int(hs[j]), rank=int(np.where(order == j)[0][0]) + 1,
                                             das=ds[j], das_lo=lo[j], das_hi=hi[j],
                                             selected_in_nearest10=sel10[j] / dcfg["n_bootstrap"]))

    # how much does swapping pastHistory for home medication move the eICU landscape?
    change = []
    for cond in CONDITIONS:
        vecs = {}
        for cname, srcs in {"DHM": ("D", "H", "M"), "DAM": ("D", "A", "M")}.items():
            c = hospital_counts(e, lobe_codes(e, cond, srcs), 7)
            c = c[c.sum(axis=1) >= min_n]
            vecs[cname] = c.div(c.sum(axis=1), axis=0)
        common = vecs["DHM"].index.intersection(vecs["DAM"].index)
        a = pairwise_angles(vecs["DHM"].loc[common].to_numpy())
        b = pairwise_angles(vecs["DAM"].loc[common].to_numpy())
        iu = np.triu_indices(len(common), 1)
        change.append(dict(condition=cond, hospitals=len(common),
                           pearson_pairwise_das=np.corrcoef(a[iu], b[iu])[0, 1],
                           spearman_pairwise_das=pd.Series(a[iu]).rank().corr(pd.Series(b[iu]).rank()),
                           median_pairwise_DHM=np.median(a[iu]), median_pairwise_DAM=np.median(b[iu])))

    out = {"source_capture": capture, "lobe_profiles": profiles, "das_to_mimic": das_rows,
           "das_summary": summary, "bootstrap_stability": stab, "bootstrap_selection_freq": freq,
           "construct_change_eicu": change, "noise_floor": noise}
    for name, rows in out.items():
        pd.DataFrame(rows).round(4).to_csv(res / f"{name}{tag}.csv", index=False)
        print(f"wrote results/{name}{tag}.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
