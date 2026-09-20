#!/usr/bin/env python
"""Collect the numbers the manuscript cites into one Markdown file.

Reads the frozen run under artifacts/ and writes artifacts/results_numbers.md
so every number in the paper can be traced to RUN_INFO.json.

Usage:
  python scripts/19_collect_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

COND = ["diabetes", "hypertension", "chf", "af"]
NICE = {"diabetes": "Diabetes", "hypertension": "Hypertension", "chf": "CHF", "af": "AF"}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    cfg = yaml.safe_load(open("configs/paths.yaml"))
    art = Path(cfg["outputs"]["artifacts"])
    out = []
    info = json.loads((art / "RUN_INFO.json").read_text()) if (art / "RUN_INFO.json").exists() else {}
    out.append(f"# Results numbers (run started {info.get('run_started')}, finished {info.get('run_finished')})\n")

    # ── Cohort flow ──
    p = art / "general" / "cohort_flow.csv"
    if p.exists():
        cf = pd.read_csv(p)
        out.append("## Cohort flow\n")
        out.append(md_table(cf[["condition", "stage", "n_stays", "n_patients", "n_hospitals"]]))
        out.append("")

    # ── Hospitals per condition, zero-source ──
    p = art / "sensitivity_zero_source" / "zero_source_hospitals.csv"
    if p.exists():
        out.append("## Hospitals with an empty evidence source\n")
        out.append(md_table(pd.read_csv(p).drop(columns=["excluded_hospitals"])))
        out.append("")

    # ── Concordance ──
    rho = pd.read_csv(art / "concordance" / "concordance_rho.csv", index_col=0)
    mp = pd.read_csv(art / "concordance" / "concordance_mantel_p.csv", index_col=0)
    nh = pd.read_csv(art / "concordance" / "concordance_n_hospitals.csv", index_col=0)
    rows = []
    for i, a in enumerate(rho.index):
        for b in rho.columns[i + 1:]:
            rows.append({"pair": f"{NICE[a]} vs {NICE[b]}", "n_hospitals": int(nh.loc[a, b]),
                         "spearman_rho": rho.loc[a, b], "mantel_p": mp.loc[a, b]})
    out.append("## Cross-condition concordance (all hospitals)\n")
    out.append(md_table(pd.DataFrame(rows)))
    out.append("")
    p = art / "sensitivity_zero_source" / "concordance_zero_source.csv"
    if p.exists():
        cz = pd.read_csv(p)
        piv = cz.pivot_table(index=["condition_1", "condition_2"], columns="sample",
                             values=["spearman_rho", "n_hospitals"], aggfunc="first").reset_index()
        piv.columns = ["_".join([c for c in col if c]) for col in piv.columns]
        out.append("## Concordance with vs without zero-source hospitals\n")
        out.append(md_table(piv))
        out.append("")
    p = art / "sensitivity" / "threshold_concordance.csv"
    if p.exists():
        tc = pd.read_csv(p)
        s = tc.groupby("threshold")["spearman_rho"].agg(["min", "mean", "max"]).reset_index()
        out.append("## Concordance by minimum-patient threshold (range over 6 pairs)\n")
        out.append(md_table(s))
        out.append("")

    # ── Hospital characteristics: PERMANOVA ──
    p = art / "general" / "all_characteristic_tests.csv"
    if p.exists():
        ct = pd.read_csv(p)
        out.append("## PERMANOVA: hospital descriptors vs DAS structure (all hospitals)\n")
        out.append(md_table(ct[["condition", "characteristic", "n_hospitals", "n_groups", "permanova_R2", "permanova_p", "kw_p_value"]]))
        out.append("")
    p = art / "sensitivity_zero_source" / "permanova_zero_source.csv"
    if p.exists():
        pz = pd.read_csv(p)
        piv = pz.pivot_table(index=["condition", "characteristic"], columns="sample",
                             values=["permanova_R2", "permanova_p"], aggfunc="first").reset_index()
        piv.columns = ["_".join([c for c in col if c]) for col in piv.columns]
        out.append("## PERMANOVA with vs without zero-source hospitals\n")
        out.append(md_table(piv))
        out.append("")

    # ── Regression ──
    for c in COND:
        p = art / c / "tables" / "model_comparison.csv"
        if not p.exists():
            continue
        mc = pd.read_csv(p)
        si = json.loads((art / c / "tables" / "analytic_sample_info.json").read_text())
        out.append(f"## Mortality models: {NICE[c]} (n={si['n_analytic']:,} stays, {si['hospitals_analytic']} hospitals)\n")
        cols = [x for x in ["model", "n_params", "AIC", "BIC", "AUC", "Brier", "RI_sd", "APACHE_OR",
                            "RaceEth_Caucasian_vs_AA_OR", "Transfer_vs_admit_OR", "LRT"] if x in mc.columns]
        out.append(md_table(mc[cols], "{:.4g}"))
        out.append("")
        p = art / c / "tables" / "variance_decomposition.csv"
        if p.exists():
            vd = pd.read_csv(p)
            vd = vd[vd["approach"] == "glmm_random_intercept"]
            out.append(md_table(vd[["model", "re_sd", "re_var", "ICC_latent", "pct_between_hospital_var_explained"]]))
            out.append("")
    p = art / "sensitivity_zero_source" / "variance_zero_source.csv"
    if p.exists():
        out.append("## Random-intercept variance explained by DAS, excluding zero-source hospitals\n")
        out.append(md_table(pd.read_csv(p)))
        out.append("")
    p = art / "sensitivity_zero_source" / "lrt_zero_source.csv"
    if p.exists():
        out.append("## LRTs excluding zero-source hospitals\n")
        out.append(md_table(pd.read_csv(p)[["condition", "description", "LR_statistic", "df", "p_value"]], "{:.3g}"))
        out.append("")

    # ── Reference-hospital sensitivity ──
    p = art / "sensitivity_reference" / "variance_reference.csv"
    if p.exists():
        vr = pd.read_csv(p)
        piv = vr.pivot(index="condition", columns="scalar", values="pct_between_hospital_var_explained").reset_index()
        out.append("## Reference-hospital sensitivity: % between-hospital variance explained by DAS, by scalar\n")
        out.append(md_table(piv, "{:.1f}"))
        out.append("")
        sc = pd.read_csv(art / "sensitivity_reference" / "scalar_correlations.csv")
        out.append("## Spearman correlations among DAS scalars\n")
        out.append(md_table(sc[["condition", "scalar_a", "scalar_b", "spearman_rho", "n"]]))
        out.append("")
        lr = pd.read_csv(art / "sensitivity_reference" / "lrt_reference.csv")
        lr = lr[lr["description"].str.contains("AS continuous add to hospital chars")]
        out.append("## LRT: DAS beyond hospital characteristics, by scalar\n")
        out.append(md_table(lr[["condition", "scalar", "LR_statistic", "df", "p_value"]], "{:.3g}"))
        out.append("")
        cc = pd.read_csv(art / "sensitivity_reference" / "capture_corr_reference.csv")
        piv = cc.pivot_table(index=["condition", "phenotype"], columns="scalar", values="spearman_rho").reset_index()
        out.append("## Capture rate vs DAS scalar (Spearman rho), by scalar\n")
        out.append(md_table(piv))
        out.append("")

    # ── APACHE completion vs DAS ──
    p = art / "general" / "apache_vs_as_correlations.csv"
    if p.exists():
        out.append("## APACHE IVa completion rate vs DAS (Spearman, hospital level)\n")
        out.append(md_table(pd.read_csv(p)))
        out.append("")

    # ── Transportability ──
    rows = []
    for c in COND:
        p = art / c / "tables" / "transportability_summary.csv"
        if not p.exists():
            continue
        ts = pd.read_csv(p)
        ts = ts[ts["size_group"] == "large"].copy()
        ts["condition"] = NICE[c]
        ts["win_pct"] = (100 * ts["nearest_wins"] / ts["n"]).round(0)
        rows.append(ts[["condition", "k", "metric", "n", "mean_nearest", "mean_farthest", "mean_delta", "win_pct", "wilcoxon_p"]])
    if rows:
        out.append("## LOHO: k nearest vs k farthest (all hospitals; delta > 0 favours nearest)\n")
        out.append(md_table(pd.concat(rows), "{:.4g}"))
        out.append("")
    p = art / "sensitivity_zero_source" / "transportability_zero_source_summary.csv"
    if p.exists():
        tz = pd.read_csv(p)
        out.append("## LOHO with vs without zero-source hospitals\n")
        out.append(md_table(tz[["condition", "sample", "k", "metric", "n", "mean_nearest", "mean_farthest", "mean_delta", "win_pct", "wilcoxon_p"]], "{:.4g}"))
        out.append("")

    # ── Phenotype capture ──
    rows = []
    for c in COND:
        p = art / c / "tables" / "phenotype_fragility_corr.csv"
        if p.exists():
            fc = pd.read_csv(p); fc.insert(0, "condition", NICE[c]); rows.append(fc)
    if rows:
        out.append("## Phenotype capture rate by rule (per-hospital range) and Spearman rho vs DAS\n")
        out.append(md_table(pd.concat(rows)[["condition", "phenotype", "mean_capture", "min_capture", "max_capture", "spearman_rho_vs_DAS", "p_value", "n_hospitals"]]))
        out.append("")

    # ── Transfer decomposition ──
    rows = []
    for c in COND:
        p = art / c / "tables" / "transfer_correlations.csv"
        if p.exists():
            t = pd.read_csv(p); t.insert(0, "condition", NICE[c]); rows.append(t)
    if rows:
        out.append("## Transfer rate correlations (hospital level)\n")
        out.append(md_table(pd.concat(rows)))
        out.append("")

    (art / "results_numbers.md").write_text("\n".join(out))
    print(f"Wrote {art / 'results_numbers.md'}")


if __name__ == "__main__":
    main()
