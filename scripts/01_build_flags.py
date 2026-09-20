#!/usr/bin/env python
"""Build patient-level evidence flags per condition.

Outputs (per condition):
  data/processed/flags_<condition>.parquet

Usage:
  python scripts/01_build_flags.py --conditions diabetes hypertension
  python scripts/01_build_flags.py --conditions all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from ehrdoc.data import load_eicu_tables
from ehrdoc.phenotypes.definitions import PhenotypeRule, TextRule
from ehrdoc.phenotypes.build_patient_flags import build_condition_flags
from ehrdoc.utils.io import ensure_dir


def load_rules(phenotypes_yaml: str | Path) -> dict[str, PhenotypeRule]:
    with open(phenotypes_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    rules: dict[str, PhenotypeRule] = {}
    for name, spec in cfg.items():
        def _txt(key: str) -> TextRule:
            s = spec.get(key, {}) or {}
            return TextRule(
                include=s.get("include", []) or [],
                exclude=s.get("exclude", []) or [],
                icd_regex=s.get("icd_regex"),
            )

        rules[name] = PhenotypeRule(
            name=name,
            diagnosis=_txt("diagnosis"),
            past_history=_txt("past_history"),
            medication=_txt("medication"),
        )
    return rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--phenotypes", default="configs/phenotypes.yaml")
    ap.add_argument("--conditions", nargs="+", default=["all"],
                    help="Condition names in phenotypes.yaml, or 'all'.")
    args = ap.parse_args()

    # load configs
    with open(args.paths, "r") as f:
        paths_cfg = yaml.safe_load(f)
    out_dir = ensure_dir(paths_cfg["outputs"]["processed"])

    rules = load_rules(args.phenotypes)
    conditions = list(rules.keys()) if args.conditions == ["all"] else args.conditions

    # load eICU tables
    patient_df, diagnosis_df, past_history_df, medication_df = load_eicu_tables(args.paths)

    # minimal hospital join — include uniquepid and hospital characteristics
    keep_patient_cols = ["patientunitstayid", "hospitalid"]
    for optional_col in ["uniquepid", "numbedscategory", "teachingstatus", "region"]:
        if optional_col in patient_df.columns:
            keep_patient_cols.append(optional_col)
    patient_small = patient_df[keep_patient_cols].drop_duplicates()

    for cond in conditions:
        if cond not in rules:
            raise KeyError(f"Unknown condition '{cond}'. Available: {sorted(rules.keys())}")

        print(f"\n{'='*60}")
        print(f"Building flags: {cond.upper()}")
        print(f"{'='*60}")

        # ── Validate medication terms against actual drugName values ──
        rule = rules[cond]
        med_terms = rule.medication.include
        if med_terms and "drugname" in medication_df.columns:
            drug_names_lower = medication_df["drugname"].str.lower()
            print(f"  Medication term validation ({len(med_terms)} terms):")
            matched_terms = []
            unmatched_terms = []
            for term in med_terms:
                n_hits = drug_names_lower.str.contains(term.lower(), na=False).sum()
                if n_hits > 0:
                    matched_terms.append((term, n_hits))
                else:
                    unmatched_terms.append(term)
            print(f"    Matched: {len(matched_terms)}/{len(med_terms)} terms")
            if unmatched_terms:
                print(f"    Unmatched ({len(unmatched_terms)}): {', '.join(unmatched_terms)}")
            # Top 10 most frequent matches
            matched_terms.sort(key=lambda x: -x[1])
            print(f"    Top matches:")
            for term, n in matched_terms[:10]:
                print(f"      {term:30s} {n:>8,} rows")

        # ── Validate diagnosis terms ──
        dx_terms = rule.diagnosis.include
        dx_excl = rule.diagnosis.exclude
        if dx_terms and "diagnosisstring" in diagnosis_df.columns:
            dx_lower = diagnosis_df["diagnosisstring"].str.lower()
            n_dx_match = 0
            for term in dx_terms:
                n_dx_match += dx_lower.str.contains(term.lower(), na=False).sum()
            n_dx_excl = 0
            for term in dx_excl:
                n_dx_excl += dx_lower.str.contains(term.lower(), na=False).sum()
            print(f"  Diagnosis: {n_dx_match:,} rows match include terms, {n_dx_excl:,} match exclude terms")

        # ── Validate past history terms ──
        ph_terms = rule.past_history.include
        if ph_terms and "pasthistorypath" in past_history_df.columns:
            ph_lower = past_history_df["pasthistorypath"].str.lower()
            n_ph_match = 0
            for term in ph_terms:
                n_ph_match += ph_lower.str.contains(term.lower(), na=False).sum()
            print(f"  Past History: {n_ph_match:,} rows match include terms")

        flags = build_condition_flags(
            patient_df=patient_small,
            diagnosis_df=diagnosis_df,
            past_history_df=past_history_df,
            medication_df=medication_df,
            rule=rule,
        )
        # ensure hospital included
        flags = flags.merge(patient_small, on="patientunitstayid", how="left")

        out_path = out_dir / f"flags_{cond}.parquet"
        flags.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} (n={len(flags):,})")


if __name__ == "__main__":
    main()
