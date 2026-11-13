# Site-level documentation variation and training-site selection (JAMIA 2026)

Code for "Site-Level EHR Documentation Variation Impacts Model Inference and Generalizability in Multi-Center
Research". It measures documentation angular separation (DAS) between eICU hospitals, relates it to between-hospital
outcome variation, and tests whether DAS can choose training hospitals for a new hospital's prediction model, in held-out
eICU hospitals and in MIMIC-IV. The rest of this repository holds the code of the earlier eICU analysis; the phenotype
rules it uses are shared (`../configs/phenotypes.yaml`, `../configs/phenotypes_specific_meds.yaml`).

## Data

eICU Collaborative Research Database v2.0, MIMIC-IV v2.2 and MIMIC-IV-Note v2.2 from PhysioNet (credentialed access).
No data or results are included here. Set the local paths in `configs/external.yaml` (`paths.eicu`, `paths.mimic`, the
note file, and `paths.rowlevel` for row-level intermediate files, which should stay outside any shared folder).
Scripts write only aggregate results to `results/` and figures to `output/`.

## Main scripts (run from this folder with `uv run --no-project --with ... python scripts/<name>.py`)

| Step | Script |
|---|---|
| Cohort, evidence flags, features | `01_build_flags.py`, `03_build_features.py`, `09_apache_features.py`, `46_flags_24h.py` |
| Hospitals and stays (Table 1), example hospitals (Figure 1) | `41_table1.py`, `42_venn_data.py` |
| Documentation variation and outcome models | `40_das_variation.py` (via `partA.sh`), `43_partA_effects.py`, `51_partA_readmission.py` |
| Training-site selection: split, development and held-out loops, pooling, MIMIC-IV | `38_iecv.py split / loop / pool / mimic --spec v6` (chain in `iecv_v6.sh`, MIMIC-IV in `mimic_v6.sh`) |
| DAS against the membership model, paired | `39_das_vs_membership.py v6_2026-11-11` |
| Selection ceiling | `49_ceiling_iecv.py` (via `ceiling.sh`) |
| Percentile of the reported (trim-mean) AUROC | `52_percentile_of_mean.py v6_2026-11-11` |
| Figures 1-5, Supplementary Figure S1 | `figures/make_fig1.py` ... `make_fig5.py`, `make_fig_s1.py` |

Every script states its inputs, outputs and settings in its docstring. `38_iecv.py --spec` selects the analysis
version; the paper uses `v6`. Other scripts and specs are earlier designs and sensitivity runs, kept for transparency.
`11_charlson_features.py` needs `MIMIC_CHARLSON_SQL` pointing to `concepts/comorbidity/charlson.sql` from the MIT-LCP
mimic-code repository.
