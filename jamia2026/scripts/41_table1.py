#!/usr/bin/env python
"""Table 1 of the paper: the 141 hospitals and their ICU stays, overall and by medication recording.

Aggregate output only: results/paper_table1.csv (rows = characteristics, columns = groups).
Usage (from mimic_external/): bash scripts/partA.sh is for 40; run this with scripts/iecv.sh's environment:
  uv run --no-project --with duckdb --with pandas --with pyarrow --with pyyaml --with scikit-learn --with joblib \
      --with statsmodels --with scipy python scripts/41_table1.py
"""

import importlib.util
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("s38", HERE / "38_iecv.py")
s38 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s38)


def main():
    cfg, _, res, rowlevel = s38.setup("configs/external.yaml")
    s38.TASKS = s38.TASKS + ["icu_readmit", "disch_facility"]  # ICU readmission row (2026-11-12)
    e, X, XM, Y, rows_of, V, no_med = s38.load_eicu(cfg, rowlevel)
    assert s38.DAS_MIN[0] == 0
    split = pd.read_csv(res / "iecv_split.csv")
    E = cfg["paths"]["eicu"]
    h = duckdb.execute(f"SELECT hospitalid, region, numbedscategory, teachingstatus FROM '{E}/hospital.parquet'").df()
    h = h.set_index("hospitalid").reindex(split.hospitalid)
    h["no_med"] = split.set_index("hospitalid").no_medication
    h["has_das"] = [any(x in V[c].index for c in s38.CONDITIONS) for x in h.index]
    assert (e.index == np.arange(len(e))).all()
    e = e[e.hospitalid.isin(split.hospitalid)].copy()
    for t, y in Y.items():
        e[t] = y[e.index]
    e["no_med"] = e.hospitalid.map(h.no_med)
    e["elig_readmit"] = s38.ELIG["icu_readmit"][e.index]
    groups = {"All hospitals": (h.index, e),
              "Medication records": (h.index[~h.no_med], e[~e.no_med]),
              "No medication records": (h.index[h.no_med], e[e.no_med])}
    rows = []

    def add(label, fn):
        rows.append({"Characteristic": label, **{g: fn(hh, ee) for g, (hh, ee) in groups.items()}})

    def cnt(mask_fn):
        return lambda hh, ee: (lambda k, n: f"{k} ({100 * k / n:.0f}%)")(int(mask_fn(h.loc[hh]).sum()), len(hh))

    def med(col, d=0):
        return lambda hh, ee: (lambda q: f"{q[1]:.{d}f} [{q[0]:.{d}f}, {q[2]:.{d}f}]")(np.percentile(ee[col], [25, 50, 75]))

    def share(col):
        return lambda hh, ee: f"{int(ee[col].sum()):,} ({100 * ee[col].mean():.1f}%)"

    add("Hospitals", lambda hh, ee: f"{len(hh)}")
    add("Hospitals with a DAS measure", cnt(lambda d: d.has_das))
    for r in ["Midwest", "Northeast", "South", "West"]:
        add(f"Region: {r}", cnt(lambda d, r=r: d.region == r))
    add("Region: unknown", cnt(lambda d: d.region.isna()))
    for b in ["<100", "100 - 249", "250 - 499", ">= 500"]:
        add(f"Beds: {b.replace(' - ', ' to ').replace('>= ', '≥')}", cnt(lambda d, b=b: d.numbedscategory == b))
    add("Beds: unknown", cnt(lambda d: d.numbedscategory.isna()))
    add("Teaching hospital", cnt(lambda d: d.teachingstatus == True))  # noqa: E712
    add("ICU stays", lambda hh, ee: f"{len(ee):,}")
    add("ICU stays per hospital, median [IQR]", lambda hh, ee: (lambda q: f"{q[1]:.0f} [{q[0]:.0f}, {q[2]:.0f}]")(
        np.percentile(ee.groupby("hospitalid").size(), [25, 50, 75])))
    add("Age, years, median [IQR]", med("agenum"))
    add("APACHE IV score, median [IQR]", med("apachescore"))
    add("Admitted from the emergency department", share("unitadmitsource=Emergency Department"))
    add("Transferred from another hospital", share("unitadmitsource=Other Hospital"))
    add("In-hospital death", share("mortality"))
    add("ICU stay over 5 days", share("A7_los_gt5d"))
    add("ICU stay over 7 days", share("A3_los_gt7d"))
    add("Hospital stay over 14 days", share("A6_hospital_los_gt14d"))
    add("ICU readmission", lambda hh, ee: share("icu_readmit")(hh, ee[ee.elig_readmit]))
    pd.DataFrame(rows).to_csv(res / "paper_table1.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
