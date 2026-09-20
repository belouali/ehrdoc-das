"""Venn diagram visualizations for documentation behavior.

Replicates the exact logic from the original analysis notebook.
Includes hospital characteristics (region, bed size, teaching status) in titles.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import pandas as pd


def _get_pid_col(df: pd.DataFrame) -> str:
    if "uniquepid" in df.columns:
        return "uniquepid"
    return "patientunitstayid"


def _to_bool(series: pd.Series) -> pd.Series:
    """Robustly convert a flag column to boolean, handling bool/int/float types."""
    return series.astype(bool)


def _get_hospital_chars(flags: pd.DataFrame, hid: int) -> dict[str, str]:
    """Extract hospital characteristics from the flags dataframe."""
    sub = flags.loc[flags["hospitalid"] == hid].head(1)
    chars = {}
    for col, label in [
        ("numbedscategory", "Beds"),
        ("teachingstatus", "Teaching"),
        ("region", "Region"),
    ]:
        if col in sub.columns and not sub[col].isna().all():
            val = sub[col].iloc[0]
            if pd.isna(val):
                continue
            if col == "teachingstatus":
                val = "Yes" if str(val).lower() in ("t", "true", "1", "yes") else "No"
            chars[label] = str(val)
    return chars


def plot_hospital_venn(
    patient_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    title: str | None = None,
    set_labels: tuple[str, str, str] = ("Medication", "Past History", "Diagnosis"),
) -> plt.Axes:
    """Plot a Venn diagram for one hospital's patients.

    Exact original logic: filter rows where flag is True, collect pid into set.
    Uses robust boolean conversion to handle bool/int/float flag columns.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    pid = _get_pid_col(patient_df)

    # Use robust boolean conversion instead of == 1
    medication_set = set(patient_df.loc[_to_bool(patient_df["exists_in_medication"]), pid])
    pastHistory_set = set(patient_df.loc[_to_bool(patient_df["exists_in_pastHistory"]), pid])
    diagnosis_set = set(patient_df.loc[_to_bool(patient_df["exists_in_diagnosis"]), pid])

    venn3(
        [medication_set, pastHistory_set, diagnosis_set],
        set_labels,
        ax=ax,
    )

    if title:
        ax.set_title(title, fontsize=9)

    return ax


def plot_triptych(
    flags: pd.DataFrame,
    hospitals: list[int],
    condition: str = "Diabetes",
    angular_distances: dict[int, float] | None = None,
    ref_hospital: int | None = None,
    figsize: tuple[float, float] = (20, 8),
    show_characteristics: bool = True,
) -> plt.Figure:
    """Create a multi-panel Venn diagram figure with hospital characteristics."""
    n = len(hospitals)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    if ref_hospital is None and hospitals:
        ref_hospital = hospitals[0]
    if angular_distances is None:
        angular_distances = {}

    pid = _get_pid_col(flags)

    for ax, hid in zip(axes, hospitals):
        cols = [c for c in [pid, "hospitalid", "exists_in_medication",
                            "exists_in_pastHistory", "exists_in_diagnosis"]
                if c in flags.columns]
        sub = flags.loc[flags["hospitalid"] == hid, cols].drop_duplicates()

        # Use robust boolean conversion for filtering
        sub = sub[
            _to_bool(sub["exists_in_medication"]) |
            _to_bool(sub["exists_in_pastHistory"]) |
            _to_bool(sub["exists_in_diagnosis"])
        ]
        total = sub[pid].nunique()

        # Build title with "Hospital ID: X" format
        title_lines = [f"Hospital ID: {hid}"]
        if hid == ref_hospital:
            title_lines[0] += " (reference)"
        elif hid in angular_distances:
            title_lines.append(
                f"Angular distance to {ref_hospital}: {angular_distances[hid]:.2f}\u00b0"
            )
        title_lines.append(f"n = {total:,} patients")

        # Add hospital characteristics
        if show_characteristics:
            chars = _get_hospital_chars(flags, hid)
            if chars:
                char_str = " | ".join(f"{k}: {v}" for k, v in chars.items())
                title_lines.append(char_str)

        plot_hospital_venn(sub, ax=ax, title="\n".join(title_lines))

    fig.suptitle(
        f"{condition.capitalize()} \u2014 Documentation Venn Diagrams",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    return fig
