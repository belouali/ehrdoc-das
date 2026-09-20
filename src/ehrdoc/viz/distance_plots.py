from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_angular_separation(distance_table: pd.DataFrame, hospital_1_id: int, hospital_2_ids=None, title_suffix: str=""):
    """Bar plot of angular separation from one hospital to others.
    Expects columns: 'Hospital 1', 'Hospital 2', 'Angular Separation'
    """
    df = distance_table[distance_table['Hospital 1'] == hospital_1_id].copy()
    if hospital_2_ids is not None:
        df = df[df['Hospital 2'].isin(hospital_2_ids)]
    df = df.sort_values('Angular Separation', ascending=True)

    x = df['Hospital 2'].astype(int).astype(str)
    y = df['Angular Separation'].astype(float)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, y)
    ax.set_xlabel('Hospital 2')
    ax.set_ylabel('Angular Separation (degrees)')
    ax.set_title(f"Angular Separation from Hospital {hospital_1_id} {title_suffix}")
    ax.tick_params(axis='x', rotation=90)
    # annotate lightly
    for i, val in enumerate(y):
        ax.text(i, val, f"{val:.1f}°", ha='center', va='bottom', fontsize=8, rotation=0)
    fig.tight_layout()
    return fig, ax

def plot_crosstab_with_distances(df: pd.DataFrame, distance_column: str, title: str='Crosstab with Distances'):
    """Heatmap-style plot without seaborn.
    Expects columns: Diagnosis1, Diagnosis2, and distance_column.
    """
    crosstab = df.pivot(index='Diagnosis1', columns='Diagnosis2', values=distance_column)
    row_sums = crosstab.sum(axis=1)
    col_sums = crosstab.sum(axis=0)
    crosstab = crosstab.loc[row_sums.sort_values(ascending=False).index, col_sums.sort_values(ascending=True).index]

    mat = crosstab.to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat)
    ax.set_xticks(range(crosstab.shape[1]), labels=crosstab.columns, rotation=45, ha='right')
    ax.set_yticks(range(crosstab.shape[0]), labels=crosstab.index)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i,j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax
