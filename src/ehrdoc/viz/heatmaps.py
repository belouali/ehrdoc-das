from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_matrix_heatmap(mat: np.ndarray, labels: list[str], title: str = "", annotate: bool = True):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha='right')
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i,j]
                ax.text(j, i, f"{val:.2f}" if np.isfinite(val) else "NA",
                        ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax
