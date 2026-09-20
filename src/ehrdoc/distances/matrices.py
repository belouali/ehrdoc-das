from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import angular_separation

def distance_matrix(
    vectors_df: pd.DataFrame,
    id_col: str = "hospitalid",
    metric = angular_separation,
) -> tuple[np.ndarray, list]:
    ids = vectors_df[id_col].tolist()
    X = vectors_df.drop(columns=[id_col]).to_numpy(dtype=float)
    n = X.shape[0]
    D = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = metric(X[i], X[j])
            D[i,j]=D[j,i]=d
    return D, ids

def flatten_upper_triangle(D: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(D, k=1)
    return D[idx]
