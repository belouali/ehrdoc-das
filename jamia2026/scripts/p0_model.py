"""Shared P0 model pipeline and metrics (stages 3 and 4).

Every preprocessing step is fitted on eICU training data only: median
imputation with missing-value indicators, standardization, then L2-penalized
logistic regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VITALS = ["hr", "map", "rr", "temp", "spo2"]
LABS = ["creatinine", "bun", "sodium", "potassium", "hgb", "wbc", "platelets"]


def feature_columns() -> list[str]:
    cols = ["age", "male", "gcs_min"]
    for v in VITALS:
        cols += [f"{v}_min", f"{v}_max", f"{v}_median"]
    for l in LABS:
        cols += [f"{l}_min", f"{l}_max"]
    return cols


def make_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=5000)),
    ])


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration(y, p) -> tuple[float, float]:
    """Calibration slope (logistic regression of y on logit p) and
    calibration-in-the-large (intercept with logit p as offset)."""
    lp = _logit(p).reshape(-1, 1)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y).coef_[0, 0]
    a = 0.0
    for _ in range(50):                      # Newton steps for the offset-only intercept
        q = 1 / (1 + np.exp(-(a + lp[:, 0])))
        step = (y - q).sum() / (q * (1 - q)).sum()
        a += step
        if abs(step) < 1e-10:
            break
    return float(slope), float(a)


def metrics(y, p) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    slope, citl = calibration(y, p)
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                brier=brier_score_loss(y, p), cal_slope=slope, cal_intercept=citl,
                n=len(y), events=int(y.sum()))
