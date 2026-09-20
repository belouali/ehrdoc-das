from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read CSV/Parquet with a small wrapper."""
    path = str(path)
    if path.endswith('.parquet'):
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, **kwargs)
