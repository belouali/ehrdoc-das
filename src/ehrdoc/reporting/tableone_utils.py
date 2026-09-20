from __future__ import annotations
import pandas as pd
from scipy.stats import normaltest

try:
    from tableone import TableOne
except Exception:
    TableOne = None

def is_normal(data) -> bool:
    stat, p = normaltest(data)
    return bool(p > 0.05)

def get_table1(df: pd.DataFrame, columns_exclude, label: str):
    if TableOne is None:
        raise ImportError("tableone is not installed. pip install tableone")
    nonnormal = []
    for col in df.columns.drop(columns_exclude):
        try:
            if not is_normal(df[col]):
                nonnormal.append(col)
        except Exception:
            pass
    columns = df.columns.drop(columns_exclude).to_list()
    return TableOne(df, columns=columns, nonnormal=nonnormal, groupby=label, label_suffix=True, pval=True)

def get_table1_adjusted(df: pd.DataFrame, columns_exclude, label: str):
    if TableOne is None:
        raise ImportError("tableone is not installed. pip install tableone")
    columns = df.columns.drop(columns_exclude).to_list()
    return TableOne(df, columns=columns, groupby=label, label_suffix=True, pval=True, pval_adjust='bonferroni')

def get_table1_significant_features(table1, p: float):
    try:
        tableone_df = pd.concat([table1.cont_table, table1.cat_table])
    except Exception:
        tableone_df = pd.concat([table1.cont_table])
    try:
        final = tableone_df[tableone_df['P-Value'] < p]
    except Exception:
        final = tableone_df[tableone_df['P-Value (adjusted)'] < p]
    return final.reset_index(drop=False)
