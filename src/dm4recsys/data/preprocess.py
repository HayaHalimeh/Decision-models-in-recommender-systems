"""5-core filtering, chronological ordering and leave-last-out bookkeeping."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _item_sort_key(items: pd.Series) -> pd.Series:
    """Natural tie-break key: numeric IDs sort numerically (ML-1M), others lexicographically."""
    num = pd.to_numeric(items, errors="coerce")
    if num.notna().all():
        return num
    return items.astype(str)


ROW = "_row"  # line number in the raw file, set once by preprocess() before any filtering


def order_interactions(df: pd.DataFrame, tie_break: str = "file_order") -> pd.DataFrame:
    """Chronological per user. Ties in timestamp: `file_order` keeps the order of the raw file
    (column `_row`, or the current row order if absent); `item_id` sorts by item ID. Adds `pos` (0-based)."""
    df = df.copy()
    if tie_break == "item_id":
        df["_ik"] = _item_sort_key(df["item"])
        df = df.sort_values(["user", "timestamp", "_ik"], kind="mergesort").drop(columns="_ik")
    elif tie_break == "file_order":
        keys = ["user", "timestamp"] + ([ROW] if ROW in df.columns else [])
        df = df.sort_values(keys, kind="mergesort")
    else:
        raise ValueError(f"data.tie_break must be 'file_order' or 'item_id', not {tie_break!r}")
    df["pos"] = df.groupby("user").cumcount()
    return df.reset_index(drop=True)


def dedup(df: pd.DataFrame, how: str = "first", tie_break: str = "file_order") -> pd.DataFrame:
    """Repeated (user, item) pairs. `first` keeps the earliest one (after ordering), `none` keeps all."""
    if how == "none":
        return df
    df = order_interactions(df, tie_break)
    keep = "first" if how == "first" else "last"
    return df.drop_duplicates(["user", "item"], keep=keep).drop(columns="pos")


def kcore(df: pd.DataFrame, k: int = 5, max_iter: int = 1000) -> tuple[pd.DataFrame, int]:
    """Iterative k-core: repeat until every user and every item has >= k interactions."""
    it = 0
    while True:
        it += 1
        uc = df["user"].map(df["user"].value_counts())
        ic = df["item"].map(df["item"].value_counts())
        mask = (uc >= k) & (ic >= k)
        if mask.all():
            return df.reset_index(drop=True), it
        df = df[mask]
        if it >= max_iter:
            raise RuntimeError("k-core did not converge")


def is_kcore(df: pd.DataFrame, k: int) -> bool:
    return bool((df["user"].value_counts() >= k).all() and (df["item"].value_counts() >= k).all())


def stats(df: pd.DataFrame) -> dict:
    n_u, n_i, n = df["user"].nunique(), df["item"].nunique(), len(df)
    lens = df.groupby("user").size()
    return {
        "users": int(n_u),
        "items": int(n_i),
        "interactions": int(n),
        "density": float(n / (n_u * n_i)) if n_u and n_i else 0.0,
        "mean_seq_len": float(lens.mean()),
        "median_seq_len": float(lens.median()),
    }


def preprocess(df: pd.DataFrame, titles: dict, dcfg_data: dict) -> tuple[pd.DataFrame, dict, dict]:
    raw = stats(df)
    tie_break = dcfg_data.get("tie_break", "file_order")
    df = df.reset_index(drop=True)
    df[ROW] = np.arange(len(df))  # raw file order, the tie-break for equal timestamps
    dropped = {"items": 0, "interactions": 0}
    if dcfg_data.get("drop_untitled_items", True):
        # before the k-core, so the 5-core runs on titled items only
        titled = df["item"].map(lambda i: bool(titles.get(i)))
        dropped = {"items": int(df.loc[~titled, "item"].nunique()), "interactions": int((~titled).sum())}
        df = df[titled]
    df = dedup(df, dcfg_data.get("dedup", "first"), tie_break)
    df, n_iter = kcore(df, dcfg_data.get("min_core", 5))
    df = order_interactions(df, tie_break).drop(columns=ROW)
    used_titles = {i: titles[i] for i in df["item"].unique()}
    st = {"raw": raw, "dropped_untitled": dropped, "final": stats(df), "kcore_iterations": n_iter}
    return df, used_titles, st
