"""Change metrics against Real (same user) and the RQ2/RQ3 tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

KEY = ["dataset", "user"]


def change(cond: pd.DataFrame, real: pd.DataFrame, k: int = 5) -> dict:
    """Paired on (dataset, user). Expects columns ndcg, rank, top1 and p_hit (tie-aware Hit@k, rq/collect._p_hit;
    derived from rank if absent).
    top1_change: share of users whose top-1 ITEM differs from Real (a refusal has no top-1 -> changed);
    mean_abs_dndcg: mean |NDCG@k(cond) - NDCG@k(real)| (tie-aware NDCG);
    hit_flip: share of users whose target moves into or out of the top k (Lyu & Xia 2026), tie-aware:
    the expected flip when tied options are ordered at random independently in both rankings,
    p(1 - q) + q(1 - p) with p, q the users' tie-aware Hit@k; equal to the plain flip when there are no ties.
    refusal_share: share of the paired users whose condition call was refused after all reruns (scored NDCG 0;
    column `refused`, 0 if absent)."""
    cond = cond if "refused" in cond else cond.assign(refused=False)
    cond, real = (x if "p_hit" in x else x.assign(p_hit=(x["rank"] <= k).astype(float)) for x in (cond, real))
    cols = ["ndcg", "rank", "top1", "p_hit"]
    m = cond[KEY + cols + ["refused"]].merge(real[KEY + cols], on=KEY, suffixes=("", "_real"))
    if m.empty:
        return {"n": 0, "top1_change": np.nan, "mean_abs_dndcg": np.nan, "hit_flip": np.nan,
                "ndcg": np.nan, "ndcg_real": np.nan, "refusal_share": np.nan}
    p, q = m["p_hit"].astype(float), m["p_hit_real"].astype(float)
    return {"n": int(len(m)),
            "top1_change": float((m["top1"] != m["top1_real"]).mean()),
            "mean_abs_dndcg": float((m["ndcg"] - m["ndcg_real"]).abs().mean()),
            "hit_flip": float((p * (1 - q) + q * (1 - p)).mean()),
            "ndcg": float(m["ndcg"].mean()), "ndcg_real": float(m["ndcg_real"].mean()),
            "refusal_share": float(m["refused"].astype(bool).mean())}


def paired(cond: pd.DataFrame, real: pd.DataFrame, col: str = "ndcg", col_real: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    col_real = col_real or col
    m = cond[KEY + [col]].merge(real[KEY + [col_real]].rename(columns={col_real: "_b"}), on=KEY)
    return m[col].to_numpy(float), m["_b"].to_numpy(float)


def friedman(table: np.ndarray) -> dict:
    """table: users x conditions (blocks = users). All-constant data -> statistic 0, p = 1."""
    table = np.asarray(table, float)
    if table.shape[0] < 2 or table.shape[1] < 3:
        return {"chi2": np.nan, "p": np.nan, "n_users": int(table.shape[0])}
    if np.allclose(table, table[:, :1]):
        return {"chi2": 0.0, "p": 1.0, "n_users": int(table.shape[0])}
    chi2, p = st.friedmanchisquare(*table.T)
    return {"chi2": float(chi2), "p": float(p) if np.isfinite(p) else 1.0, "n_users": int(table.shape[0])}


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar test on paired binary outcomes a, b (e.g. top-1 changed under label swap vs. rerun)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = 1.0 if n10 + n01 == 0 else float(st.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)
    return {"n10": n10, "n01": n01, "p": p, "n": int(len(a)), "rate_a": float(a.mean()) if len(a) else np.nan,
            "rate_b": float(b.mean()) if len(b) else np.nan}


def label_outcome(real_top1_item: str, real_top1_id: str, swap_top1_item: str, swap_options: dict) -> str:
    """same_item: the same item (now with a new ID); same_label: the item that now carries Real's top-1 ID."""
    if swap_top1_item == real_top1_item:
        return "same_item"
    if swap_options.get(real_top1_id) == swap_top1_item:
        return "same_label"
    return "other"


def popularity_percentiles(counts: dict, items: list[str]) -> dict[str, float]:
    """Percentile rank (0, 1] of each item's training interaction count within the dataset (ties averaged)."""
    s = pd.Series({i: counts.get(i, 0) for i in items}, dtype=float)
    return s.rank(method="average", pct=True).to_dict()
