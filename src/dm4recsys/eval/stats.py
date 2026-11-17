from __future__ import annotations

import numpy as np
from scipy import stats as st


def bootstrap_ci(x: np.ndarray, resamples: int = 1000, seed: int = 2027, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling users."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(resamples, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_t(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if len(d) < 2 or np.allclose(d, d[0]):
        return {"t": float("nan") if not np.allclose(d, 0) else 0.0, "p": 1.0 if np.allclose(d, 0) else 0.0,
                "n": int(len(d)), "mean_diff": float(d.mean()) if len(d) else float("nan")}
    t, p = st.ttest_rel(a, b)
    return {"t": float(t), "p": float(p), "n": int(len(d)), "mean_diff": float(d.mean())}


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value (zero differences dropped); robustness check for paired_t."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.allclose(a - b, 0):
        return 1.0
    return float(st.wilcoxon(a, b).pvalue)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()

