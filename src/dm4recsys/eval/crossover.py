from __future__ import annotations

import pandas as pd


def curve(seq: pd.DataFrame, value: str = "ndcg") -> pd.DataFrame:
    """Mean over users per (frac, seed), then mean / min / max over seeds per frac."""
    per_seed = seq.groupby(["frac", "seed"])[value].mean().reset_index()
    return per_seed.groupby("frac")[value].agg(["mean", "min", "max", "std"]).reset_index().sort_values("frac")


def crossover(curve_df: pd.DataFrame, threshold: float):
    """Smallest training share (%) whose seed-mean exceeds `threshold`; 'never' if none within 100 %."""
    for f, m in zip(curve_df["frac"], curve_df["mean"]):
        if m > threshold:
            return f
    return "never"
