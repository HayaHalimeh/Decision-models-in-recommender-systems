"""Load per-user results into tidy DataFrames. Runs are averaged per user before any test."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import model_cfg, path
from .metrics import list_tie_groups, tie_groups, with_cutoff
from ..retrieval.candidates import load_candidates
from ..run.runner import n_repeats, result_path
from ..models.sequential.train import curve_path
from ..utils import read_jsonl

log = logging.getLogger(__name__)


def candidate_frame(cfg, datasets) -> pd.DataFrame:
    rows = []
    for d in datasets:
        recs, _ = load_candidates(cfg, d)
        rows += [{"dataset": d, "user": r["user"], "r_u": r["r_u"], "bpr_target_rank": r["bpr_target_rank"]} for r in recs]
    return pd.DataFrame(rows)


def require_tie_fields(rows: list[dict], where: str, k: int) -> None:
    """Every result row carries the tie-aware NDCG fields (ndcg{k}_order, tie)."""
    if rows and any(f"ndcg{k}_order" not in r or "tie" not in r for r in rows):
        raise RuntimeError(f"{where}: rows without the tie-aware NDCG fields (ndcg{k}_order, tie)")


def _zero_stats(row, k: int, probs: bool) -> tuple[float, float]:
    """(fewer than k options with probability > 0, target probability == 0) for one result row. Only defined for
    probability vectors (decision models); rows without scores, or PLM scores that can be negative -> NaN."""
    scores = row.get("scores")
    if not probs or not isinstance(scores, dict) or not scores:
        return np.nan, np.nan
    tid = row["order"][row["ranked_items"].index(row["target"])]
    return float(sum(1 for v in scores.values() if v > 0) < k), float(scores[tid] == 0)


def _ranking_key(row) -> tuple:
    """The ranking a row fixes without our seeded tie-break (blocks of tied options, see tie_groups); for a ranked
    list without scores (LLMs) the named options in order, then the unnamed ones as one block (list_tie_groups).
    Run-to-run stability compares these keys."""
    scores = row.get("scores")
    if isinstance(scores, dict) and scores:
        return tie_groups(scores, (row.get("info") or {}).get("fixed_first"))
    return list_tie_groups(row["order"], (row.get("info") or {}).get("n_found_valid"))


def zero_shot(cfg, model: str, datasets: list[str], cands: pd.DataFrame) -> pd.DataFrame | None:
    """Per-user frame (dataset, user, ndcg, r_u, compliant, in_tok, out_tok, calls) or None if incomplete."""
    k = cfg["eval"]["k"]
    R = n_repeats(cfg, model_cfg(cfg, model))
    frames = []
    for d in datasets:
        want = set(cands.loc[cands["dataset"] == d, "user"])
        runs = []
        for r in range(R):
            rows = with_cutoff(read_jsonl(result_path(cfg, d, model, r)), k)
            if model == "bpr":  # BPR's own ranking: a target it did not retrieve is below rank n (NDCG 0 even at
                for x in rows:  # k = n, where the saved rank n within the injected set would score)
                    if not x["r_u"]:
                        x[f"ndcg{k}"] = 0.0
            got = {x["user"] for x in rows}
            if got != want:
                log.warning("%s/%s run %d incomplete (%d/%d users) -> model left out", model, d, r, len(got & want), len(want))
                return None
            runs += rows
        require_tie_fields(runs, f"{model}/{d}", k)
        df = pd.DataFrame(runs)
        u = df.get("usage", pd.Series([{}] * len(df))).apply(lambda x: x or {})
        df["in_tok"] = u.apply(lambda x: x.get("input_tokens") or 0)
        df["out_tok"] = u.apply(lambda x: x.get("output_tokens") or 0)
        df["calls"] = u.apply(lambda x: x.get("calls") or 0)
        df["compliant_f"] = df["compliant"].map(lambda c: np.nan if c is None else float(c))
        probs = model_cfg(cfg, model)["group"] == "decision"
        z = df.apply(lambda x: _zero_stats(x, k, probs), axis=1, result_type="expand")
        df["lt_k_nonzero"], df["target_zero"] = z[0], z[1]
        df["has_scores"] = df["scores"].apply(lambda x: isinstance(x, dict) and bool(x))
        info = df.get("info", pd.Series([{}] * len(df))).apply(lambda x: x or {})
        df["refusal"] = info.apply(lambda x: float(bool(x.get("api_refusal"))))
        df["check_failed"] = info.apply(lambda x: float(bool(x.get("answer_check_failed"))))
        df["ranking"] = df.apply(_ranking_key, axis=1)  # tie blocks, not our tie-break
        df["top1"] = df["ranking"].apply(lambda x: x[0])
        agg = df.groupby("user").agg(ndcg=(f"ndcg{k}", "mean"), r_u=("r_u", "first"), n_runs=("run", "nunique"),
                                     ndcg_sd=(f"ndcg{k}", lambda s: float(np.std(s, ddof=0))),  # run-to-run
                                     top1_same=("top1", lambda s: float(s.nunique() == 1)),
                                     ranking_same=("ranking", lambda s: float(s.nunique() == 1)),
                                     refusals=("refusal", "sum"), check_failed=("check_failed", "sum"),
                                     compliant=("compliant_f", "mean"), in_tok=("in_tok", "mean"),
                                     out_tok=("out_tok", "mean"), calls=("calls", "mean"),
                                     ndcg_order=(f"ndcg{k}_order", "mean"), has_scores=("has_scores", "all"),
                                     lt_k_nonzero=("lt_k_nonzero", "mean"),
                                     target_zero=("target_zero", "mean")).reset_index()
        agg["dataset"] = d
        frames.append(agg)
    return pd.concat(frames, ignore_index=True)


def sequential(cfg, model: str, datasets: list[str], cands: pd.DataFrame) -> pd.DataFrame | None:
    k = cfg["eval"]["k"]
    frames = []
    for d in datasets:
        n = int((cands["dataset"] == d).sum())
        for f in cfg["sequential"]["fractions"]:
            for s in cfg["sequential"]["seeds"]:
                rows = with_cutoff(read_jsonl(curve_path(cfg, d, model, f, s)), k)
                if len(rows) != n:
                    log.warning("%s/%s f=%s s=%s missing -> model left out", model, d, f, s)
                    return None
                require_tie_fields(rows, f"{model}/{d} f={f} s={s}", k)
                df = pd.DataFrame(rows)[["user", f"ndcg{k}", "r_u"]].rename(columns={f"ndcg{k}": "ndcg"})
                df["dataset"], df["frac"], df["seed"] = d, f, s
                frames.append(df)
    return pd.concat(frames, ignore_index=True)
