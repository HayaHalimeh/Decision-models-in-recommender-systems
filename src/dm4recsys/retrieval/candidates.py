"""Injected and realistic candidate sets from BPR scores, the invariants that tie them together,
and the frozen on-disk format."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import path
from ..utils import read_json, read_jsonl, rng_for, sha256_file, write_json, write_jsonl

log = logging.getLogger(__name__)


class FrozenCandidatesError(RuntimeError):
    pass


def build_record(user: str, target: str, history: list[str], item_tokens: np.ndarray, scores: np.ndarray,
                 n: int, shuffle_seed: int, dataset: str, split: str) -> dict:
    """item_tokens[j] is the item for scores[j] (padding/unknown entries must already be removed).

    - every history item is excluded from the ranking (all earlier interactions);
    - injected = target + the (n-1) highest-scoring other items;
    - realistic = BPR top n (r_u = 1 if the target is among them);
    - order = shuffle with a fixed per-user seed; option IDs "1".."n" follow that order.
    """
    hist = set(history)
    assert target not in hist, f"target {target} appears in the history of user {user}"
    keep = np.array([t not in hist for t in item_tokens])
    toks, sc = item_tokens[keep], scores[keep]
    if len(toks) < n:
        raise ValueError(f"user {user}: only {len(toks)} items outside the history (< {n} candidates)")
    order = np.lexsort((np.arange(len(sc)), -sc))  # score desc, ties by catalogue index (stable)
    ranked = toks[order]
    target_rank = int(np.flatnonzero(ranked == target)[0]) + 1
    top_n = ranked[:n].tolist()
    r_u = int(target in top_n)
    others = [t for t in ranked[: n + 1].tolist() if t != target][: n - 1]
    injected = [target] + others
    score_of = dict(zip(toks.tolist(), sc.tolist()))
    perm = rng_for(shuffle_seed, dataset, user).permutation(n)
    cands = [injected[i] for i in perm]
    rec = {
        "user": user,
        "dataset": dataset,
        "split": split,
        "target": target,
        "history": list(history),
        "candidates": cands,
        "option_ids": [str(i + 1) for i in range(n)],
        "target_option": str(cands.index(target) + 1),
        "r_u": r_u,
        "bpr_target_rank": target_rank,
        "bpr_scores": [float(score_of[c]) for c in cands],
        "realistic_top": top_n,
    }
    check_record(rec, n)
    return rec


def check_record(rec: dict, n: int) -> None:
    c = rec["candidates"]
    assert len(c) == n and len(set(c)) == n, "candidate set must have n distinct items"
    assert c.count(rec["target"]) == 1, "injected set must contain the target exactly once"
    assert not (set(c) & set(rec["history"])), "candidates must not contain history items"
    assert rec["target"] not in rec["history"], "history must not contain the target"
    if rec["r_u"] == 1:
        # Realistic protocol: for r_u = 1 the realistic set IS the injected set (exact set equality),
        # so injected-run outputs are reused without running any model again.
        assert set(c) == set(rec["realistic_top"]), "r_u=1 but injected set != BPR top-n"
    else:
        assert rec["target"] not in rec["realistic_top"]
        assert rec["bpr_target_rank"] > n


def candidates_path(cfg: dict, dataset: str, split: str = "test") -> Path:
    if split == "longhist":  # RQ2 history-length sample, frozen separately
        return path(cfg, "candidates", f"{dataset}_longhist.jsonl")
    suffix = "" if split == "test" else f".{split}"
    return path(cfg, "candidates", f"{dataset}{suffix}.jsonl")


def freeze(cfg: dict, dataset: str, split: str, records: list[dict], meta: dict, force: bool = False) -> Path:
    p = candidates_path(cfg, dataset, split)
    if p.exists() and not force:
        raise FrozenCandidatesError(f"{p} exists and is frozen. Pass --force to regenerate deliberately.")
    write_jsonl(p, records)
    meta = dict(meta, sha256=sha256_file(p), n_users=len(records))
    write_json(p.with_suffix(".meta.json"), meta)
    log.info("Froze %d candidate sets -> %s", len(records), p)
    return p


def load_candidates(cfg: dict, dataset: str, split: str = "test") -> tuple[list[dict], dict]:
    p = candidates_path(cfg, dataset, split)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `make candidates` first")
    meta = read_json(p.with_suffix(".meta.json"))
    if sha256_file(p) != meta["sha256"]:
        raise FrozenCandidatesError(f"{p} was modified after freezing (hash mismatch)")
    recs = read_jsonl(p)
    n = cfg["candidates"]["n"]
    for r in recs:
        check_record(r, n)
    return recs, meta
