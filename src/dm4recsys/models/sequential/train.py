"""SASRec / BERT4Rec / GRU4Rec with RecBole: grid search, training-data curve, scoring of frozen candidates.

Item vocabulary always = full dataset (the .item atomic file lists every item). At small training
fractions many items never occur in training; their embeddings stay near initialisation and are only
pushed down through the softmax denominator of the CE loss (see README).
"""
from __future__ import annotations

import itertools
import logging
import math
import shutil
import time

import numpy as np

from ...config import model_cfg, path
from ...data.store import DatasetStore
from ...prompts.transforms import select_history
from ...retrieval import recbole_utils as rb
from ...retrieval.candidates import load_candidates
from ...run.progress import report
from ...run.runner import make_row
from ...run.trials import TrialStore
from ...utils import pick_device, read_json, read_jsonl, rng_for, sha256_file, write_json, write_jsonl
from ..base import RankRequest, RankResult, rank_from_scores

log = logging.getLogger(__name__)


TUNE_SEED = 0


def grid(g: dict, tied: dict | None = None) -> list[dict]:
    """All grid points. `tied` = {param: source param}: param takes the source's value at every point."""
    keys = sorted(g)
    points = [dict(zip(keys, v)) for v in itertools.product(*(g[k] for k in keys))]
    for p in points:
        p.update({k: p[src] for k, src in (tied or {}).items()})
    return points


def sample_train_users(users: list[str], frac: float, seed: int, dataset: str, min_users: int) -> list[str]:
    users = sorted(users)
    if frac >= 100:
        return users
    n = max(min_users, math.ceil(len(users) * frac / 100.0))
    n = min(n, len(users))
    idx = rng_for("train-frac", dataset, frac, seed).choice(len(users), size=n, replace=False)
    return [users[i] for i in sorted(idx)]


def curve_path(cfg, dataset, model, frac, seed):
    return path(cfg, "runs", dataset, model, f"f{frac}_s{seed}.jsonl")


def test_history(cfg: dict, rec: dict) -> list[str]:
    if cfg["sequential"].get("test_history", "last_k") == "full":
        return select_history(rec, [{"name": "last_k", "k": cfg["history"]["full_history_k"]}])
    return select_history(rec, cfg["history"]["transforms"])


def score_candidates(cfg, model_name, trained, recs, frac, seed, cand_sha):
    model, ds = trained["model"], trained["dataset"]
    max_len = cfg["sequential"]["max_seq_length"]
    dev = str(next(model.parameters()).device)
    rows = []
    for rec in recs:
        hist = test_history(cfg, rec)
        t0 = time.perf_counter()
        full = rb.seq_full_scores(model, ds, hist, max_len)
        cidx = rb.item_index(ds, rec["candidates"])
        sc = full[cidx]
        scores = dict(zip(rec["option_ids"], sc.tolist()))
        order = rank_from_scores(scores, rec["option_ids"], model_name, rec["dataset"], rec["user"], seed)
        lat = time.perf_counter() - t0
        req = RankRequest(rec["dataset"], rec["user"], hist, [], rec["candidates"], [], rec["option_ids"])
        res = RankResult(order=order, scores=scores, model_version=f"recbole-{trained['config']['model']}")
        rows.append(make_row(cfg, rec, req, res, model_name, seed, lat, cand_sha, frac=frac, seed=seed,
                             device=dev, train_users=trained.get("n_train_users")))
    return rows


def train_one(cfg, store, mname, params, users, seed, tag):
    mcfg = model_cfg(cfg, mname)
    scfg = cfg["sequential"]
    device = pick_device(scfg.get("device", "auto"))
    work = path(cfg, "training", "recbole_data", store.name, tag)
    seqs = {u: store.sequences[u] for u in users}
    rb.write_atomic(work, store.name, seqs, store.all_items)
    seen = {u: s[:-2] for u, s in seqs.items()}  # items before the validation target (split valid_and_test)
    try:
        res = rb.train(mcfg["recbole_model"], work, store.name, {**mcfg.get("fixed", {}), **params}, scfg,
                       "valid_and_test", seed, device, seen=seen)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    res["n_train_users"] = len(users)
    res["device"] = device
    return res


_TUNED: dict = {}  # (dataset, model) -> trained best model; reused as the 100 % / seed-0 cell of the curve


# sequential settings that do not influence a tuning trial (only the curve / scoring use them)
_NOT_TUNING = {"models", "fractions", "seeds", "min_train_users", "test_history"}


def seq_fingerprint(cfg, store, mname, device) -> dict:
    """Everything that determines a tuning trial's result besides its grid point."""
    mcfg = model_cfg(cfg, mname)
    env = rb.env_info(device)
    return {"model": mname, "recbole_model": mcfg["recbole_model"], "fixed": mcfg.get("fixed", {}),
            "dataset": store.name, "seed": TUNE_SEED, "fraction": 100,
            "sequential": {k: v for k, v in cfg["sequential"].items() if k not in _NOT_TUNING},
            "interactions_sha256": sha256_file(path(cfg, "processed", store.name, "interactions.tsv")),
            "n_items": len(store.all_items), **{k: env.get(k) for k in ("device", "gpu", "torch", "recbole")}}


def tune(cfg, store, mname) -> dict:
    """Grid search at 100 % of training users, seed 0, on the validation targets.
    Every finished trial is saved at once (training/.../sequential/{ds}/{model}.trials.jsonl); a restart reuses saved
    trials with the same fingerprint and trains only the missing grid points."""
    tp = path(cfg, "training", "sequential", store.name, f"{mname}.json")
    if tp.exists():
        return read_json(tp)
    mcfg = model_cfg(cfg, mname)
    scfg = cfg["sequential"]
    device = pick_device(scfg.get("device", "auto"))
    saved = TrialStore(tp.with_suffix(".trials.jsonl"), seq_fingerprint(cfg, store, mname, device))
    users = store.users
    points = grid(mcfg["grid"], mcfg.get("tied"))
    trials, best_res, best_valid = [], None, -np.inf
    for i, params in enumerate(points, 1):
        t, res = saved.get(params), None
        if t is None:
            t0 = time.perf_counter()
            res = train_one(cfg, store, mname, params, users, TUNE_SEED, f"tune_{mname}")
            t = saved.add({"params": params, "best_valid": res["best_valid"], "best_epoch": res["best_epoch"],
                           "minutes": (time.perf_counter() - t0) / 60})
        trials.append({"params": t["params"], "best_valid": t["best_valid"], "best_epoch": t["best_epoch"],
                       "minutes": t.get("minutes")})
        log.info("%s %s tuning %d/%d %s valid=%.4f best_epoch=%d (%.1f min, %s) -> %s", mname, store.name, i,
                 len(points), t["params"], t["best_valid"], t["best_epoch"], t.get("minutes") or 0,
                 "trained" if res is not None else "reused (saved)", saved.path)
        if t["best_valid"] > best_valid:  # first maximum in grid order, as max() below
            best_valid = t["best_valid"]
            best_res = (params, res) if res is not None else None  # a reused best is retrained by the curve
        else:
            del res
    best = max(trials, key=lambda t: t["best_valid"])
    if best_res is not None:
        _TUNED[(store.name, mname)] = best_res
    out = {"trials": trials, "best": best, "seed": TUNE_SEED, "fraction": 100,
           "valid_metric": scfg["valid_metric"], "valid_masks_seen_items": True,
           "max_epochs": scfg["max_epochs"], "patience": scfg["patience"], "max_seq_length": scfg["max_seq_length"],
           **rb.env_info(device)}
    write_json(tp, out)
    log.info("%s %s tuning done: best %s valid=%.4f best_epoch=%d -> %s", mname, store.name, best["params"],
             best["best_valid"], best["best_epoch"], tp)
    return out


def run_curve(cfg: dict, dataset: str, mname: str, fractions=None, seeds=None) -> None:
    scfg = cfg["sequential"]
    store = DatasetStore.load(cfg, dataset)
    recs, meta = load_candidates(cfg, dataset)
    t_model = time.perf_counter()
    tuned = tune(cfg, store, mname)
    best = tuned["best"]["params"]
    fractions, seeds = fractions or scfg["fractions"], seeds or scfg["seeds"]
    k = cfg["eval"]["k"]
    n_cells, i = len(fractions) * len(seeds), 0
    for frac in fractions:
        for seed in seeds:
            i += 1
            out = curve_path(cfg, dataset, mname, frac, seed)
            if len(read_jsonl(out)) == len(recs):
                continue
            users = sample_train_users(store.users, frac, seed, dataset, scfg["min_train_users"])
            t0 = time.perf_counter()
            cached = _TUNED.pop((dataset, mname), None) if frac >= 100 and seed == TUNE_SEED else None
            if cached is not None and cached[0] == best:
                trained = cached[1]  # identical to retraining: same users, params and seed
            else:
                trained = train_one(cfg, store, mname, best, users, seed, f"{mname}_f{frac}_s{seed}")
            rows = score_candidates(cfg, mname, trained, recs, frac, seed, meta["sha256"])
            write_jsonl(out, rows)  # written atomically once complete -> a crash re-trains this cell only
            log.info("%s %s cell %d/%d f=%s s=%s users=%d valid=%.4f best_epoch=%d ndcg%d=%.4f (%.1f min) -> %s",
                     mname, dataset, i, n_cells, frac, seed, len(users), trained["best_valid"], trained["best_epoch"],
                     k, np.mean([r[f"ndcg{k}"] for r in rows]), (time.perf_counter() - t0) / 60, out)
            del trained
    # summary of this model on this dataset (all cells, including those finished before a restart)
    cells = {(f, s): read_jsonl(curve_path(cfg, dataset, mname, f, s)) for f in fractions for s in seeds}
    done = {fs: rows for fs, rows in cells.items() if len(rows) == len(recs)}
    by_frac = {str(f): float(np.mean([np.mean([r[f"ndcg{k}"] for r in done[(f, s)]]) for s in seeds]))
               for f in fractions if all((f, s) in done for s in seeds)}  # mean over seeds of the user mean
    summary = {"model": mname, "dataset": dataset, "best_params": best, "best_valid": tuned["best"]["best_valid"],
               "best_epoch": tuned["best"]["best_epoch"], "cells_done": f"{len(done)}/{n_cells}",
               f"ndcg{k}_by_fraction": by_frac}
    full = [r for (f, s), rows in done.items() if f == 100 for r in rows]
    if full:  # injected and realistic (= injected x r_u) NDCG at 100 % training users, over all seeds
        summary[f"ndcg{k}_100pct"] = float(np.mean([r[f"ndcg{k}"] for r in full]))
        summary[f"realistic_ndcg{k}_100pct"] = float(np.mean([r[f"ndcg{k}"] * r["r_u"] for r in full]))
    report(cfg, "sequential", **summary, minutes_this_session=(time.perf_counter() - t_model) / 60,
           files=str(curve_path(cfg, dataset, mname, 100, 0).parent))
