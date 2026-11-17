"""BPR retriever: grid search on validation targets, final training without test targets,
candidate generation for the test users and the RQ2 long-history sample."""
from __future__ import annotations

import itertools
import logging
import time

import numpy as np

from ..config import path
from ..data.store import DatasetStore
from ..run.progress import report
from ..run.trials import TrialStore
from ..utils import pick_device, sha256_file, stable_seed, write_json
from . import recbole_utils as rb
from .candidates import build_record, candidates_path, freeze

log = logging.getLogger(__name__)


def grid(g: dict) -> list[dict]:
    keys = sorted(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]


def trials_path(cfg: dict, dataset: str):
    return path(cfg, "training", "bpr", f"{dataset}.trials.jsonl")


def bpr_fingerprint(cfg: dict, store: DatasetStore, device: str) -> dict:
    """Everything that determines a tuning run's result besides its grid point."""
    env = rb.env_info(device)
    return {"model": "BPR", "dataset": store.name, "seed": cfg["seed"],
            "retriever": {k: v for k, v in cfg["retriever"].items() if k != "grid"},
            "interactions_sha256": sha256_file(path(cfg, "processed", store.name, "interactions.tsv")),
            "n_items": len(store.all_items),
            **{k: env.get(k) for k in ("device", "gpu", "torch", "recbole")}}


def train_bpr(cfg: dict, store: DatasetStore) -> dict:
    rcfg = cfg["retriever"]
    device = pick_device(rcfg.get("device", "auto"))
    if device == "mps":
        device = "cpu"  # RecBole's BPR sampler is CPU bound; MPS gives no benefit
    work = path(cfg, "training", "recbole_data", store.name, "full")
    rb.write_atomic(work, store.name, store.sequences, store.all_items)
    # 1) tuning: train on all but the last two items, early-stop on the validation target.
    #    Every finished run is saved at once; a restart reuses saved runs with the same fingerprint.
    saved = TrialStore(trials_path(cfg, store.name), bpr_fingerprint(cfg, store, device))
    points = grid(rcfg["grid"])
    trials = []
    for i, params in enumerate(points, 1):
        t = saved.get(params)
        if t is None:
            t0 = time.perf_counter()
            res = rb.train("BPR", work, store.name, params, rcfg, "valid_and_test", cfg["seed"], device)
            t = saved.add({"params": params, "best_valid": res["best_valid"], "best_epoch": res["best_epoch"],
                           "minutes": (time.perf_counter() - t0) / 60})
            del res
            source = "trained"
        else:
            source = "reused (saved)"
        trials.append({"params": t["params"], "best_valid": t["best_valid"], "best_epoch": t["best_epoch"],
                       "minutes": t.get("minutes")})
        log.info("BPR %s run %d/%d %s valid=%.4f best_epoch=%d (%.1f min, %s) -> %s", store.name, i, len(points),
                 t["params"], t["best_valid"], t["best_epoch"], t.get("minutes") or 0, source, saved.path)
    best = max(trials, key=lambda t: t["best_valid"])
    # 2) final model: all interactions except each user's test target, fixed #epochs from tuning
    if rcfg.get("final_train_includes_valid", True):
        final = rb.train("BPR", work, store.name, best["params"], rcfg, "test_only", cfg["seed"], device,
                         fixed_epochs=best["best_epoch"] + 1)
    else:
        final = rb.train("BPR", work, store.name, best["params"], rcfg, "valid_and_test", cfg["seed"], device)
    final["tuning"] = {"trials": trials, "best": best}
    final["device"] = device
    return final


def longhist_users(cfg: dict, store: DatasetStore) -> list[str]:
    """RQ2 history-length sample: the n users with the longest pre-test histories after the 5-core
    (any user, not only test users). Ties in length are broken by a fixed per-user hash."""
    n = int(cfg["longhist"]["n_users"])
    seed = cfg["data"]["sample_seed"]
    key = {u: (-len(store.history_before_test(u)), stable_seed(seed, "longhist", store.name, u)) for u in store.users}
    return sorted(store.users, key=key.__getitem__)[:n]


def longhist_meta(cfg: dict, recs: list[dict]) -> dict:
    cap = min(len(r["history"]) for r in recs)
    return {"cap": cap, "lengths": [L for L in cfg["longhist"]["lengths"] if L <= cap],
            "lengths_dropped": [L for L in cfg["longhist"]["lengths"] if L > cap],
            "max_history": max(len(r["history"]) for r in recs)}


def _env(device: str | None) -> dict:
    """Software/hardware that produced the candidates (needed to rebuild them bit-exactly)."""
    return {("bpr_device" if k == "device" else k): v for k, v in rb.env_info(device).items()}


def generate_candidates(cfg: dict, store: DatasetStore, bpr: dict, force: bool = False,
                        splits: tuple[str, ...] = ("test", "longhist")) -> dict:
    n = cfg["candidates"]["n"]
    model, ds = bpr["model"], bpr["dataset"]
    id2tok = np.array(ds.field2id_token["item_id"], dtype=object)
    out = {}
    users_of = {"test": store.splits["test_users"]}
    if "longhist" in splits and cfg.get("longhist", {}).get("enabled", True):
        users_of["longhist"] = longhist_users(cfg, store)
    for split, users in ((s, users_of[s]) for s in splits if s in users_of):
        recs = []
        for i in range(0, len(users), 256):
            batch = users[i:i + 256]
            scores = rb.bpr_full_scores(model, ds, batch)
            for u, s in zip(batch, scores):
                toks, sc = id2tok[1:], s[1:]  # index 0 is RecBole's [PAD]
                recs.append(build_record(u, store.test_target(u), store.history_before_test(u), toks, sc, n,
                                         cfg["candidates"]["shuffle_seed"], store.name, split))
        recall = float(np.mean([r["r_u"] for r in recs])) if recs else 0.0
        meta = {"dataset": store.name, "split": split, "n": n, "bpr_recall_at_n": recall,
                "bpr_tuning": bpr["tuning"], "bpr_final_epochs": bpr["best_epoch"] + 1,
                "shuffle_seed": cfg["candidates"]["shuffle_seed"], "mode": cfg["mode"],
                "bpr_valid_metric": cfg["retriever"].get("valid_metric"),
                "bpr_seed": cfg["seed"], "bpr_reproducibility": True, **_env(bpr.get("device"))}
        if split == "longhist":
            meta["longhist"] = longhist_meta(cfg, recs)
        freeze(cfg, store.name, split, recs, meta, force=force)
        out[split] = meta
        log.info("%s/%s: BPR recall@%d = %.3f", store.name, split, n, recall)
    return out


def run(cfg: dict, dataset: str, force: bool = False) -> dict:
    if candidates_path(cfg, dataset).exists() and not force:
        log.info("Candidates for %s are frozen; skipping (use --force to regenerate).", dataset)
        if cfg.get("longhist", {}).get("enabled", True) and not candidates_path(cfg, dataset, "longhist").exists():
            return run_longhist(cfg, dataset)
        return {}
    t0 = time.perf_counter()
    store = DatasetStore.load(cfg, dataset)
    bpr = train_bpr(cfg, store)
    meta = generate_candidates(cfg, store, bpr, force=force)
    write_json(path(cfg, "training", "bpr", f"{dataset}.json"), {"tuning": bpr["tuning"], "history": bpr["history"],
                                                             "final_epochs": bpr["best_epoch"] + 1,
                                                             "seed": cfg["seed"], **_env(bpr.get("device"))})
    # keep the final retriever so further candidate samples (e.g. RQ2 long-history) use the SAME BPR model
    import torch

    torch.save(bpr["model"].state_dict(), state_path(cfg, dataset))
    best = bpr["tuning"]["best"]
    n_users = {"test": len(store.splits.get("test_users", []))}
    if "longhist" in meta:
        n_users["longhist"] = int(cfg["longhist"]["n_users"])
    report(cfg, "candidates", dataset=dataset,
           tuning_runs=[{**t["params"], "valid": t["best_valid"], "best_epoch": t["best_epoch"],
                         "minutes": t.get("minutes")} for t in bpr["tuning"]["trials"]],
           best_params=best["params"], best_valid=best["best_valid"],
           best_epoch=best["best_epoch"], final_epochs=bpr["best_epoch"] + 1, users=n_users,
           bpr_recall_at_20={s: m["bpr_recall_at_n"] for s, m in meta.items()},
           minutes=(time.perf_counter() - t0) / 60, files=str(candidates_path(cfg, dataset).parent))
    return meta


def state_path(cfg: dict, dataset: str):
    p = path(cfg, "training", "bpr", f"{dataset}.final.pt")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def run_longhist(cfg: dict, dataset: str, force: bool = False) -> dict:
    """Generate + freeze only the long-history candidates with the saved final BPR model."""
    from ..utils import read_json

    if candidates_path(cfg, dataset, "longhist").exists() and not force:
        log.info("Long-history candidates for %s are frozen; skipping.", dataset)
        return {}
    sp = state_path(cfg, dataset)
    if not sp.exists():
        raise FileNotFoundError(f"{sp} missing: the final BPR model was not saved. Run `make candidates` "
                                f"(it trains BPR and freezes the test and long-history candidates together).")
    store = DatasetStore.load(cfg, dataset)
    meta_test = read_json(candidates_path(cfg, dataset).with_suffix(".meta.json"))
    best = meta_test["bpr_tuning"]["best"]["params"]
    rcfg = cfg["retriever"]
    device = pick_device(rcfg.get("device", "auto"))
    device = "cpu" if device == "mps" else device
    work = path(cfg, "training", "recbole_data", store.name, "full")
    rb.write_atomic(work, store.name, store.sequences, store.all_items)
    split = "test_only" if rcfg.get("final_train_includes_valid", True) else "valid_and_test"
    bpr = rb.load_trained("BPR", work, store.name, best, rcfg, split, cfg["seed"], device, sp)
    bpr.update(tuning=meta_test["bpr_tuning"], best_epoch=meta_test["bpr_final_epochs"] - 1, device=device)
    return generate_candidates(cfg, store, bpr, force=force, splits=("longhist",))
