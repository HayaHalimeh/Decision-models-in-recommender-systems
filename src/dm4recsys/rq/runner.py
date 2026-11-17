"""Resumable per-condition runner for RQ2/RQ3: one JSONL file per (dataset, model, condition)."""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import model_cfg, path
from ..data.store import DatasetStore
from ..models.base import Ranker
from ..prompts.request import build_request
from ..retrieval.candidates import load_candidates
from ..run.runner import RawLog, _rank_timed, check_concurrency, make_row, write_rows
from ..utils import JsonlAppender, read_jsonl
from .conditions import RERUN, call_key

log = logging.getLogger(__name__)
# decision models and LLM rankers; rq.models lists which models run
RQ_GROUPS = ("decision", "llm")


def cond_path(cfg: dict, dataset: str, model: str, cond: str) -> Path:
    """Results: runs/{mode}/rq/{ds}/{model}/{cond}.jsonl (read by the RQ report)."""
    return path(cfg, "runs", "rq", dataset, model, f"{cond}.jsonl")


def cond_raw_path(cfg: dict, dataset: str, model: str, cond: str) -> Path:
    return path(cfg, "logs", "rq", dataset, model, f"{cond}.raw.jsonl")


def rerun_from_rq1(cfg: dict, mcfg: dict) -> bool:
    return cfg["rq"].get("rerun_source", "new") == "rq1_run1" and mcfg.get("deployment") == "api"


def request_run(cfg: dict, mcfg: dict, cond: dict) -> int:
    """Run index passed to the request (it sets a seeded LLM's request seed = request_seed + run). Every condition
    keeps Real's (RQ1 run 0), as everything else stays the same. Seeded LLMs (Qwen3.5, seed_requests) get a seed no
    RQ1 run used for the rerun, so the noise floor is a fresh sample: with Real's seed it would repeat Real's sample
    and understate the sampling noise (rq.rerun_seed_run)."""
    if cond["name"] == RERUN and mcfg.get("seed_requests"):
        return int(cfg["rq"].get("rerun_seed_run", 3))
    return 0


def transform_ctx(cfg: dict, store: DatasetStore, recs: list[dict]) -> dict:
    return {"dataset": store.name, "seed": int(cfg["rq"]["seed"]), "recs": {r["user"]: r for r in recs},
            "all_items": store.all_items}


def longhist_lengths(cfg: dict, dataset: str) -> list[int]:
    _, meta = load_candidates(cfg, dataset, "longhist")
    return list(meta["longhist"]["lengths"])


def run_condition(cfg: dict, model: str, cond: dict, datasets: list[str], max_users: int | None = None,
                  ranker: Ranker | None = None, stores: dict | None = None) -> dict:
    """One condition for one model on every dataset (resumable; max_users = the first N frozen users, a pilot)."""
    mcfg = model_cfg(cfg, model)
    if mcfg["group"] not in RQ_GROUPS:
        raise ValueError(f"RQ2/RQ3 run decision models and LLM rankers only, not {model} ({mcfg['group']})")
    conc = check_concurrency(model, mcfg)
    req_run = request_run(cfg, mcfg, cond)
    stores = stores or {d: DatasetStore.load(cfg, d) for d in datasets}
    own = ranker is None
    summary = {}
    if cond["name"] == RERUN and rerun_from_rq1(cfg, mcfg):
        log.info("%s: rerun taken from RQ1 run 1 (rq.rerun_source = rq1_run1)", model)
        return {d: "rq1_run1" for d in datasets}
    try:
        for ds in datasets:
            if cond["split"] == "longhist" and cond["length"] not in longhist_lengths(cfg, ds):
                log.info("%s/%s %s: above this dataset's history-length cap, not run", model, ds, cond["name"])
                summary[ds] = "above_cap"
                continue
            recs, meta = load_candidates(cfg, ds, cond["split"])
            store = stores[ds]
            ctx = transform_ctx(cfg, store, recs)  # pairing pool = ALL frozen records, also when --max-users is set
            todo_recs = recs[:max_users] if max_users else recs
            out = cond_path(cfg, ds, model, cond["name"])
            done = {row["user"] for row in read_jsonl(out)}
            todo = [r for r in todo_recs if r["user"] not in done]
            if not todo:
                summary[ds] = len(done)
                continue
            if ranker is None:
                from ..models.registry import build

                ranker = build(model, cfg, stores)
            log.info("%s/%s %s: %d done, %d to do", model, ds, cond["name"], len(done), len(todo))
            raw = RawLog(cond_raw_path(cfg, ds, model, cond["name"]), model)
            ranker.raw_logger = raw
            try:
                def one(rec, ds=ds, store=store, ctx=ctx, meta=meta, raw=raw, req_run=req_run):
                    # conc > 1 (LLM rankers): calls run in threads; the raw log context is per thread
                    key = call_key(cfg, model, cond, ds, rec["user"])
                    raw.ctx = {"dataset": ds, "user": rec["user"], "condition": cond["name"], "call_key": key}
                    req = build_request(cfg, store, rec, run=req_run, history_transforms=cond["history"],
                                        candidate_transforms=cond["candidates"], ctx=ctx,
                                        extra=dict(cond["extra"], call_key=key))
                    res, lat = _rank_timed(ranker, req)
                    return make_row(cfg, rec, req, res, model, 0, lat, meta["sha256"],
                                    condition=cond["name"], call_key=key, split=cond["split"], trial=False,
                                    history_shown=req.history_items,
                                    options=[[i, c] for i, c in zip(req.option_ids, req.cand_items)],
                                    target_position=req.cand_items.index(rec["target"]) + 1,
                                    target_id=req.option_ids[req.cand_items.index(rec["target"])],
                                    transform_log=req.extra.get("transform_log", {}),
                                    request_run=req_run)

                with JsonlAppender(out) as w:
                    write_rows(w, todo, one, conc)
            finally:
                raw.close()
                ranker.raw_logger = None
            summary[ds] = len(done) + len(todo)
    finally:
        if own and ranker is not None:
            ranker.close()
    return summary
