"""Resumable runner: one JSONL line per (model, dataset, run, user) plus raw request/response logs."""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..config import model_cfg, path
from ..data.store import DatasetStore
from ..eval.metrics import expected_ndcg, list_tie_rank, ndcg_at_k, tie_aware_ndcg, tie_rank
from ..models.base import RankRequest, RankResult, Ranker
from ..prompts.request import build_request
from ..retrieval.candidates import load_candidates
from ..utils import JsonlAppender, read_jsonl, write_json
from .progress import report

log = logging.getLogger(__name__)


def result_path(cfg: dict, dataset: str, model: str, run: int) -> Path:
    """Results: runs/{mode}/{ds}/{model}/run{r}.jsonl (read by the report)."""
    return path(cfg, "runs", dataset, model, f"run{run}.jsonl")


def raw_log_path(cfg: dict, dataset: str, model: str, run: int) -> Path:
    return path(cfg, "logs", dataset, model, f"run{run}.raw.jsonl")


def n_repeats(cfg: dict, mcfg: dict) -> int:
    if mcfg.get("repeats"):
        return int(mcfg["repeats"])
    if mcfg.get("deployment") == "api":
        return int(cfg["runs"]["repeats_api"])
    if float(mcfg.get("temperature", 0) or 0) > 0:
        return int(cfg["runs"]["repeats_sampling"])
    return int(cfg["runs"]["repeats_local"])


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def ndcg_fields(k: int, order: list[str], scores: dict | None, target_id: str, first: str | None,
                refused: bool = False, n_ranked: int | None = None) -> dict:
    """Metric fields of one result row:
    ndcg{k}       tie-aware NDCG@k, the main metric for every model: the expected NDCG@k over all orders of the
                  options tied with the target. With a score vector (decision models, PLMs, sequential models,
                  references) ties are equal scores (the decision API's own choice fixed at rank 1, see fixed_first);
                  with a ranked list (LLMs) the options the model did not name (all after the first `n_ranked`) are
                  one tied block, so without such a block this is plain NDCG@k;
    ndcg{k}_order plain NDCG@k of the saved order (seeded tie-break), kept for the robustness comparison;
    tie           {"higher": h, "tied": t}: the target is equally likely at ranks h+1 .. h+t.
    refused: the API refused every attempt (decision APIs, after runs.refusal_retries reruns) -> NDCG@k = 0 and the
    target counts as ranked last; ndcg{k}_order keeps the value of the saved (seeded) order."""
    rank = order.index(target_id) + 1
    if refused:
        return {f"ndcg{k}": 0.0, f"ndcg{k}_order": ndcg_at_k(rank, k), "tie": {"higher": len(order) - 1, "tied": 1},
                "refusal_scored_zero": True}
    if isinstance(scores, dict) and set(scores) == set(order):
        h, t = tie_rank(scores, target_id, first)
        assert h < rank <= h + t, f"saved order puts the target at {rank}, scores say {h + 1}..{h + t}"
        nd = tie_aware_ndcg(scores, target_id, k, first)
    else:
        h, t = list_tie_rank(order, target_id, n_ranked)
        nd = expected_ndcg(h, t, k)
    return {f"ndcg{k}": nd, f"ndcg{k}_order": ndcg_at_k(rank, k), "tie": {"higher": h, "tied": t}}


def n_ranked(scores: dict | None, info: dict | None) -> int | None:
    """Ranked lists without scores (LLMs): how many options the model named itself (parser: n_found_valid)."""
    if isinstance(scores, dict) and scores:
        return None
    return (info or {}).get("n_found_valid")


def make_row(cfg: dict, rec: dict, req: RankRequest, res: RankResult, model: str, run: int, latency: float,
             cand_sha: str, **extra) -> dict:
    assert sorted(res.order) == sorted(req.option_ids), f"{model} returned an invalid permutation"
    id2item = dict(zip(req.option_ids, req.cand_items))
    ranked_items = [id2item[o] for o in res.order]
    rank = ranked_items.index(rec["target"]) + 1
    k = cfg["eval"]["k"]
    row = {
        "dataset": rec["dataset"], "user": rec["user"], "model": model, "run": run,
        "order": res.order, "ranked_items": ranked_items, "target": rec["target"], "rank": rank,
        **ndcg_fields(k, res.order, res.scores, res.order[rank - 1], (res.info or {}).get("fixed_first"),
                      bool((res.info or {}).get("api_refusal")), n_ranked=n_ranked(res.scores, res.info)),
        "r_u": rec["r_u"],
        "compliant": res.compliant, "scores": res.scores, "usage": res.usage, "latency_s": latency,
        "model_version": res.model_version, "candidates_sha256": cand_sha, "ts": now(), "info": res.info,
    }
    row.update(extra)
    return row


class RawLog:
    """Raw request/response log. The (dataset, user, run) context is per thread, so concurrent calls are labelled
    with their own user."""

    def __init__(self, p: Path, model: str):
        self.w = JsonlAppender(p)
        self.model = model
        self._local = threading.local()

    @property
    def ctx(self) -> dict:
        return getattr(self._local, "ctx", {})

    @ctx.setter
    def ctx(self, value: dict) -> None:
        self._local.ctx = value

    def __call__(self, **kw) -> None:
        self.w.write({"model": self.model, **self.ctx, "ts": now(), **kw})

    def close(self) -> None:
        self.w.close()


def _rank_timed(ranker: Ranker, req: RankRequest) -> tuple[RankResult, float]:
    t0 = time.perf_counter()
    res = ranker.rank(req)
    return res, time.perf_counter() - t0


def determinism_check(cfg: dict, ranker: Ranker, store: DatasetStore, recs: list[dict]) -> dict:
    n = int(cfg["runs"]["determinism_users"])
    a = [_rank_timed(ranker, build_request(cfg, store, r))[0].order for r in recs[:n]]
    b = [_rank_timed(ranker, build_request(cfg, store, r))[0].order for r in recs[:n]]
    same = [x == y for x, y in zip(a, b)]
    out = {"model": ranker.name, "dataset": store.name, "users": len(same), "identical": int(sum(same)), "ts": now()}
    write_json(path(cfg, "runs", store.name, ranker.name, "determinism.json"), out)
    if not all(same):
        raise AssertionError(
            f"{ranker.name} is not deterministic on {store.name}: {sum(same)}/{len(same)} identical. "
            f"Set `repeats: 3` for this model in its config.")
    return out


def report_model(cfg: dict, model: str, ds: str, runs: list[int], n_users: int) -> dict:
    """Progress report for one model on one dataset, from its saved result files (all runs)."""
    k = cfg["eval"]["k"]
    rows = [row for r in runs for row in read_jsonl(result_path(cfg, ds, model, r))]
    users = {row["user"] for row in rows}
    fields = {"model": model, "dataset": ds, "runs": len(runs), "users": f"{len(users)}/{n_users}"}
    if rows:
        fields[f"ndcg{k}"] = float(np.mean([row[f"ndcg{k}"] for row in rows]))
        fields[f"realistic_ndcg{k}"] = float(np.mean([row[f"ndcg{k}"] * row["r_u"] for row in rows]))
        fields["hit_at_1"] = float(np.mean([(row["tie"]["higher"] == 0) / row["tie"]["tied"] for row in rows]))
    fields["files"] = str(result_path(cfg, ds, model, 0).parent)
    return report(cfg, "model", **fields)


def check_concurrency(model: str, mcfg: dict) -> int:
    """concurrency: requests kept in flight at once. Each request is still one independent call
    (same prompt, own seed), so results do not depend on it; rows are written as answers arrive. Only for LLM
    rankers (stateless clients); decision backends keep per-call state and must run one call at a time."""
    conc = int(mcfg.get("concurrency", 1) or 1)
    if conc > 1 and mcfg["group"] != "llm":
        raise ValueError(f"{model}: concurrency > 1 is only supported for LLM rankers")
    return conc


def write_rows(w: JsonlAppender, todo: list, one, conc: int) -> None:
    """Calls one(rec) for every record and writes each row as soon as it is ready. conc > 1: that many calls in
    flight. On the first error no new calls are started, the ones in flight finish and are written, then the error
    is raised, so a failed call never leaves a row (a restart picks the user up)."""
    if conc == 1:
        for rec in todo:
            w.write(one(rec))
        return
    pool = ThreadPoolExecutor(max_workers=conc)
    futs, err = [pool.submit(one, rec) for rec in todo], None
    try:
        for f in as_completed(futs):
            if f.cancelled():
                continue
            try:
                w.write(f.result())
            except BaseException as e:  # first error: queue no new users, but keep
                if err is None:         # writing the calls that still succeed
                    err = e
                    for g in futs:
                        g.cancel()
    finally:  # Ctrl-C: drop queued users, let the ones in flight finish
        pool.shutdown(wait=True, cancel_futures=True)
    if err is not None:
        raise err


def run_model(cfg: dict, model: str, datasets: list[str], limit: int | None = None, ranker: Ranker | None = None,
              runs: list[int] | None = None, check_determinism: bool = True) -> dict:
    mcfg = model_cfg(cfg, model)
    stores = {d: DatasetStore.load(cfg, d) for d in datasets}
    if ranker is None:
        from ..models.registry import build

        ranker = build(model, cfg, stores)
    R = n_repeats(cfg, mcfg)
    conc = check_concurrency(model, mcfg)
    summary = {}
    try:
        for ds in datasets:
            recs, meta = load_candidates(cfg, ds)
            recs = recs[:limit] if limit else recs
            store = stores[ds]
            if check_determinism and R == 1 and ranker.deterministic and mcfg["group"] != "reference":
                determinism_check(cfg, ranker, store, recs)
            for r in runs if runs is not None else range(R):
                out = result_path(cfg, ds, model, r)
                done = {row["user"] for row in read_jsonl(out)}
                todo = [rec for rec in recs if rec["user"] not in done]
                if not todo:
                    log.info("%s/%s run %d: complete (%d users)", model, ds, r, len(done))
                    continue
                log.info("%s/%s run %d: %d done, %d to do", model, ds, r, len(done), len(todo))
                raw = RawLog(raw_log_path(cfg, ds, model, r), model)
                ranker.raw_logger = raw
                def one(rec, r=r, raw=raw):
                    raw.ctx = {"dataset": ds, "user": rec["user"], "run": r}
                    req = build_request(cfg, store, rec, run=r)
                    res, lat = _rank_timed(ranker, req)
                    # API models and servers: latency of the successful attempt (retries and back-off
                    # excluded); the whole call incl. retries is kept as latency_total_s
                    lat_api = (res.info or {}).get("latency_attempt_s")
                    return make_row(cfg, rec, req, res, model, r, lat if lat_api is None else lat_api,
                                    meta["sha256"], latency_total_s=lat)

                try:
                    with JsonlAppender(out) as w:
                        write_rows(w, todo, one, conc)
                finally:
                    raw.close()
                    ranker.raw_logger = None
            summary[ds] = {"runs": R, "users": len(recs)}
            report_model(cfg, model, ds, list(runs if runs is not None else range(R)), len(recs))
    finally:
        ranker.close()
    return summary
