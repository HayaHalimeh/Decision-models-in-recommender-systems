"""Turn a frozen candidate record into a RankRequest (applies history and candidate transforms)."""
from __future__ import annotations

from ..data.store import DatasetStore
from ..models.base import RankRequest
from .transforms import select_candidates, select_history


def build_request(cfg: dict, store: DatasetStore, rec: dict, run: int = 0,
                  history_transforms: list[dict] | None = None, candidate_transforms: list[dict] | None = None,
                  ctx: dict | None = None, extra: dict | None = None) -> RankRequest:
    """RQ1 calls this with defaults. RQ2/RQ3 conditions pass their own transforms, a ctx (dataset, seed,
    records, catalogue) and `extra` (e.g. history_block=False, call_key). Transform diagnostics end up in
    req.extra['transform_log']."""
    ht = history_transforms if history_transforms is not None else cfg["history"]["transforms"]
    ct = candidate_transforms if candidate_transforms is not None else (cfg["candidates"].get("transforms") or [])
    tctx = dict(ctx or {})
    tctx["log"] = {}
    hist = select_history(rec, ht, tctx)
    cands, ids = select_candidates(rec, ct, tctx)
    score_of = dict(zip(rec["candidates"], rec["bpr_scores"]))
    ex = dict(extra or {})
    if tctx["log"]:
        ex["transform_log"] = tctx["log"]
    return RankRequest(
        dataset=rec["dataset"],
        user=rec["user"],
        history_items=hist,
        history_titles=[store.title(i) for i in hist],
        cand_items=cands,
        cand_titles=[store.title(i) for i in cands],
        option_ids=ids,
        item_noun=cfg["datasets"][rec["dataset"]].get("item_noun", "items"),
        bpr_scores=[score_of[c] for c in cands],
        run=run,
        extra=ex,
    )
