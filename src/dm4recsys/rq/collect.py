"""Load RQ1 'Real' rows and RQ2/RQ3 condition rows into per-user frames."""
from __future__ import annotations

import logging

import pandas as pd

from ..config import model_cfg
from ..eval.metrics import list_tie_groups, tie_groups, with_cutoff
from ..retrieval.candidates import load_candidates
from ..run.runner import result_path
from ..utils import read_jsonl
from .conditions import REAL, RERUN, all_conditions
from .runner import cond_path, longhist_lengths, rerun_from_rq1

log = logging.getLogger(__name__)
COLS = ["dataset", "user", "model", "condition", "ndcg", "rank", "top1", "top1_id", "target", "target_position",
        "ranked_items", "options", "transform_log", "p_hit", "blocks", "refused"]


def _p_hit(r: dict, k: int) -> float:
    """Tie-aware Hit@k: the probability that the target is in the top k when the options tied
    with it are put in random order, i.e. the share of its possible ranks h+1 .. h+t that are <= k. Refused rows
    (target ranked last) give 0; rows without ties give the plain 0/1 hit."""
    t = r.get("tie") or {"higher": r["rank"] - 1, "tied": 1}
    h, n = int(t["higher"]), int(t["tied"])
    return sum(1 for j in range(h + 1, h + n + 1) if j <= k) / n


def _blocks(r: dict, id2item: dict) -> list[list[str]]:
    """The output ranking as blocks of tied items, best first (decision API's own choice as its own first block);
    a ranking without scores is one item per named option, then the unnamed options as one block. Used for tie-aware popularity by output rank (Fig. 4b)."""
    sc = r.get("scores")
    if isinstance(sc, dict) and sc:
        return [[id2item[o] for o in sorted(g, key=str)] for g in tie_groups(sc, (r.get("info") or {}).get("fixed_first"))]
    return [sorted(g, key=str) for g in list_tie_groups(r["ranked_items"], (r.get("info") or {}).get("n_found_valid"))]


def _frame(rows: list[dict], cfg: dict, cond: str, recs: dict | None = None) -> pd.DataFrame:
    k = cfg["eval"]["k"]
    out = []
    for r in with_cutoff(rows, k):
        opts = r.get("options")
        if opts is None:  # RQ1 rows: default display order and IDs from the frozen record
            rec = recs[r["user"]]
            opts = [[i, c] for i, c in zip(rec["option_ids"], rec["candidates"])]
        items = [c for _, c in opts]
        refused = bool((r.get("info") or {}).get("api_refusal"))  # no ranking: no top-1 (counts as a change),
        out.append({"dataset": r["dataset"], "user": r["user"], "model": r["model"], "condition": cond,  # target last
                    "ndcg": r[f"ndcg{k}"], "rank": len(items) if refused else r["rank"],
                    "top1": None if refused else r["ranked_items"][0], "top1_id": None if refused else r["order"][0],
                    "target": r["target"], "target_position": r.get("target_position", items.index(r["target"]) + 1),
                    "ranked_items": r["ranked_items"], "options": dict(opts),
                    "transform_log": r.get("transform_log", {}),
                    "p_hit": _p_hit(r, k), "blocks": _blocks(r, dict(opts)), "refused": refused})
    return pd.DataFrame(out, columns=COLS)


class RQData:
    """Lazy cache of per-user frames keyed by (model, condition)."""

    def __init__(self, cfg: dict, datasets: list[str]):
        self.cfg, self.datasets = cfg, datasets
        self.recs = {d: {r["user"]: r for r in load_candidates(cfg, d)[0]} for d in datasets}
        conds = all_conditions(cfg)
        self._split = {c["name"]: c["split"] for c in conds}  # length_* -> longhist
        self._length = {c["name"]: c.get("length") for c in conds}
        self._users: dict = {}
        self._cache: dict = {}

    def expected_users(self, d: str, cond: str) -> set:
        """Users a complete condition has: the test users, or the long-history sample for length_* conditions."""
        split = self._split.get(cond, "test")
        if split == "test":
            return set(self.recs[d])
        if (d, split) not in self._users:
            self._users[(d, split)] = {r["user"] for r in load_candidates(self.cfg, d, split)[0]}
        return self._users[(d, split)]

    def get(self, model: str, cond: str, datasets: list[str] | None = None) -> pd.DataFrame:
        key = (model, cond, tuple(datasets or self.datasets))
        if key not in self._cache:
            frames = []
            for d in datasets or self.datasets:
                if cond == REAL:
                    rows = read_jsonl(result_path(self.cfg, d, model, int(self.cfg["rq"]["reference_run"])))
                elif cond == RERUN and rerun_from_rq1(self.cfg, model_cfg(self.cfg, model)):
                    rows = read_jsonl(result_path(self.cfg, d, model, 1))
                else:
                    rows = read_jsonl(cond_path(self.cfg, d, model, cond))
                frames.append(_frame(rows, self.cfg, cond, self.recs[d]))
            self._cache[key] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
        return self._cache[key]

    def complete(self, model: str, cond: str) -> bool:
        df = self.get(model, cond)
        for d in self.datasets:
            if self._split.get(cond) == "longhist" and self._length.get(cond) not in longhist_lengths(self.cfg, d):
                continue  # above this dataset's history-length cap: not run on purpose
            got, want = set(df.loc[df["dataset"] == d, "user"]), self.expected_users(d, cond)
            if got != want:
                log.warning("%s/%s/%s: %d of %d users (incomplete; paired analyses use the users present)",
                            model, d, cond, len(got & want), len(want))
                return False
        return True
