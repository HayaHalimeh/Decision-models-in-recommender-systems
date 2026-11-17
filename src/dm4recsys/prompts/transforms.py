"""Pluggable history and candidate transforms, applied before any model sees the input.

`select_history` is the ONLY function that decides which history items a model sees. RQ1 uses
[last_k(10)]. RQ2 will register shuffle / reverse / replace / truncate here; RQ3 will register
candidate rotations and ID reassignment in CANDIDATE_TRANSFORMS. Each transform is a pure function
of (items, rec, params, ctx) so it can be seeded per user.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..utils import rng_for

HistoryTransform = Callable[[list[str], dict, dict, dict], list[str]]
CandidateTransform = Callable[[list[str], list[str], dict, dict, dict], tuple[list[str], list[str]]]

HISTORY_TRANSFORMS: dict[str, HistoryTransform] = {}
CANDIDATE_TRANSFORMS: dict[str, CandidateTransform] = {}


def register_history(name: str):
    def deco(fn):
        HISTORY_TRANSFORMS[name] = fn
        return fn
    return deco


def register_candidates(name: str):
    def deco(fn):
        CANDIDATE_TRANSFORMS[name] = fn
        return fn
    return deco


@register_history("last_k")
def last_k(items: list[str], rec: dict, params: dict, ctx: dict) -> list[str]:
    """Up to k most recent items, oldest to newest."""
    k = int(params["k"])
    return list(items[-k:]) if k > 0 else []


@register_history("identity")
def identity_history(items, rec, params, ctx):
    return list(items)


# ctx keys used below: dataset, seed, recs (all frozen records of this dataset/split, by user),
# all_items (catalogue), log (dict the transform may write diagnostics into; stored with the result row).

def _log(ctx: dict) -> dict:
    return ctx.setdefault("log", {})


def other_user_partner(rec: dict, n_items: int, ctx: dict) -> tuple[str, list[str], int]:
    """Random partner from the same dataset's frozen records (fixed per-user seed). A draw is rejected if the
    partner's history is shorter than `n_items` or its `n_items` most recent items contain this user's target."""
    users = sorted(u for u in ctx["recs"] if u != rec["user"])
    order = rng_for(ctx["seed"], "other_user", ctx["dataset"], rec["user"]).permutation(len(users))
    redraws = 0
    for j in order:
        v = ctx["recs"][users[j]]
        hist = v["history"][-n_items:] if n_items > 0 else []
        if len(v["history"]) < n_items or rec["target"] in hist:
            redraws += 1
            continue
        return v["user"], hist, redraws
    raise ValueError(f"no valid partner for user {rec['user']} in {ctx['dataset']}")


@register_history("other_user")
def other_user(items, rec, params, ctx):
    """The most recent items of another test user, truncated to this user's (shown) history length."""
    partner, hist, redraws = other_user_partner(rec, len(items), ctx)
    cands = set(rec["candidates"])
    _log(ctx).update(partner=partner, redraws=redraws,
                     overlap_candidates=sum(i in cands for i in hist),
                     overlap_own_history=len(set(hist) & set(items)))
    return list(hist)


@register_history("random_fake")
def random_fake(items, rec, params, ctx):
    """len(items) catalogue items, uniform without replacement, excluding the user's full history, the
    candidates and the target; random order."""
    banned = set(rec["history"]) | set(rec["candidates"]) | {rec["target"]}
    pool = [i for i in ctx["all_items"] if i not in banned]
    rng = rng_for(ctx["seed"], "random_fake", ctx["dataset"], rec["user"])
    if len(pool) < len(items):
        raise ValueError(f"user {rec['user']}: catalogue too small for a random fake history")
    return [pool[j] for j in rng.choice(len(pool), size=len(items), replace=False)]


@register_history("shuffle")
def shuffle(items, rec, params, ctx):
    """Same items in a random order (fixed per-user seed); identity for length 1."""
    if len(items) <= 1:
        _log(ctx)["shuffle_identity_len1"] = True
        return list(items)
    perm = rng_for(ctx["seed"], "shuffle", ctx["dataset"], rec["user"]).permutation(len(items))
    out = [items[j] for j in perm]
    if out == list(items):
        _log(ctx)["shuffle_identity"] = True
    return out


@register_history("reverse")
def reverse(items, rec, params, ctx):
    """Newest first. The prompt still says 'oldest to newest'."""
    return list(reversed(items))


@register_history("empty")
def empty(items, rec, params, ctx):
    return []


@register_candidates("identity")
def identity_candidates(cands, option_ids, rec, params, ctx):
    return list(cands), list(option_ids)



@register_candidates("rotate")
def rotate(cands, option_ids, rec, params, ctx):
    """Cyclic rotation by k: the item at position p moves to position ((p - 1 + k) mod n) + 1.
    Each item keeps its own ID, so the IDs move with their items."""
    k = int(params["k"]) % len(cands)
    return list(np.roll(np.array(cands, dtype=object), k)), list(np.roll(np.array(option_ids, dtype=object), k))


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random derangement by rejection (expected ~e draws)."""
    if n < 2:
        raise ValueError("no derangement of fewer than 2 elements")
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


@register_candidates("label_swap")
def label_swap(cands, option_ids, rec, params, ctx):
    """Positions unchanged; the IDs are reassigned with a derangement, so every item gets a new ID."""
    rng = rng_for(int(params["seed"]), "label_swap", ctx.get("dataset", rec["dataset"]), rec["user"])
    p = derangement(len(option_ids), rng)
    return list(cands), [option_ids[j] for j in p]


def select_history(rec: dict, transforms: list[dict], ctx: dict | None = None) -> list[str]:
    items = list(rec["history"])  # all interactions before the target, oldest -> newest
    for t in transforms:
        params = {k: v for k, v in t.items() if k != "name"}
        items = HISTORY_TRANSFORMS[t["name"]](items, rec, params, ctx or {})
    assert rec["target"] not in items, "history must never contain the target"
    return items


def select_candidates(rec: dict, transforms: list[dict], ctx: dict | None = None) -> tuple[list[str], list[str]]:
    cands, ids = list(rec["candidates"]), list(rec["option_ids"])
    for t in transforms:
        params = {k: v for k, v in t.items() if k != "name"}
        cands, ids = CANDIDATE_TRANSFORMS[t["name"]](cands, ids, rec, params, ctx or {})
    return cands, ids
