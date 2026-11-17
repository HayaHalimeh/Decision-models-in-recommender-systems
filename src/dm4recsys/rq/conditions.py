"""Condition registry for RQ2 / RQ3. A condition is plain data: which frozen candidate split it uses, the
history and candidate transforms (prompts/transforms.py) and any prompt flags. Nothing else differs."""
from __future__ import annotations

import copy
import hashlib
import json

REAL = "real"      # the RQ1 default run (never rerun)
RERUN = "rerun"    # plain rerun of the default input: the noise floor shared by RQ2 and RQ3

RQ2_HISTORY = ["other_user", "random_fake", "no_history", "shuffled", "reversed"]
CONTENT = ["other_user", "random_fake", "no_history"]
ORDER = ["shuffled", "reversed"]
LABELS = {REAL: "Real", RERUN: "Rerun", "other_user": "Other user's", "random_fake": "Random fake",
          "no_history": "No history", "shuffled": "Shuffled", "reversed": "Reversed", "label_swap": "Label swap"}


def _cond(name, rq, history, candidates=None, split="test", extra=None, **kw) -> dict:
    return {"name": name, "rq": rq, "split": split, "history": history, "candidates": candidates or [],
            "extra": extra or {}, **kw}


def base_history(cfg: dict) -> list[dict]:
    return copy.deepcopy(cfg["history"]["transforms"])


def rq2_conditions(cfg: dict) -> list[dict]:
    b = base_history(cfg)
    out = [
        _cond(RERUN, "shared", b),
        _cond("other_user", "rq2", b + [{"name": "other_user"}]),
        _cond("random_fake", "rq2", b + [{"name": "random_fake"}]),
        _cond("no_history", "rq2", b + [{"name": "empty"}], extra={"history_block": False}),
        _cond("shuffled", "rq2", b + [{"name": "shuffle"}]),
        _cond("reversed", "rq2", b + [{"name": "reverse"}]),
    ]
    for L in cfg["longhist"]["lengths"]:
        out.append(_cond(f"length_{L:02d}", "rq2", [{"name": "last_k", "k": int(L)}], split="longhist", length=int(L)))
    return out


def rq3_conditions(cfg: dict) -> list[dict]:
    b = base_history(cfg)
    out = [_cond(RERUN, "shared", b)]
    for k in range(1, int(cfg["rq"]["n_rotations"])):
        out.append(_cond(rotation_name(k), "rq3", b, [{"name": "rotate", "k": k}], rotation=k))
    out.append(_cond("label_swap", "rq3", b, [{"name": "label_swap", "seed": int(cfg["rq"]["label_swap_seed"])}]))
    return out


def rotation_name(k: int) -> str:
    return f"rotation_{k:02d}"


def all_conditions(cfg: dict, rq: str = "all") -> list[dict]:
    out, seen = [], set()
    for c in (rq2_conditions(cfg) if rq in ("rq2", "all") else []) + (rq3_conditions(cfg) if rq in ("rq3", "all") else []):
        if c["name"] not in seen:
            seen.add(c["name"])
            out.append(c)
    return out


def call_key(cfg: dict, model: str, cond: dict, dataset: str, user: str) -> str:
    """Unique per (model, condition incl. its transforms and seeds, dataset, user): a response for one condition
    can never be served for another (rotation k, swap seed and pairing seed are all part of the key)."""
    blob = json.dumps({"model": model, "condition": cond["name"], "history": cond["history"],
                       "candidates": cond["candidates"], "extra": cond["extra"], "seed": cfg["rq"]["seed"],
                       "split": cond["split"], "dataset": dataset, "user": user}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]
