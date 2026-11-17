"""Build any zero-shot ranker from its config name."""
from __future__ import annotations

from ..config import model_cfg
from .base import Ranker


def build(name: str, cfg: dict, stores: dict | None = None) -> Ranker:
    mcfg = model_cfg(cfg, name)
    group = mcfg["group"]
    if group == "decision":
        from .decision import build as b

        return b(mcfg, cfg)
    if group == "llm":
        from .llm import build as b

        return b(mcfg, cfg)
    if group == "plm":
        from .plm import build as b

        return b(mcfg, cfg)
    if group == "reference":
        from .reference import BPRRank, Popularity, RandomRank

        if name == "bpr":
            return BPRRank(mcfg, cfg)
        if name == "popularity":
            if stores is None:
                raise ValueError("popularity needs dataset stores")
            return Popularity(mcfg, cfg, {k: s.popularity for k, s in stores.items()})
        if name == "random":
            return RandomRank(mcfg, cfg)
    raise ValueError(f"Cannot build model {name} (group {group}); sequential models use scripts/train_sequential.py")
