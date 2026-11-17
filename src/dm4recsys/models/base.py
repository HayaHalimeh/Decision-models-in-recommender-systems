"""Common request/result types and the Ranker interface used by every model."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils import rng_for


@dataclass
class RankRequest:
    dataset: str
    user: str
    history_items: list[str]          # after history transforms (oldest -> newest)
    history_titles: list[str]
    cand_items: list[str]             # after candidate transforms (display order)
    cand_titles: list[str]
    option_ids: list[str]             # IDs shown to the model, aligned with cand_items
    item_noun: str = "items"
    bpr_scores: list[float] | None = None
    run: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class RankResult:
    order: list[str]                              # option IDs, best first (always a full permutation)
    scores: dict[str, float] | None = None        # per option ID (probabilities for decision models)
    compliant: bool | None = None                 # LLMs: complete and valid ranking as produced
    usage: dict[str, Any] = field(default_factory=dict)   # input_tokens, output_tokens, calls
    model_version: str | None = None
    info: dict[str, Any] = field(default_factory=dict)


class Ranker(abc.ABC):
    def __init__(self, mcfg: dict, cfg: dict):
        self.mcfg, self.cfg = mcfg, cfg
        self.name = mcfg["name"]
        self.group = mcfg["group"]
        self.deployment = mcfg.get("deployment", "local")
        self.raw_logger = None  # set by the runner

    @property
    def deterministic(self) -> bool:
        return self.deployment == "local" and float(self.mcfg.get("temperature", 0) or 0) == 0

    @property
    def model_version(self) -> str:
        return str(self.mcfg.get("model_version") or self.mcfg.get("hf_model_id") or self.mcfg.get("model_id") or self.name)

    def log_raw(self, request: Any, response: Any, latency_s: float, **kw) -> None:
        if self.raw_logger is not None:
            self.raw_logger(request=request, response=response, latency_s=latency_s,
                            model_version=self.model_version, **kw)

    @abc.abstractmethod
    def rank(self, req: RankRequest) -> RankResult: ...

    def close(self) -> None:
        pass


def rank_from_scores(scores: dict[str, float], option_ids: list[str], *seed_parts) -> list[str]:
    """Sort descending; ties broken at random with a fixed seed."""
    rng = rng_for("tiebreak", *seed_parts)
    tie = rng.random(len(option_ids))
    vals = np.array([scores[o] for o in option_ids], dtype=float)
    idx = np.lexsort((tie, -vals))
    return [option_ids[i] for i in idx]
