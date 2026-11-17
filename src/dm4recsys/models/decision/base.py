"""Decision-model adapter protocol and the ranker that wraps any backend."""
from __future__ import annotations

import logging
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from ...prompts.render import decision_context
from ..base import RankRequest, RankResult, Ranker, rank_from_scores

log = logging.getLogger(__name__)


@runtime_checkable
class DecisionModel(Protocol):
    def choose(self, context: str, options: list[str], option_ids: list[str]) -> dict[str, float]: ...


def check_answer(raw: dict[str, float], option_ids: list[str], choice: str | None, tol: float = 0.02) -> list[str]:
    """Answer checks: the probability keys equal the options,
    every value is finite and in [0, 1], they sum to 1 within `tol`, and the API's choice is one of the options.
    Returns the failed checks (empty = valid). We record failures instead of discarding the answer."""
    bad = []
    if set(raw) != set(option_ids):
        bad.append("keys_differ_from_options")
    vals = [float(v) for v in raw.values()]
    if any(not np.isfinite(v) or v < 0 or v > 1 for v in vals):
        bad.append("value_outside_0_1")
    if vals and abs(sum(vals) - 1) >= tol:
        bad.append("sum_not_1")
    if choice is not None and choice not in option_ids:
        bad.append("choice_not_an_option")
    return bad


def fixed_first(probs: dict[str, float], choice: str | None, near_tie: float) -> str | None:
    """The option the near-tie rule puts at rank 1 whatever the tie-break: the API's own choice if its probability
    is within `near_tie` of the highest, else None. Tie-aware NDCG treats this option as rank 1 for certain."""
    if choice is None or choice not in probs or probs[choice] < max(probs.values()) - near_tie - 1e-9:
        return None
    return choice


def choice_first(order: list[str], probs: dict[str, float], choice: str | None, near_tie: float) -> tuple[list[str], bool]:
    """Near-tie rule: the API's own choice is taken as the top option when its
    probability is within `near_tie` of the highest (2-decimal rounding makes such ties common). Returns
    (order, near_tie_applied); the remaining order is unchanged."""
    if order[0] == choice or fixed_first(probs, choice, near_tie) is None:
        return order, False
    return [choice] + [o for o in order if o != choice], True


def normalise(probs: dict[str, float], option_ids: list[str]) -> tuple[dict[str, float], dict]:
    """Map onto exactly `option_ids`; missing -> 0, negatives clipped, renormalised. Returns (probs, info)."""
    info = {}
    missing = [o for o in option_ids if o not in probs]
    extra = [k for k in probs if k not in set(option_ids)]
    if missing:
        info["missing_options"] = missing
    if extra:
        info["unknown_options"] = extra
    p = np.array([max(0.0, float(probs.get(o, 0.0))) for o in option_ids])
    s = p.sum()
    p = p / s if s > 0 else np.full(len(p), 1.0 / len(p))
    return dict(zip(option_ids, p.tolist())), info


class DecisionRanker(Ranker):
    """Ranks candidates by the decision model's probability per option (descending, seeded ties)."""

    def __init__(self, mcfg: dict, cfg: dict, backend: DecisionModel):
        super().__init__(mcfg, cfg)
        self.backend = backend
        if hasattr(backend, "set_logger"):
            backend.set_logger(self.log_raw)

    def rank(self, req: RankRequest) -> RankResult:
        ctx = decision_context(req.history_titles, req.item_noun, len(req.option_ids), self.cfg["prompt"].get("max_title_chars"),
                               include_history=req.extra.get("history_block", True))
        if hasattr(self.backend, "run"):
            self.backend.run = req.run
        if hasattr(self.backend, "salt"):
            self.backend.salt = req.extra.get("call_key", "")  # RQ2/RQ3: condition-specific call key
        # refusal: re-send the same call up to runs.refusal_retries times; if every attempt is
        # refused the row keeps api_refusal and scores NDCG@k = 0 (run/runner.py: ndcg_fields)
        retries = int(self.cfg.get("runs", {}).get("refusal_retries", 0))
        for n_calls in range(1, retries + 2):
            raw = self.backend.choose(ctx, list(req.cand_titles), list(req.option_ids))
            if not (getattr(self.backend, "last_info", {}) or {}).get("api_refusal"):
                break
        probs, info = normalise(raw, req.option_ids)
        info.update(getattr(self.backend, "last_info", {}) or {})  # e.g. which response shape Jev returned
        if n_calls > 1 or info.get("api_refusal"):
            info["refusal_calls"] = n_calls  # calls made for this row (1 + reruns after a refusal)
        order = rank_from_scores(probs, req.option_ids, self.name, req.dataset, req.user, req.run)
        choice = getattr(self.backend, "last_choice", None)  # the API's own choice, if it reports one
        if raw and hasattr(self.backend, "last_choice"):
            failed = check_answer(raw, list(req.option_ids), choice, float(self.mcfg.get("prob_sum_tol", 0.02)))
            if failed:
                info["answer_check_failed"] = failed
                log.warning("%s %s/%s: answer check failed: %s", self.name, req.dataset, req.user, failed)
        near_tie = float(self.mcfg.get("near_tie", 0.01))
        order, nt = choice_first(order, probs, choice, near_tie)
        if choice is not None:
            info["api_choice"] = choice
            info["near_tie"] = nt
            info["fixed_first"] = fixed_first(probs, choice, near_tie)  # rank 1 for certain (tie-aware NDCG)
        if getattr(self.backend, "last_latency", None) is not None:  # real APIs / servers: per-attempt latency
            info["latency_attempt_s"] = self.backend.last_latency
            info["attempts"] = self.backend.last_attempts
        usage = dict(getattr(self.backend, "last_usage", {}) or {})
        usage.setdefault("calls", 1)
        if n_calls > 1:
            usage["calls"] = n_calls
        version = getattr(self.backend, "last_model_version", None) or self.model_version  # API-reported if any
        return RankResult(order=order, scores=probs, usage=usage, model_version=version, info=info)


LogFn = Callable[..., None]
