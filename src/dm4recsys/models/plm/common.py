from __future__ import annotations

from ...models.decision.systemone import split_context
from ...prompts.render import decision_context, pointwise_question
from ...utils import pick_device
from ..base import RankRequest, RankResult, Ranker, rank_from_scores


class PLMRanker(Ranker):
    """Scores every candidate against the user's history; ranks by score (seeded ties).

    `history` = the decision models' state, word for word (the history block "A user has interacted with the
    following {noun}, in chronological order from oldest to newest:" + numbered list); candidates = the plain titles
    the decision models get as options. `instruction` (EmbSim, CrossEnc) = the model config's `instruction`,
    "Given a user's interaction history, retrieve the {item} the user will interact with next" (the
    retrieval-instruction style of both model cards); without one, the shared question without any reference to the
    options, since these models never see the candidate list: "Which {movie} is the user most likely to interact
    with next?". ItemCLS takes no instruction."""

    def __init__(self, mcfg: dict, cfg: dict):
        super().__init__(mcfg, cfg)
        self.device = pick_device(cfg["runs"].get("device", "auto"))

    def texts(self, req: RankRequest, history_titles: list[str] | None = None) -> tuple[str, str]:
        titles = req.history_titles if history_titles is None else history_titles
        ctx = decision_context(titles, req.item_noun, len(req.option_ids), self.cfg["prompt"].get("max_title_chars"),
                               include_history=req.extra.get("history_block", True))
        history, _ = split_context(ctx)  # the decision models' state
        noun = self.cfg["datasets"][req.dataset]["item_noun_singular"]
        tmpl = self.mcfg.get("instruction")  # per-model instruction (EmbSim, CrossEnc), {item} singular / {items} plural noun
        instr = tmpl.format(item=noun, items=req.item_noun) if tmpl else pointwise_question(noun)
        return history, instr

    def score(self, req: RankRequest, cands: list[str]) -> list[float]:
        raise NotImplementedError

    def rank(self, req: RankRequest) -> RankResult:
        self.current = req
        self.last_info = {}
        s = self.score(req, list(req.cand_titles))
        scores = dict(zip(req.option_ids, map(float, s)))
        order = rank_from_scores(scores, req.option_ids, self.name, req.dataset, req.user, req.run)
        return RankResult(order=order, scores=scores, model_version=self.model_version, info=dict(self.last_info))


def torch_dtype(mcfg: dict, device: str):
    """The configured dtype on CUDA (PLMs: float32); CPU/MPS always float32."""
    import torch

    name = mcfg.get("dtype")
    return getattr(torch, name) if name and str(device).startswith("cuda") else torch.float32
