from __future__ import annotations

from ...prompts.render import listwise_prompt
from ..base import RankRequest, RankResult, Ranker
from .parser import parse_ranking


class LLMRanker(Ranker):
    def __init__(self, mcfg: dict, cfg: dict, client):
        super().__init__(mcfg, cfg)
        self.client = client

    def messages(self, req: RankRequest) -> list[dict]:
        user = listwise_prompt(req.history_titles, req.cand_titles, req.option_ids, req.item_noun,
                               self.cfg["prompt"].get("max_title_chars"),
                               include_history=req.extra.get("history_block", True))  # RQ2 no history: as decision models
        system = self.cfg["prompt"].get("system")  # none by default: the decision models get no system text either
        return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]

    def rank(self, req: RankRequest) -> RankResult:
        out = self.client.chat(self.messages(req), run=req.run)
        self.log_raw(request=out["request"], response=out["raw"], latency_s=out["latency_s"],
                     usage=out["usage"], api_model_version=out.get("model_version"))
        pr = parse_ranking(out["text"], req.option_ids, self.name, req.dataset, req.user, req.run,
                           min_ranked=int(self.cfg["eval"]["k"]),  # compliant = top k named by the model
                           titles=req.cand_titles, max_title_chars=self.cfg["prompt"].get("max_title_chars"))
        info = {"complete": pr.complete, "n_found_valid": pr.n_found_valid, "n_by_title": pr.n_by_title,
                "duplicates": pr.duplicates, "invalid": pr.invalid,
                "n_missing": len(pr.missing), "parse_mode": pr.mode, "text": out["text"],
                "latency_attempt_s": out["latency_s"], "attempts": out.get("attempts"),
                "finish_reason": out.get("finish_reason"), "stop_reason": out.get("stop_reason"),
                "thinking_chars": out.get("thinking_chars"),
                "reasoning_tokens": (out.get("usage") or {}).get("reasoning_tokens")}
        return RankResult(order=pr.order, compliant=pr.compliant, usage=out["usage"],
                          model_version=out.get("model_version") or self.model_version, info=info)
