"""OpenAI Decisions, hosted: POST https://api.openai.com/v1/decisions, Bearer auth, model "gpt-6-luna" (the only
supported model; the same model is our GPT-6 Luna LLM ranker).

Format (developers.openai.com/api/docs/guides/decisions and the API reference); it differs from
the System One body of Jev / MS-Decision-1:
    request  {"model", "input": <text>, "questions": [{"type": "choice", "name", "instructions",
                                                       "choices": [{"value": <id>, "description": <text>}, ...]}]}
    response {"model", "answers": [{"type": "choice", "name", "choice", "confidence",
                                    "probabilities": [{"value", "probability"}, ...]}],
              "usage": {"input_tokens", "output_tokens", "total_tokens", ...}}
We send `input` = the history block (empty string under RQ2 *no history*; the schema allows minLength 0),
`instructions` = the decision question, and `value` = our option ID ("1".."20") with `description` = the title,
so the input carries IDs and the RQ3 label swap applies. 2-255 choices are allowed.
A question can come back as {"type": "refusal"}; we then return no scores (uniform after normalisation, seeded
tie-break) and flag `info.api_refusal`. No temperature/seed parameters exist.
"""
from __future__ import annotations

from .hosted import HostedDecisionModel
from .systemone import QUESTION_NAME, split_context

DEFAULT_ENDPOINT = "https://api.openai.com/v1/decisions"


class OpenAIDecisionModel(HostedDecisionModel):

    def __init__(self, mcfg: dict):
        mcfg = dict(mcfg)
        mcfg.setdefault("endpoint", DEFAULT_ENDPOINT)
        super().__init__(mcfg)
        self.model = mcfg.get("api_model", "gpt-6-luna")
        self.last_info: dict = {}
        self.last_model_version: str | None = None
        self.last_choice: str | None = None

    def build_payload(self, context, options, option_ids):
        state, question = split_context(context)
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("openai_decisions: option IDs must be unique (they are the choice values)")
        if state == "" and "empty_state" in self.mcfg:  # RQ2 no history
            state = self.mcfg["empty_state"]
        return {"model": self.model, "input": state,
                "questions": [{"type": "choice", "name": QUESTION_NAME, "instructions": question,
                               "choices": [{"value": i, "description": t} for i, t in zip(option_ids, options)]}]}

    def parse_response(self, data, options, option_ids):
        answers = data.get("answers")
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"openai_decisions: no 'answers' list in response: {list(data)[:10]}")
        ans = next((a for a in answers if a.get("name") == QUESTION_NAME), answers[0])
        self.last_model_version = data.get("model")
        self.last_choice = None if ans.get("choice") is None else str(ans["choice"])
        if ans.get("type") == "refusal":
            self.last_info = {"api_shape": "refusal", "api_refusal": True}
            return {}
        probs = {str(p["value"]): float(p["probability"]) for p in ans.get("probabilities") or []}
        if not probs:
            raise ValueError(f"openai_decisions: no probabilities in answer: {str(ans)[:300]}")
        self.last_info = {"api_shape": "probabilities:list", "api_confidence": ans.get("confidence"),
                          "n_zero_prob": sum(1 for o in option_ids if not probs.get(o))}
        return probs

    def parse_usage(self, data):
        u = data.get("usage") or {}
        cached = (u.get("input_tokens_details") or {}).get("cached_tokens")
        return {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens", 0),
                "cached_tokens": cached, "calls": 1}
