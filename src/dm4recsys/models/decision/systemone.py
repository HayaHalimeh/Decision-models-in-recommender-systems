"""Shared adapter for "System One"-style decision APIs: Jev (TypeSafe AI) and Microsoft-Decision-1 (Foundry).

Both documented request bodies are
    {["model": <deployment>,] "state": <text>, "questions": {<name>: {"type": "choice", "instructions": <question>,
                                                                      "criteria": {<label>: <description>, ...}}}}
(Jev docs; learn.microsoft.com "Deploy and use Microsoft-Decision-1 in Microsoft Foundry", 2026-10-07; Microsoft
adds `model` = the deployment name). We send the history block as `state`, the decision question as
`instructions`, and the 20 candidates as `criteria` with our option IDs ("1".."20") as labels and the item titles
as descriptions. So the input carries IDs and the RQ3 label swap applies.

Response: `answers[<name>]`. Microsoft states that a choice answer contains "the selected option and a
probability for every option" plus token usage, but neither provider shows the JSON. `parse_response` accepts
per-option scores (dict or list) and a bare choice, and records the shape it saw (`info.api_shape`). A bare
choice is mapped to one-hot and flagged `label_only`.
"""
from __future__ import annotations

from ...prompts.render import QUESTION_PREFIX
from .hosted import HostedDecisionModel

QUESTION_NAME = "next_item"
SCORE_KEYS = ("probabilities", "distribution", "scores", "confidences", "confidence", "options", "criteria")
LABEL_KEYS = ("choice", "label", "answer", "value", "selected", "result")


def split_context(context: str) -> tuple[str, str]:
    """decision_context() = state + blank line + 'Question: ...'. Returns (state, instruction without 'Question: ')."""
    if context.startswith(QUESTION_PREFIX):  # RQ2 'no history': no state at all
        return "", context.removeprefix(QUESTION_PREFIX)
    state, sep, q = context.rpartition("\n\n" + QUESTION_PREFIX)
    if not sep:
        raise ValueError("context does not end with the decision question")
    return state, q


class SystemOneDecisionModel(HostedDecisionModel):
    default_endpoint = ""

    def __init__(self, mcfg: dict):
        mcfg = dict(mcfg)
        if self.default_endpoint:
            mcfg.setdefault("endpoint", self.default_endpoint)
        super().__init__(mcfg)
        self.last_info: dict = {}
        self.last_model_version: str | None = None
        self.last_choice: str | None = None

    def build_payload(self, context, options, option_ids):
        state, question = split_context(context)
        if len(set(option_ids)) != len(option_ids):
            raise ValueError(f"{self.name}: option IDs must be unique (they are the criteria labels)")
        if state == "" and "empty_state" in self.mcfg:  # RQ2 no history: the API's accepted form of "empty"
            state = self.mcfg["empty_state"]
        return self.extra_fields() | {"state": state,
                "questions": {QUESTION_NAME: {"type": "choice", "instructions": question,
                                              "criteria": dict(zip(option_ids, options))}}}

    def parse_usage(self, data: dict) -> dict:
        u = data.get("usage") or {}
        return {"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"), "calls": 1}

    def extra_fields(self) -> dict:
        m = self.mcfg.get("api_model")  # Jev: pin the version (honoured by the API)
        return {"model": m} if m else {}

    def parse_response(self, data, options, option_ids):
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise ValueError(f"{self.name}: no 'answers' object in response: {list(data)[:10]}")
        ans = answers.get(QUESTION_NAME, answers if len(answers) != 1 else next(iter(answers.values())))
        probs, shape = self._scores(ans, option_ids)
        self.last_choice = str(ans.get("choice")) if isinstance(ans, dict) and ans.get("choice") is not None else None
        self.last_model_version = data.get("model")  # e.g. "jev-1.14.0"
        self.last_info = {"api_shape": shape}
        if data.get("provider") is not None:  # OpenRouter: the upstream provider that served the call
            self.last_info["api_provider"] = data["provider"]
        if isinstance(ans, dict) and "confidence" in ans:
            self.last_info["api_confidence"] = ans["confidence"]
        if probs is not None:  # Jev rounds to 2 decimals: zeros are tied, ordered by the seeded tie-break
            self.last_info["n_zero_prob"] = sum(1 for o in option_ids if not probs.get(o))
        if probs is None:
            raise ValueError(f"{self.name}: unrecognised answer shape: {str(ans)[:300]}")
        return probs

    @staticmethod
    def _scores(ans, option_ids) -> tuple[dict[str, float] | None, str]:
        ids = set(option_ids)
        if isinstance(ans, str) and ans in ids:
            return {ans: 1.0}, "label_only"
        if not isinstance(ans, dict):
            return None, "unknown"
        if ans and set(ans) <= ids and all(isinstance(v, (int, float)) for v in ans.values()):
            return {k: float(v) for k, v in ans.items()}, "label_to_score"
        for key in SCORE_KEYS:
            v = ans.get(key)
            if isinstance(v, dict) and v and all(isinstance(x, (int, float)) for x in v.values()):
                return {str(k): float(x) for k, x in v.items()}, f"{key}:dict"
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                out = {}
                for x in v:
                    lab = next((x[k] for k in ("label", "id", "option", "choice", "name") if k in x), None)
                    sc = next((x[k] for k in ("probability", "p", "score", "confidence", "weight") if k in x), None)
                    if lab is None or sc is None:
                        break
                    out[str(lab)] = float(sc)
                else:
                    return out, f"{key}:list"
        for key in LABEL_KEYS:
            v = ans.get(key)
            if isinstance(v, (str, int)) and str(v) in ids:
                return {str(v): 1.0}, "label_only"
        return None, "unknown"
