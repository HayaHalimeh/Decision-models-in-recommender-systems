"""Microsoft-Decision-1 via OpenRouter, same System One body as Jev (`systemone.py`):
POST https://openrouter.ai/api/alpha/decisions, `Authorization: Bearer $OPENROUTER_API_KEY`,
`"model": "microsoft/microsoft-decision-1-20261009"` (dated, pinned version). OpenRouter reports
the upstream `provider` ("Azure"). OpenRouter may route to different upstream providers, so the provider is pinned
with `"provider": {"order": [...], "allow_fallbacks": false}` (`openrouter_provider` in the config); every row
stores the reported provider (`info.api_provider`) and a different one stops the run.
"""
from __future__ import annotations

from .systemone import SystemOneDecisionModel

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_MODEL = "microsoft/microsoft-decision-1-20261009"


class MSDecisionModel(SystemOneDecisionModel):
    def __init__(self, mcfg: dict):
        mcfg = dict(mcfg)
        if mcfg.get("provider", "openrouter") != "openrouter":
            raise ValueError("ms_decision1.provider must be 'openrouter'")
        mcfg["endpoint"] = mcfg.get("openrouter_endpoint", OPENROUTER_ENDPOINT)
        mcfg["api_key_env"] = mcfg.get("openrouter_api_key_env", "OPENROUTER_API_KEY")
        super().__init__(mcfg)
        self.api_model = mcfg.get("openrouter_model", OPENROUTER_MODEL)
        self.routing = mcfg.get("openrouter_provider")  # e.g. {"order": ["Azure"], "allow_fallbacks": False}

    def headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def extra_fields(self):
        if self.routing:
            return {"model": self.api_model, "provider": dict(self.routing)}
        return {"model": self.api_model}

    def parse_response(self, data, options, option_ids):
        probs = super().parse_response(data, options, option_ids)
        want = (self.routing or {}).get("order")
        got = data.get("provider")
        if want and got not in want:
            raise RuntimeError(f"{self.name}: OpenRouter routed to provider {got!r}, pinned {want} "
                               f"(allow_fallbacks: false was not honoured) -> stop")
        return probs
