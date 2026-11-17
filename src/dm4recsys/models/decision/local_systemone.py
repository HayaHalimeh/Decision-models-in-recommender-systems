"""Open decision models served locally with their own native heads, all behind the System One API
(`POST /v1/systemone`, the same body as Jev / MS-Decision-1: state + questions{choice, instructions, criteria}):

* Clef-flash (Cloudflare/clef-flash): frozen Qwen3.5-9B backbone + joint schema head; the checkpoint's
  `joint_schema_model.systemone` is wrapped as an HTTP `/v1/systemone` endpoint.
* Perplexity Decider flash v1.2 (perplexity-ai/pplx-decider-flash-v1.2): full fine-tune of Qwen3.5-4B with a readout
  head and noncausal full attention; its package exposes `Decider.decide(request)` with the same request schema,
  wrapped as an HTTP `/v1/systemone` endpoint.
* Kev-4B (jaredpalmer/kev-4b): LoRA + pointer head on Qwen3.5-4B-Base; `python -m kev.serve` implements
  TypeSafe's System One API (model card, "How to use").

None of them reads out through an lm_head we could score ourselves (the Decider has none at all), so they are
always called through their own heads.
"""
from __future__ import annotations

from .systemone import SystemOneDecisionModel


class LocalSystemOneModel(SystemOneDecisionModel):
    def extra_fields(self) -> dict:
        return {"model": self.mcfg["api_model"]} if self.mcfg.get("api_model") else {}
