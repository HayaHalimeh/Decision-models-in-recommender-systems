from __future__ import annotations

from ...utils import load_object
from .base import DecisionModel, DecisionRanker

HOSTED = {"jev": "jev:JevDecisionModel", "openai_decisions": "openai_decisions:OpenAIDecisionModel",
          "ms_decision1": "ms_decision:MSDecisionModel"}
# open decision models: native heads behind a local System One server (same request as Jev / MS-Decision-1)
LOCAL_SERVERS = {n: "local_systemone:LocalSystemOneModel" for n in ("clef_flash", "pplx_decider_flash", "kev4b")}


def build_backend(mcfg: dict, cfg: dict) -> DecisionModel:
    import importlib

    name = mcfg["name"]
    if mcfg.get("backend_factory"):  # optional "module:function" returning a backend object
        return load_object(mcfg["backend_factory"])(mcfg, cfg)
    spec = HOSTED.get(name) or LOCAL_SERVERS.get(name) or mcfg.get("class")
    if spec is None:
        raise KeyError(f"No backend registered for decision model {name}")
    mod, cls = spec.split(":")
    klass = getattr(importlib.import_module(f"{__name__}.{mod}"), cls)
    return klass(mcfg)


def build(mcfg: dict, cfg: dict) -> DecisionRanker:
    return DecisionRanker(mcfg, cfg, build_backend(mcfg, cfg))
