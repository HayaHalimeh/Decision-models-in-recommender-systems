from __future__ import annotations

from ...utils import load_object
from .client import OpenAICompatClient
from .parser import parse_ranking
from .ranker import LLMRanker


def build(mcfg: dict, cfg: dict) -> LLMRanker:
    factory = mcfg.get("backend_factory")  # optional "module:function" returning a chat client
    client = load_object(factory)(mcfg, cfg) if factory else OpenAICompatClient(mcfg)
    return LLMRanker(mcfg, cfg, client)
