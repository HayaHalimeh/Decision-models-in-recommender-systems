"""Jev (TypeSafe AI), hosted: POST https://jevtypesafeai.com/api/v1/decide, Bearer auth.

Request/response handling is shared with Microsoft-Decision-1 (`systemone.py`). Jev's documented body has no
`model` field. Response of jev-1.14.0:
    {"model": "jev-1.14.0", "answers": {name: {"type": "choice", "choice": <label>, "confidence": float,
     "probabilities": {label: float, ...}}}, "usage": {"input_tokens", "output_tokens", ...}}
All 20 labels come back; probabilities are rounded to 2 decimals, so low options tie at 0 (`info.n_zero_prob`).
"""
from __future__ import annotations

from .systemone import SystemOneDecisionModel, split_context  # noqa: F401  (split_context re-exported)

DEFAULT_ENDPOINT = "https://jevtypesafeai.com/api/v1/decide"


class JevDecisionModel(SystemOneDecisionModel):
    default_endpoint = DEFAULT_ENDPOINT
