"""Shared HTTP skeleton for hosted decision-model APIs.

Request/response logging, retries, latency and token accounting are implemented here. The three
subclasses only fill in `build_payload` and `parse_response`.
"""
from __future__ import annotations

import os
import time

import httpx


class HostedDecisionModel:
    def __init__(self, mcfg: dict):
        self.mcfg = mcfg
        self.name = mcfg["name"]
        self.endpoint = os.environ.get(mcfg.get("endpoint_env", ""), "") or mcfg.get("endpoint", "")
        self.api_key = os.environ.get(mcfg.get("api_key_env", ""), "")
        missing = [v for v, ok in ((mcfg.get("endpoint_env") or "endpoint", self.endpoint),
                                   (mcfg.get("api_key_env"), self.api_key))
                   if v and not ok]  # no api_key_env = local server without auth
        if missing:
            raise RuntimeError(f"{self.name}: set {' and '.join(missing)} in .env")
        self.client = httpx.Client(timeout=mcfg.get("timeout_s", 60))
        self.max_retries = int(mcfg.get("max_retries", 5))
        self.last_usage: dict = {}
        self.last_latency: float | None = None  # seconds of the successful attempt (no retries, no back-off)
        self.last_attempts: int | None = None
        self._log = None

    def set_logger(self, fn) -> None:
        self._log = fn

    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def build_payload(self, context: str, options: list[str], option_ids: list[str]) -> dict:
        """The API's request body (implemented per API)."""
        raise NotImplementedError

    def parse_response(self, data: dict, options: list[str], option_ids: list[str]) -> dict[str, float]:
        """Option ID -> probability from the API's response (implemented per API)."""
        raise NotImplementedError

    def parse_usage(self, data: dict) -> dict:
        u = data.get("usage") or {}
        return {"input_tokens": u.get("input_tokens", u.get("prompt_tokens")),
                "output_tokens": u.get("output_tokens", u.get("completion_tokens", 0)), "calls": 1}

    def choose(self, context: str, options: list[str], option_ids: list[str]) -> dict[str, float]:
        payload = self.build_payload(context, options, option_ids)
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = self.client.post(self.endpoint, json=payload, headers=self.headers())
                latency = time.perf_counter() - t0
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"text": r.text}
                if self._log:
                    self._log(request=payload, response=body, latency_s=latency, status=r.status_code, attempt=attempt)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                if r.status_code in (401, 402, 403):  # auth / credits: retrying cannot help
                    raise RuntimeError(f"{self.name}: HTTP {r.status_code} from {self.endpoint}: {body}")
                r.raise_for_status()
                self.last_latency, self.last_attempts = latency, attempt + 1  # the successful attempt only
                self.last_usage = self.parse_usage(body)
                return self.parse_response(body, options, option_ids)
            except httpx.TransportError as e:
                if self._log:
                    self._log(request=payload, response={"error": repr(e)}, latency_s=time.perf_counter() - t0,
                              attempt=attempt)
                if attempt >= self.max_retries:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")
