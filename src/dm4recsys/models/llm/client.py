"""OpenAI-compatible chat client (hosted APIs and vLLM servers)."""
from __future__ import annotations

import logging
import os
import threading
import time

log = logging.getLogger(__name__)


def completion_prompt(messages: list[dict]) -> str:
    """Plain-text prompt for base models: the message contents, unchanged, joined by blank lines (normally just the
    one user message, since there is no system message), followed by a newline so the answer starts on its own line."""
    return "\n\n".join(m["content"] for m in messages) + "\n"


def split_thinking(message) -> tuple[str | None, str | None]:
    """(answer, thinking) of a chat message. Qwen3.5 in thinking mode writes `<think>...</think>` before the answer:
    vLLM with `--reasoning-parser qwen3` returns it separately (`reasoning_content` / `reasoning`); without the
    parser it is part of `content` and is cut off here, so IDs written while thinking never reach the ranking parser.
    Thinking cut off by max_tokens (no `</think>`) -> no answer (non-compliant, logged via finish_reason)."""
    text = message.content
    extra = getattr(message, "model_extra", None) or {}
    thinking = getattr(message, "reasoning_content", None) or extra.get("reasoning_content") or extra.get("reasoning")
    if text and "</think>" in text:
        head, _, text = text.partition("</think>")
        thinking = (thinking or "") + head.replace("<think>", "", 1)
    elif text and text.lstrip().startswith("<think>"):
        thinking, text = (thinking or "") + text.lstrip()[len("<think>"):], None
    return (text.strip() if text else text), thinking


class OpenAICompatClient:
    def __init__(self, mcfg: dict):
        from openai import OpenAI

        base_url = os.environ.get(mcfg.get("base_url_env") or "", "") or mcfg.get("base_url")
        api_key = os.environ.get(mcfg.get("api_key_env") or "", "") or ("EMPTY" if mcfg.get("deployment") == "local" else "")
        if not api_key:
            raise RuntimeError(f"{mcfg['name']}: set {mcfg.get('api_key_env')} in .env")
        self.mcfg = mcfg
        # retries are done here, not by the SDK, so that latency is measured per attempt (one rate limit must not
        # inflate the reported latency); same policy as the decision APIs (hosted.py)
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=mcfg.get("timeout_s", 120), max_retries=0)
        self.max_retries = int(mcfg.get("max_retries", 5))
        self.base_url = base_url
        # min_interval_s: at least this many seconds between the starts of two requests (every attempt counts), so a
        # per-key requests/min limit is not hit and no time is lost in 429 back-off. Pacing only: the requests and
        # hence the results are unchanged. Shared by all threads of this client.
        self.min_interval = float(mcfg.get("min_interval_s", 0) or 0)
        self._next_start = 0.0
        self._pace_lock = threading.Lock()

    def _pace(self) -> None:
        if self.min_interval <= 0:
            return
        with self._pace_lock:
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + self.min_interval
        if start > now:
            time.sleep(start - now)

    def chat(self, messages: list[dict], run: int = 0) -> dict:
        # gpt-6-luna rejects `max_tokens` and needs `max_completion_tokens`; vLLM takes `max_tokens`
        tok = self.mcfg.get("max_tokens_param", "max_tokens")
        kw = {"model": self.mcfg["model_id"], "messages": messages, tok: self.mcfg.get("max_tokens", 300)}
        if self.mcfg.get("temperature", 0) is not None:  # null = send no temperature (reasoning models: default only)
            kw["temperature"] = self.mcfg.get("temperature", 0)
        if self.mcfg.get("seed_requests"):
            kw["seed"] = int(self.mcfg.get("request_seed", 0)) + run
        if self.mcfg.get("stop"):  # e.g. Qwen3.5-4B-Base: stop at the first blank line
            kw["stop"] = list(self.mcfg["stop"])
        if self.mcfg.get("extra_body"):
            kw["extra_body"] = self.mcfg["extra_body"]
        if self.mcfg.get("api") == "completions":
            # base models (no chat template, e.g. Qwen3.5-4B-Base): the same prompt text as plain completion
            kw["prompt"] = completion_prompt(kw.pop("messages"))
            create = self.client.completions.create
        else:
            create = self.client.chat.completions.create
        resp, latency, attempts = self._with_retries(create, kw)
        if self.mcfg.get("api") == "completions":
            text, thinking = resp.choices[0].text, None
        else:
            text, thinking = split_thinking(resp.choices[0].message)
        u = resp.usage
        return {
            "text": text,
            "usage": {"input_tokens": getattr(u, "prompt_tokens", None),
                      "output_tokens": getattr(u, "completion_tokens", None), "calls": 1,
                      "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None)},
            "model_version": resp.model,
            "raw": resp.model_dump(),
            "request": {k: v for k, v in kw.items()},
            "thinking_chars": None if thinking is None else len(thinking),
            "latency_s": latency,      # the successful attempt only
            "attempts": attempts,
            "finish_reason": getattr(resp.choices[0], "finish_reason", None),
            # vLLM: the stop string that ended generation (None = model's own end token or max_tokens)
            "stop_reason": getattr(resp.choices[0], "stop_reason", None)
            or (getattr(resp.choices[0], "model_extra", None) or {}).get("stop_reason"),
        }

    def _with_retries(self, create, kw):
        """Retry on 429 / 5xx / connection errors with exponential back-off (1, 2, 4, ... s); auth errors are final.
        Returns (response, latency of the successful attempt in s, number of attempts)."""
        import openai

        delay = 1.0
        for attempt in range(self.max_retries + 1):
            self._pace()
            t0 = time.perf_counter()
            try:
                resp = create(**kw)
                return resp, time.perf_counter() - t0, attempt + 1
            except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
                if attempt >= self.max_retries:
                    raise
                log.warning("%s: %s (attempt %d), retrying in %.0f s", self.mcfg["name"], type(e).__name__,
                            attempt + 1, delay)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")
