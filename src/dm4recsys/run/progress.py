"""Progress reports: one line per finished step, in the log and in results/{mode}/progress.jsonl.

The log line starts with `REPORT <step>` so a long pipeline log can be filtered with `grep REPORT`.
"""
from __future__ import annotations

import datetime as dt
import json
import logging

from ..config import path
from ..utils import JsonlAppender, _json_default

log = logging.getLogger("dm4recsys.progress")


def progress_path(cfg: dict):
    return path(cfg, "results", "progress.jsonl")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3g}" if v == 0 or abs(v) < 0.01 else f"{v:.4f}"  # 0.0005, 1e-05 instead of 0.0000
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def report(cfg: dict, step: str, **fields) -> dict:
    fields = json.loads(json.dumps(fields, default=_json_default))  # numpy -> Python, field order kept
    row = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "step": step, **fields}
    with JsonlAppender(progress_path(cfg)) as w:
        w.write(row)
    log.info("REPORT %s | %s", step, "  ".join(f"{k}={_fmt(v)}" for k, v in fields.items()))
    return row
