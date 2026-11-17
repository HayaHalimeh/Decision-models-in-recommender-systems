from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

log = logging.getLogger("dm4recsys")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("dm4recsys").setLevel(getattr(logging, level.upper(), logging.INFO))


def stable_seed(*parts: Any) -> int:
    """Deterministic 32-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "little")


def rng_for(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*parts))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # a truncated last line from an interrupted run is ignored (and later rewritten)
                log.warning("Skipping malformed JSONL line in %s", p)
    return out


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


class JsonlAppender:
    """Append-only JSONL writer that flushes and fsyncs after every row (safe for resumable runs)."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _repair_trailing_line(self.path)
        self._f = open(self.path, "a")
        self._lock = threading.Lock()  # concurrent runs (runner `concurrency`) write from several threads

    def write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            self._f.write(line)
            self._f.flush()
            os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _repair_trailing_line(p: Path) -> None:
    """Drop a partially written last line (no trailing newline) left by a crash."""
    if not p.exists() or p.stat().st_size == 0:
        return
    with open(p, "rb") as f:
        data = f.read()
    if data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n")
    with open(p, "wb") as f:
        f.write(data[: cut + 1] if cut >= 0 else b"")


def write_json(path: str | os.PathLike, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def read_json(path: str | os.PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def timer() -> Iterator[dict]:
    d: dict = {}
    t0 = time.perf_counter()
    try:
        yield d
    finally:
        d["seconds"] = time.perf_counter() - t0


def pick_device(pref: str = "auto") -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync_device(device: str) -> None:
    """Block until queued kernels finish, so wall-clock latency is honest."""
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv

        from .config import REPO_ROOT

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass


def load_object(spec: str):
    """Import "package.module:name" and return the attribute (e.g. a factory function)."""
    import importlib

    mod, _, name = spec.partition(":")
    return getattr(importlib.import_module(mod), name)
