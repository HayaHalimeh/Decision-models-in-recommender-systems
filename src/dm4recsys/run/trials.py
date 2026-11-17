"""Tuning trials saved one by one, so an interrupted grid search continues where it stopped.

Each finished trial is appended (flushed + fsynced) to a JSONL file together with a fingerprint of everything that
determines its result (seed, training settings, data hash, device, library versions). On a restart, trials whose
fingerprint matches are reused and only the missing grid points are trained; trials with another fingerprint
(changed settings, data or environment) are ignored, never silently reused.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..utils import JsonlAppender, _json_default, read_jsonl

log = logging.getLogger(__name__)


def canonical(obj) -> object:
    """JSON round trip with sorted keys (numpy -> Python): the form in which params and fingerprints are compared."""
    return json.loads(json.dumps(obj, sort_keys=True, default=_json_default))


def params_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=_json_default)


class TrialStore:
    def __init__(self, path: str | Path, fingerprint: dict):
        self.path = Path(path)
        self.fingerprint = canonical(fingerprint)
        self._saved: dict[str, dict] = {}
        ignored = 0
        for row in read_jsonl(self.path):
            if row.get("fingerprint") == self.fingerprint:
                self._saved[params_key(row["params"])] = row
            else:
                ignored += 1
        if ignored:
            log.warning("%s: %d saved trial(s) with a different fingerprint ignored", self.path, ignored)

    def get(self, params: dict) -> dict | None:
        return self._saved.get(params_key(canonical(params)))

    def __len__(self) -> int:
        return len(self._saved)

    def add(self, trial: dict) -> dict:
        row = {**canonical(trial), "fingerprint": self.fingerprint}
        with JsonlAppender(self.path) as w:
            w.write(row)
        self._saved[params_key(row["params"])] = row
        return row
