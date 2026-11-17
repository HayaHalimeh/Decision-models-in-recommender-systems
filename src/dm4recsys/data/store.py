"""Read access to a processed dataset: sequences, titles, splits, popularity."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from ..config import path
from ..utils import read_json


@dataclass
class DatasetStore:
    name: str
    interactions: pd.DataFrame          # user, item, timestamp, pos (ordered)
    titles: dict[str, str]
    splits: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @classmethod
    def load(cls, cfg: dict, name: str) -> "DatasetStore":
        d = path(cfg, "processed", name)
        inter = pd.read_csv(d / "interactions.tsv", sep="\t", dtype={"user": str, "item": str})
        items = pd.read_csv(d / "items.tsv", sep="\t", dtype=str, keep_default_na=False)
        titles = dict(zip(items["item"], items["title"]))
        sp = path(cfg, "splits", f"{name}.json")
        splits = read_json(sp) if sp.exists() else {}
        info = read_json(d / "stats.json") if (d / "stats.json").exists() else {}
        return cls(name, inter, titles, splits, info)

    @cached_property
    def sequences(self) -> dict[str, list[str]]:
        """Full chronological item sequence per user (includes validation and test targets)."""
        return {u: g["item"].tolist() for u, g in self.interactions.groupby("user", sort=False)}

    def test_target(self, user: str) -> str:
        return self.sequences[user][-1]

    def val_target(self, user: str) -> str:
        return self.sequences[user][-2]

    def history_before_test(self, user: str) -> list[str]:
        """All interactions before the test target, oldest to newest (never contains the target position)."""
        return self.sequences[user][:-1]

    @cached_property
    def popularity(self) -> dict[str, int]:
        """Training interaction count: all interactions except each user's test target."""
        train = self.interactions[self.interactions["pos"] < self.interactions.groupby("user")["pos"].transform("max")]
        return train["item"].value_counts().to_dict()

    @property
    def all_items(self) -> list[str]:
        return sorted(self.titles)

    @property
    def users(self) -> list[str]:
        return list(self.sequences)

    def title(self, item: str) -> str:
        return self.titles[item]


def processed_exists(cfg: dict, name: str) -> bool:
    return (Path(path(cfg, "processed", name)) / "interactions.tsv").exists()
