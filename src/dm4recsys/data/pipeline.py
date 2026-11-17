"""Build one processed dataset: load raw -> preprocess -> write -> sample test/val users."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import path
from ..utils import write_json
from . import download as dl
from ..utils import load_object
from .loaders import LOADERS
from .preprocess import preprocess

log = logging.getLogger(__name__)


def _raw_available(cfg: dict, name: str) -> bool:
    dcfg = cfg["datasets"][name]
    raw = path(cfg, "raw")
    loader = dcfg["loader"]
    if loader == "ml1m":
        if not dl.ml1m_present(raw) and cfg["data"].get("download"):
            try:
                dl.ensure_ml1m(raw, dcfg["url"])
            except Exception as e:
                log.warning("ML-1M download failed: %s", e)
        return dl.ml1m_present(raw)
    if loader == "amazon":
        if not dl.amazon_present(raw, dcfg) and cfg["data"].get("download"):
            dl.ensure_amazon(raw, dcfg)
        return dl.amazon_present(raw, dcfg)
    if loader == "yelp":
        return dl.yelp_present(raw, dcfg)
    raise ValueError(loader)


def disambiguate_titles(titles: dict, suffix: dict) -> tuple[dict, dict]:
    """Yelp: every item whose title is shared by another item of the final dataset gets ' (<suffix>)' (its street
    address). Unique titles stay unchanged. Applied to the item file, so history and candidates show the same text.
    Items that still share a text afterwards (same name AND address: duplicate Yelp listings) are left and counted."""
    n = pd.Series(titles).value_counts()
    out = {i: (f"{t} ({suffix[i]})" if n[t] > 1 and suffix.get(i) else t) for i, t in titles.items()}
    m = pd.Series(out).value_counts()
    st = {"items_with_shared_title": int(sum(n[t] > 1 for t in titles.values())),
          "titles_changed": int(sum(out[i] != titles[i] for i in titles)),
          "items_still_shared": int(sum(m[t] > 1 for t in out.values()))}
    return out, st


def load_raw(cfg: dict, name: str):
    dcfg = cfg["datasets"][name]
    if dcfg.get("loader_factory"):  # optional "module:function" (cfg, name) -> (df, titles, info)
        return load_object(dcfg["loader_factory"])(cfg, name)
    if _raw_available(cfg, name):
        return LOADERS[dcfg["loader"]](path(cfg, "raw"), dcfg)
    raise FileNotFoundError(
        f"Raw data for '{name}' missing. See README (Yelp must be downloaded by hand into data/raw/yelp/)."
    )


def sample_users(users: list[str], n_test: int, n_val: int, seed: int) -> dict:
    """Test users: random sample of ALL users after 5-core. Validation users: disjoint random sample."""
    users = sorted(users)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(users))
    n_test = min(n_test, len(users))
    n_val = min(n_val, len(users) - n_test)
    test = [users[i] for i in perm[:n_test]]
    val = [users[i] for i in perm[n_test:n_test + n_val]]
    return {"seed": seed, "test_users": test, "val_users": val}


def prepare_dataset(cfg: dict, name: str, seed: int | None = None) -> dict:
    """seed: the test/validation user sample (default data.sample_seed, i.e. our test users)."""
    df, titles, info = load_raw(cfg, name)
    dcfg_data = dict(cfg["data"])
    suffix = info.pop("_title_suffix", None)
    df, titles, st = preprocess(df, titles, dcfg_data)
    if suffix is not None:
        titles, st["title_disambiguation"] = disambiguate_titles(titles, suffix)
        log.info("%s: title disambiguation %s", name, st["title_disambiguation"])
    st["info"] = info
    need = cfg["data"]["n_test_users"] + cfg["data"]["n_val_users"]
    if st["final"]["users"] < need:
        raise ValueError(f"{name}: only {st['final']['users']} users after the {cfg['data']['min_core']}-core "
                         f"(< {need} test + validation users).")
    out = path(cfg, "processed", name)
    out.mkdir(parents=True, exist_ok=True)
    df[["user", "item", "timestamp", "pos"]].to_csv(out / "interactions.tsv", sep="\t", index=False)
    pd.DataFrame(sorted(titles.items()), columns=["item", "title"]).to_csv(out / "items.tsv", sep="\t", index=False)
    write_json(out / "stats.json", st)
    sp = sample_users(df["user"].unique().tolist(), cfg["data"]["n_test_users"], cfg["data"]["n_val_users"],
                      cfg["data"]["sample_seed"] if seed is None else seed)
    write_json(path(cfg, "splits", f"{name}.json"), sp)
    log.info("%s: %s", name, st["final"])
    return st
