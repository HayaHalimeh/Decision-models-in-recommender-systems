"""Raw loaders. Each returns (interactions, titles, info):

interactions: DataFrame[user:str, item:str, timestamp:float]  (one row per rating/review)
titles:       dict[item -> title]
info:         dict with provenance (e.g. chosen Yelp city)
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


def clean_title(t) -> str:
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return ""
    return _WS.sub(" ", str(t)).strip()


_YEAR = re.compile(r"\s*\(\d{4}\)\s*$")


def load_ml1m(raw_dir: Path, dcfg: dict):
    d = Path(raw_dir) / "ml-1m"
    df = pd.read_csv(d / "ratings.dat", sep="::", engine="python", header=None,
                     names=["user", "item", "rating", "timestamp"], dtype=str)
    df["timestamp"] = df["timestamp"].astype(float)
    movies = pd.read_csv(d / "movies.dat", sep="::", engine="python", header=None,
                         names=["item", "title", "genres"], dtype=str, encoding="latin-1")
    titles = {}
    for i, t in zip(movies["item"], movies["title"]):
        t = clean_title(t)
        if dcfg.get("strip_year"):
            t = _YEAR.sub("", t)
        titles[i] = t
    return df[["user", "item", "timestamp"]], titles, {"source": "real"}


def load_amazon(raw_dir: Path, dcfg: dict):
    d = Path(raw_dir) / "amazon" / dcfg["category"]
    df = pd.read_csv(d / Path(dcfg["ratings_file"]).name, dtype=str)
    df = df.rename(columns={"user_id": "user", "parent_asin": "item"})
    df["timestamp"] = df["timestamp"].astype(float)
    titles = {}
    with open(d / Path(dcfg["meta_file"]).name) as f:
        for line in f:
            m = json.loads(line)
            titles[m["parent_asin"]] = clean_title(m.get("title"))
    return df[["user", "item", "timestamp"]], titles, {"source": "real"}


def _norm_city(c: str, normalise: bool) -> str:
    c = clean_title(c)
    return c.casefold() if normalise else c


def load_yelp(raw_dir: Path, dcfg: dict):
    d = Path(raw_dir) / "yelp"
    norm = dcfg.get("city_normalise", True)
    biz_city, biz_name, biz_city_raw, biz_addr = {}, {}, {}, {}
    with open(d / dcfg["business_file"]) as f:
        for line in f:
            b = json.loads(line)
            biz_city[b["business_id"]] = _norm_city(b.get("city", ""), norm)
            biz_city_raw[b["business_id"]] = clean_title(b.get("city", ""))
            biz_name[b["business_id"]] = clean_title(b.get("name"))
            biz_addr[b["business_id"]] = clean_title(b.get("address") or "") or clean_title(b.get("postal_code") or "")
    review_path = d / dcfg["review_file"]
    city = dcfg.get("city", "auto")
    counts: Counter = Counter()  # reviews per city, always counted (stats.json: top_cities_by_reviews)
    with open(review_path) as f:
        for line in f:
            bid = json.loads(line)["business_id"]
            counts[biz_city.get(bid, "")] += 1
    city_key = counts.most_common(1)[0][0] if city == "auto" else _norm_city(city, norm)
    users, items, ts = [], [], []
    with open(review_path) as f:
        for line in f:
            r = json.loads(line)
            if biz_city.get(r["business_id"]) == city_key:
                users.append(r["user_id"])
                items.append(r["business_id"])
                ts.append(pd.Timestamp(r["date"]).timestamp())
    df = pd.DataFrame({"user": users, "item": items, "timestamp": ts})
    raw_names = Counter(biz_city_raw[b] for b in df["item"].unique())
    info = {
        "source": "real",
        "city_key": city_key,
        "city_spellings": dict(raw_names.most_common(10)),
        "top_cities_by_reviews": dict(counts.most_common(5)) if counts else None,
    }
    log.info("Yelp city: %s (%d reviews)", city_key, len(df))
    titles = {b: biz_name.get(b, "") for b in df["item"].unique()}
    if dcfg.get("disambiguate_titles") == "address":  # applied after the k-core (pipeline.disambiguate_titles)
        info["_title_suffix"] = {b: biz_addr.get(b, "") for b in titles}
    return df, titles, info


LOADERS = {"ml1m": load_ml1m, "amazon": load_amazon, "yelp": load_yelp}
