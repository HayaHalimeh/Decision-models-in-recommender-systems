"""Downloads for ML-1M and Amazon Reviews 2023. Yelp is placed by hand (licence)."""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import requests

log = logging.getLogger(__name__)


def fetch(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for b in r.iter_content(chunk):
                f.write(b)
    tmp.replace(dest)
    return dest


def ensure_ml1m(raw_dir: Path, url: str) -> Path:
    d = Path(raw_dir) / "ml-1m"
    if (d / "ratings.dat").exists() and (d / "movies.dat").exists():
        return d
    z = fetch(url, Path(raw_dir) / "ml-1m.zip")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(raw_dir)
    return d


def ensure_amazon(raw_dir: Path, dcfg: dict) -> Path:
    d = Path(raw_dir) / "amazon" / dcfg["category"]
    for rel in (dcfg["ratings_file"], dcfg["meta_file"]):
        fetch(f"{dcfg['hf_base']}/{rel}", d / Path(rel).name)
    return d


def amazon_present(raw_dir: Path, dcfg: dict) -> bool:
    d = Path(raw_dir) / "amazon" / dcfg["category"]
    return (d / Path(dcfg["ratings_file"]).name).exists() and (d / Path(dcfg["meta_file"]).name).exists()


def ml1m_present(raw_dir: Path) -> bool:
    d = Path(raw_dir) / "ml-1m"
    return (d / "ratings.dat").exists() and (d / "movies.dat").exists()


def yelp_present(raw_dir: Path, dcfg: dict) -> bool:
    d = Path(raw_dir) / "yelp"
    return (d / dcfg["review_file"]).exists() and (d / dcfg["business_file"]).exists()
