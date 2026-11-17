"""sha256 + size of every data file, so a rebuild can be checked against the original without publishing the data.

Covers the raw files the loaders read (pins the exact dataset versions) and the processed datasets (the titles and
histories the models see); splits and candidates are published as they are. Writes data/{mode}/checksums.json
(paths relative to the repository); --check compares the local files against that file instead (exit 1 on a
mismatch or a missing file). Only checksums and sizes are stored, no data.

    python scripts/make_checksums.py --config configs/server.yaml           # write
    python scripts/make_checksums.py --config configs/server.yaml --check   # verify a rebuild
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dm4recsys.cli import parser, setup
from dm4recsys.utils import sha256_file

GROUPS = ("raw", "processed")


def raw_files(cfg: dict) -> list[Path]:
    """The raw files the loaders read (configs/datasets/*.yaml), not everything in data/raw."""
    raw, out = Path(cfg["paths"]["raw"]), []
    for d in cfg["run_datasets"]:
        dc = cfg["datasets"][d]
        if dc["loader"] == "ml1m":
            out += [raw / "ml-1m.zip", raw / "ml-1m" / "ratings.dat", raw / "ml-1m" / "movies.dat"]
        elif dc["loader"] == "amazon":
            out += [raw / "amazon" / dc["category"] / Path(dc[k]).name for k in ("ratings_file", "meta_file")]
        elif dc["loader"] == "yelp":
            out += [raw / "yelp" / dc[k] for k in ("review_file", "business_file")]
    return [p for p in out if p.exists()]


def collect(cfg: dict) -> dict:
    root = Path.cwd()
    out = {}
    for g in GROUPS:
        base = Path(cfg["paths"][g])
        if g == "raw":
            files = raw_files(cfg)
        else:
            files = sorted(p for p in base.rglob("*") if p.is_file() and p.name != ".DS_Store") if base.exists() else []
        out[g] = {os.path.relpath(p, root): {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in files}
    return out


def main() -> None:
    p = parser(__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="verify local files against checksums.json")
    args = p.parse_args()
    cfg = setup(args)
    target = Path(cfg["paths"]["splits"]).parent / "checksums.json"
    now = collect(cfg)
    if not args.check:
        doc = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": cfg["mode"],
               "note": "sha256 and size per file; regenerate with scripts/make_checksums.py, verify with --check",
               **now}
        target.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {target}: " + ", ".join(f"{g} {len(now[g])} files" for g in GROUPS))
        return
    ref = json.loads(target.read_text())
    bad = 0
    for g in GROUPS:
        for f, v in ref.get(g, {}).items():
            got = now[g].get(f)
            status = "missing" if got is None else "ok" if got["sha256"] == v["sha256"] else "DIFFERS"
            if status != "ok":
                bad += 1
                print(f"  {status:8s} {f}")
        print(f"{g}: {len(ref.get(g, {}))} files checked")
    print("all files match" if not bad else f"{bad} files missing or different")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
