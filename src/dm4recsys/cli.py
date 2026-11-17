"""Shared CLI plumbing for scripts/*.py."""
from __future__ import annotations

import argparse
import warnings

from .config import load_config
from .utils import load_dotenv_if_present, setup_logging


def parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default="configs/server.yaml", help="run configuration (default configs/server.yaml)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. --set models.jev.timeout_s=120")
    p.add_argument("--datasets", nargs="*", default=None)
    return p


def setup(args) -> dict:
    warnings.filterwarnings("ignore", category=FutureWarning)
    load_dotenv_if_present()
    cfg = load_config(args.config, args.set)
    setup_logging(cfg.get("log_level", "INFO"))
    if args.datasets:
        cfg["run_datasets"] = list(args.datasets)
    return cfg
