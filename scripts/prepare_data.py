"""Step 1: download/load raw data, 5-core, chronological order, test/val user sampling.

Default: the users are sampled with data.sample_seed, i.e. our test users. --force samples new test users with a
new random seed (stored in data/{mode}/splits/{dataset}.json); then rebuild the candidates with
`make candidates ARGS=--force`.
"""
import logging
import secrets

from dm4recsys.cli import parser, setup
from dm4recsys.data.pipeline import prepare_dataset

if __name__ == "__main__":
    p = parser(__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true", help="sample new test users with a new random seed")
    args = p.parse_args()
    cfg = setup(args)
    seed = secrets.randbelow(2**31) if args.force else None
    if seed is not None:
        logging.getLogger("prepare_data").warning("new test users, seed %d: rebuild the candidates with "
                                                  "`make candidates ARGS=--force`", seed)
    for d in cfg["run_datasets"]:
        prepare_dataset(cfg, d, seed=seed)
