"""Step 2: tune + train BPR, generate and freeze injected/realistic candidate sets."""
from dm4recsys.cli import parser, setup
from dm4recsys.retrieval.bpr import run

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--force", action="store_true", help="deliberately overwrite frozen candidates")
    args = p.parse_args()
    cfg = setup(args)
    for d in cfg["run_datasets"]:
        run(cfg, d, force=args.force)
