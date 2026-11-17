"""Step 3: SASRec / BERT4Rec / GRU4Rec grid search + training-data curve; scores the frozen candidates."""
from dm4recsys.cli import parser, setup
from dm4recsys.models.sequential.train import run_curve

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--fractions", nargs="*", type=int, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    args = p.parse_args()
    cfg = setup(args)
    for d in cfg["run_datasets"]:
        for m in args.models or cfg["sequential"]["models"]:
            run_curve(cfg, d, m, args.fractions, args.seeds)
