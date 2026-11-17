"""Step 4: run one zero-shot model. Example: python scripts/run_model.py --model jev"""
from dm4recsys.cli import parser, setup
from dm4recsys.config import models_in_group, zero_shot_models
from dm4recsys.run.runner import run_model

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--model", required=True, help="model name, 'all', or a group name (decision, llm, plm, reference)")
    p.add_argument("--limit", type=int, default=None, help="first N users only (debugging)")
    p.add_argument("--runs", nargs="*", type=int, default=None)
    p.add_argument("--no-determinism-check", action="store_true")
    args = p.parse_args()
    cfg = setup(args)
    groups = {"decision", "llm", "plm", "reference"}
    models = (zero_shot_models(cfg) if args.model == "all"
              else models_in_group(cfg, args.model) if args.model in groups else [args.model])
    for m in models:
        run_model(cfg, m, cfg["run_datasets"], limit=args.limit, runs=args.runs,
                  check_determinism=not args.no_determinism_check)
