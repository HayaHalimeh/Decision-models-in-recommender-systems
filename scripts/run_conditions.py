"""RQ2/RQ3: run decision models under the history / candidate conditions.

Examples:
  python scripts/run_conditions.py --rq rq2 --condition no_history --model jev --max-users 50   # pilot
  python scripts/run_conditions.py --rq all --condition all --model all
  python scripts/run_conditions.py --list
"""
from dm4recsys.cli import parser, setup
from dm4recsys.data.store import DatasetStore
from dm4recsys.rq.conditions import all_conditions
from dm4recsys.rq.runner import run_condition

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--rq", choices=["rq2", "rq3", "all"], default="all")
    p.add_argument("--condition", nargs="*", default=["all"], help="condition names or 'all'")
    p.add_argument("--model", nargs="*", default=["all"], help="decision model names or 'all'")
    p.add_argument("--max-users", type=int, default=None, help="first N frozen users per dataset (pilot)")
    p.add_argument("--list", action="store_true", help="print the condition names and exit")
    args = p.parse_args()
    cfg = setup(args)
    conds = all_conditions(cfg, args.rq)
    if args.list:
        for c in conds:
            print(f"{c['name']:14s} {c['rq']:6s} split={c['split']:8s} history={c['history']} candidates={c['candidates']}")
        raise SystemExit(0)
    if args.condition != ["all"]:
        unknown = set(args.condition) - {c["name"] for c in conds}
        if unknown:
            raise SystemExit(f"unknown condition(s) for --rq {args.rq}: {sorted(unknown)}")
        conds = [c for c in conds if c["name"] in args.condition]
    models = cfg["rq"]["models"] if args.model == ["all"] else args.model
    stores = {d: DatasetStore.load(cfg, d) for d in cfg["run_datasets"]}
    from dm4recsys.models.registry import build

    for m in models:
        rr = build(m, cfg, stores)
        try:
            for c in conds:
                run_condition(cfg, m, c, cfg["run_datasets"], max_users=args.max_users, ranker=rr, stores=stores)
        finally:
            rr.close()
