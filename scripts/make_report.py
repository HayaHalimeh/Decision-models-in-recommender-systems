"""Reports from the result files in runs/{mode}/ (no model is called):
  rq1: Table 1, Figure 1, text_numbers.json
  rq2: Figure 3, its caption, CSVs (results/{mode}/rq_csv/), RQ2 text numbers
  rq3: Figure 4, its caption, CSVs, RQ3 text numbers
  posthoc: the post-hoc tests (report/posthoc.py): CSVs in results/{mode}/, crossover also at NDCG@10/@20
  all: all of the above
"""
import json

from dm4recsys.cli import parser, setup
from dm4recsys.report.build import build_report
from dm4recsys.report.posthoc import build_posthoc
from dm4recsys.rq.report import build_rq_report

if __name__ == "__main__":
    p = parser(__doc__.splitlines()[0])
    p.add_argument("--rq", choices=["rq1", "rq2", "rq3", "posthoc", "all"], default="all")
    args = p.parse_args()
    cfg = setup(args)
    if args.rq in ("rq1", "all"):
        tn = build_report(cfg)
        print(json.dumps({k: tn[k] for k in ("mode", "missing_models", "crossover_pooled", "bpr_recall_at_20")},
                         indent=1, default=str))
    if args.rq in ("rq2", "rq3", "all"):
        build_rq_report(cfg, args.rq)  # rq2 | rq3 | all (= both)
    if args.rq in ("posthoc", "all"):
        build_posthoc(cfg)
    print(f"Report outputs written to {cfg['paths']['results']}")
