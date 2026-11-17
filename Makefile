# Decision models as zero-shot next-item rankers: RQ1 (accuracy), RQ2 (history), RQ3 (position, labels, popularity).
#
# Replicating the experiments, in this order (every step is resumable and skips what is already done):
#   make env                                   # Python environment (uv)
#   make data                                  # raw datasets in data/raw -> data/full/processed, splits (our test users)
#   make data ARGS=--force                     # sample new test users with a new random seed instead
#   make candidates                            # BPR, 20 frozen candidates per test user (+ long-history sample);
#                                              # the frozen candidates are in data/full/candidates (skipped if present)
#   make candidates ARGS=--force               # retrain BPR and rebuild the candidates instead
#   make sequential                            # SASRec, BERT4Rec, GRU4Rec (training-size curves)
#   bash scripts/local_serve.sh qwen35_9b      # local LLMs / open decision models: start the model's server first
#   make rq1-run MODEL=jev                     # one model (or a group: decision, llm, plm, reference); default all
#   make rq1-report                            # Table 1, Figure 1, text numbers -> results/full/
#   make rq2-run MODEL=jev COND=no_history     # RQ2 history conditions; defaults: every RQ2/RQ3 model, every condition
#   make rq3-run MODEL=jev COND=label_swap     # RQ3 position / label / popularity conditions; same defaults
#   make rq2-report / make rq3-report          # Figures 3 / 4, captions, CSVs (results/full/rq_csv/), text numbers
#   make posthoc                               # post-hoc tests (report/posthoc.py) -> results/full/*.csv
#   make report                                # everything: all reports + post-hoc at NDCG@5, and all reports at
#                                              # NDCG@10 / @20 (robustness) -> results/full/k10/, results/full/k20/
# Options: CONFIG=configs/server.yaml (default), ARGS="..." passes further flags to the script
# (e.g. ARGS="--datasets ml1m" or ARGS="--max-users 50"). API keys go in .env (see .env.example).
PY      ?= .venv/bin/python
CONFIG  ?= configs/server.yaml
MODEL   ?= all
COND    ?= all
ARGS    ?=
export PYTHONPATH := src

.PHONY: env data candidates sequential rq1-run rq1-report rq2-run rq3-run rq2-report rq3-report posthoc report

env:
	uv sync

# ---------------------------------------------------------------- data, candidates, trained baselines
data:
	$(PY) scripts/prepare_data.py --config $(CONFIG) $(ARGS)

candidates:
	$(PY) scripts/make_candidates.py --config $(CONFIG) $(ARGS)

sequential:
	$(PY) scripts/train_sequential.py --config $(CONFIG) $(ARGS)

# ---------------------------------------------------------------- RQ1: all zero-shot rankers and references
rq1-run:
	$(PY) scripts/run_model.py --config $(CONFIG) --model $(MODEL) $(ARGS)

rq1-report:
	$(PY) scripts/make_report.py --config $(CONFIG) --rq rq1 $(ARGS)

# ---------------------------------------------------------------- RQ2 / RQ3: decision models and LLMs
# Run a specific condition with: $(PY) scripts/run_conditions.py --rq rq2 --condition <condition_name> --model $(MODEL) $(ARGS)
# List the condition names with: $(PY) scripts/run_conditions.py --list
rq2-run:
	$(PY) scripts/run_conditions.py --config $(CONFIG) --rq rq2 --condition $(COND) --model $(MODEL) $(ARGS)

rq3-run:
	$(PY) scripts/run_conditions.py --config $(CONFIG) --rq rq3 --condition $(COND) --model $(MODEL) $(ARGS)

rq2-report:
	$(PY) scripts/make_report.py --config $(CONFIG) --rq rq2 $(ARGS)

rq3-report:
	$(PY) scripts/make_report.py --config $(CONFIG) --rq rq3 $(ARGS)

posthoc:
	$(PY) scripts/make_report.py --config $(CONFIG) --rq posthoc $(ARGS)

ROBUST_K ?= 10 20
report:
	$(PY) scripts/make_report.py --config $(CONFIG) --rq all $(ARGS)
	for k in $(ROBUST_K); do \
	  $(PY) scripts/make_report.py --config $(CONFIG) --rq all --set eval.k=$$k --set "paths.results=results/{mode}/k$$k" \
	    --set report.robustness_k=[] $(ARGS) || exit 1; \
	done
