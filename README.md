# Evaluating Decision Models as Zero-Shot Rankers in Recommender Systems

Code, protocol and results for the WWW 2027 short paper. Six decision models ("System One" models that return a
probability distribution over a fixed set of options) rank 20 BPR candidates per user and are compared with LLM
rankers, PLM rankers and trained sequential recommenders (RQ1), tested for how they use the interaction history (RQ2)
and for position, label and popularity biases (RQ3).

Contents: [Reproducing the paper](#reproducing-the-paper) ·
[Paper → files](#paper--repository-mapping) · [Models](#models-and-versions) · [Prompts](#prompts) ·
[Data](#data-and-candidates) · [LLM output parsing](#llm-output-parsing) · [Sampling and runs](#sampling-runs-and-determinism) ·
[Statistics](#statistics) · [RQ2 / RQ3 conditions](#rq2-and-rq3-conditions) · [Latency](#latency) ·
[NDCG@10 / @20](#ndcg10-and-ndcg20) · [Layout](#layout)

---

## Reproducing the paper

Python 3.11, dependencies pinned in `pyproject.toml` / `uv.lock` (`uv sync`). API keys go into `.env`
(`cp .env.example .env`; never committed). Every step is resumable and skips what is already done.

```bash
make data                       # ML-1M and Amazon Video Games are downloaded; Yelp by hand (configs/datasets/yelp.yaml)
make candidates                 # BPR tuning + final BPR, 20 frozen candidates per test user (+ long-history sample)
make sequential                 # SASRec, BERT4Rec, GRU4Rec: 6 training shares x 3 seeds
make rq1-run MODEL=jev          # one model (or a group: decision, llm, plm, reference); default all
make rq2-run / make rq3-run     # RQ2 / RQ3 conditions (MODEL=..., COND=...; scripts/run_conditions.py --list)
make report                     # everything below, from the result files only (no model call)
```

`make report` writes all outputs to `results/full/`: Table 1, Figure 1, the RQ2/RQ3 figures, all CSVs and text
numbers at NDCG@5, the post-hoc tests (`make posthoc` alone), and the complete report again at NDCG@10 and @20
(`results/full/k10/`, `results/full/k20/`). Local models are served with `scripts/local_serve.sh <model>`
(vLLM 0.33.0 for the Qwen3.5 LLMs, the System One servers for the open decision models; bf16).

## Paper → repository mapping

Figure numbers in the paper differ from the file names.

| Paper | File | Made by |
|---|---|---|
| Table 1 | `results/full/table1.tex` | `make rq1-report` |
| Figure 1 (RQ1, sequential vs. decision models) | `results/full/fig1.{png,pdf}` | `make rq1-report` |
| Figure 2 (RQ2) | `results/full/fig3.{png,pdf}`, caption `fig3_caption.txt` | `make rq2-report` |
| Figure 3 (RQ3) | `results/full/fig4.{png,pdf}`, caption `fig4_caption.txt` | `make rq3-report` |

Numbers in the text (paths relative to `results/full/`; `rq_csv/` = `results/full/rq_csv/`):

| Section | Statement | Source |
|---|---|---|
| Setup | 500 test users per domain, BPR Recall@20 (14.8 / 9.6 / 9.2 / 11.2 %) | `text_numbers.json` → `n_test_users`, `bpr_recall_at_20` |
| RQ1 | Best ranker per domain; best vs. second best (not significant) | `text_numbers.json` → `injected_per_domain`; `rq1_within_dm_tests.csv` (`best_vs_second_dm`) |
| RQ1 | Decision models vs. LLM rankers (.018–.058) and PLM rankers (.065, .030), per domain | `rq1_class_tests.csv` (`class_average`) |
| RQ1 | On Yelp no ranker significantly above a random ranking | `rq1_vs_expected_random.csv` |
| RQ1 | Jev, MS-Decision-1, Perplexity Decider lead (.216–.230); OpenAI Decisions at random level | `text_numbers.json` → `pooled`; `rq1_within_dm_tests.csv` (`jev_vs_dm`, `openai_vs_random`) |
| RQ1 | Open-weight vs. closed-source decision models (.202 vs. .204; ML-1M, Games) | `rq1_open_vs_closed_dm.csv` |
| RQ1 | Decision models vs. their backbone LLMs (.039, .057, .025) | `rq1_dm_vs_backbone.csv` (scope `pooled`) |
| RQ1 | Qwen3.5-4B-Base compliance (69.5 %) | `text_numbers.json` → `llm_compliance_rate` (see [parsing](#llm-output-parsing)) |
| RQ1 | Clef-flash / Perplexity Decider vs. Qwen3.8-27B and GPT-6 Luna; OpenAI Decisions vs. GPT-6 Luna | `rq1_dm_vs_large_llm.csv` (`injection_oracle`) |
| RQ1 | 168 retrieved users; class differences .039 / .097 | `rq1_class_tests_retrieved.csv` |
| RQ1 | η of GPT-6 Luna vs. Jev (.333 vs. .341); no ranker better than BPR (η = .285) | `text_numbers.json` → `pooled`; `rq1_vs_bpr_retrieved.csv` |
| RQ1 | Sequential recommenders: ML-1M from 10 %; Games/Yelp below; Realistic pooled SASRec 25 %, GRU4Rec 50 %, BERT4Rec 75 % | `rq1_seq_crossover_tests.csv`, `rq1_seq_crossover_pooled.csv`, `rq1_seq_crossover_thresholds.csv` |
| RQ2 | Drop without a history (31–60 %); at or below Random | `rq_csv/rq2_no_history_drop.csv` (scope `pooled`); `rq2_nohistory_vs_random.csv` |
| RQ2 | Other user's / random history (≤ .037, 2 of 12 significant; GPT-6 Luna on) | `rq_csv/rq2_tests_conditions.csv`, `rq_csv/rq2_change_table.csv` |
| RQ2 | Top-1 change after reordering (31–54 %) vs. rerun (≤ 11 %; LLMs 36–81 %) | `rq_csv/rq2_change_table.csv` (`top1_change`, `noise_top1_change`, scope `pooled`) |
| RQ2 | History length 5 → 40 | `rq_csv/rq2_tests_history_length.csv`, `rq_csv/rq2_history_length_curves.csv` |
| RQ3 | Primacy gap .016 vs. .085; position tests | `rq_csv/rq3_class_tests.csv` (`primacy_gap`), `rq_csv/rq3_tests_position.csv`, `rq_csv/rq3_position_ndcg.csv` |
| RQ3 | Label following (≤ 3.7 % vs. 10.6 %) | `rq_csv/rq3_label_swap.csv`, `rq_csv/rq3_tests_label.csv`, `rq_csv/rq3_class_tests.csv` (`follows_label`) |
| RQ3 | Popularity of the top-1 (.91–.93) vs. targets (.78) and candidates (.916) | `rq_csv/rq3_tests_popularity.csv`, `rq_csv/rq3_popularity_by_rank.csv` |
| RQ3 | Decision models vs. LLM rankers as classes (all RQ3 measures) | `rq_csv/rq3_class_tests.csv`, `rq_csv/rq3_class_measures_per_model.csv` |
| Repo | Latency | `latency.csv`, `latency_factors.csv` |
| Repo | NDCG@10 / @20 | `results/full/k10/`, `results/full/k20/` (same files as above) |

All statistics in the CSVs carry the unadjusted `p`, the Holm-adjusted `p_holm` and `significant` (α = .05).

## Models and versions

URLs last visited 16 Nov 2026. Pinned versions are also in `configs/models/*.yaml` and in every result row
(`model_version`).

| Model | Type | URL | Version used |
|---|---|---|---|
| Jev | decision, closed | https://docs.typesafe.ai/concepts/system-one | `jev-1.14.0` |
| OpenAI Decisions | decision, closed | https://developers.openai.com/api/docs/guides/decisions | `gpt-6-luna-decisions` |
| MS-Decision-1 | decision, closed | https://openrouter.ai/microsoft/microsoft-decision-1 | `microsoft-decision-1-20261009`, provider pinned to Azure |
| Clef-flash | decision, open (backbone Qwen3.5-9B) | https://blog.cloudflare.com/clef-decision-models/ | `Cloudflare/clef-flash@22b7550` |
| Perplexity Decider flash v1.2 | decision, open (backbone Qwen3.5-4B) | https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2 | `@9ff7cf9` |
| Kev-4B | decision, open (backbone Qwen3.5-4B-Base) | https://huggingface.co/jaredpalmer/kev-4b | `@6cfce5c` |
| GPT-6 Luna (reasoning on / off) | LLM | OpenAI API | `gpt-6-luna`, `reasoning_effort` medium / none |
| Qwen3.8-27B | LLM | OpenAI-compatible endpoint | `qwen3.8-27b` |
| Qwen3.5-9B / 4B / 4B-Base | LLM | Hugging Face | revisions `c202236`, `851bf6e`, `1001bb4` |
| EmbSim | PLM | Qwen/Qwen3-Embedding-0.6B | `@97b0c61` |
| CrossEnc | PLM | Qwen/Qwen3-Reranker-0.6B | `@e61197e` |
| ItemCLS | PLM | MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli | `@b3546ea` |

Sequential recommenders (RecBole 1.2.1: SASRec, BERT4Rec, GRU4Rec) are tuned once at 100 % of the training users
(seed 0) and the chosen settings are reused for every share; BPR is tuned on validation Recall@20
(`data/full/candidates/*.meta.json`).

Related: Lyu et al. (arXiv 2609.40241) also include a single-pass label-logit LLM baseline, relevant to the question
whether the decision interface or the decoding makes the difference.

## Prompts

The paper paraphrases the instruction; these are the exact texts (`src/dm4recsys/prompts/render.py`). `{noun}` is
*movies* (ML-1M), *products* (Games) or *businesses* (Yelp). Titles longer than 200 characters are clipped.

**Decision models.** The history block is the context, the question is the instruction and the 20 candidates are
the options, labelled with their identifiers:

```
state / input:
A user has interacted with the following {noun}, in chronological order from oldest to newest:
1. <title>
...
10. <title>

instructions:
Which of the options is the user most likely to interact with next? Rank all 20 options from most to least likely.

options: {"1": "<title>", ..., "20": "<title>"}
```

Jev, MS-Decision-1 and the three open models receive the System One body
`{"state", "questions": {"next_item": {"type": "choice", "instructions", "criteria": {id: title}}}}`;
OpenAI Decisions receives `{"input", "questions": [{"type": "choice", "name", "instructions",
"choices": [{"value": id, "description": title}]}]}` (`src/dm4recsys/models/decision/`).

**LLM rankers.** One user message, no system message (Qwen3.5-4B-Base: the same text via `/v1/completions`):

```
A user has interacted with the following {noun}, in chronological order from oldest to newest:
1. <title>
...
10. <title>

Question: Which of the options is the user most likely to interact with next? Rank all 20 options from most to least likely.

Options:
[1] <title>
...
[20] <title>

Output all 20 IDs exactly once, in the format [ID], [ID], ... Output only the IDs, without explanations.
```

**PLM rankers** score each candidate separately with the same history block. EmbSim: cosine between
`Instruct: {instruction}\nQuery:{history}` and the title; CrossEnc: P("yes") for
`<Instruct>: {instruction}\n<Query>: {history}\n<Document>: {title}` in the model card's template; ItemCLS:
entailment probability of "The user will {watch | buy | visit} {title} next." given the history. Instruction:
"Given a user's interaction history, retrieve the {movie | product | business} the user will interact with next".

## Data and candidates

* **Datasets.** ML-1M; Amazon Reviews 2023 Video Games; Yelp Open Dataset restricted to Philadelphia. Chronological
  order (equal timestamps keep the raw file order), first of repeated (user, item) pairs kept, items without a title
  dropped, 5-core filtering. After filtering: ML-1M 6,040 users / 3,416 items / 999,611 interactions; Games
  94,745 / 25,610 / 814,461; Yelp 35,879 / 10,949 / 562,413 (`data/full/processed/*/stats.json`).
* **Split.** Each user's last interaction is the test target, the second-to-last the validation target. 500 test
  users per domain are drawn at random (`data.sample_seed` 2027; 100 further users for trial runs only).
* **Retriever.** BPR (embedding size 64, learning rate from {0.0005, 0.001, 0.002}) is tuned on the validation
  Recall@20 with the last two interactions held out, then retrained on all interactions except the test targets.
  Items in the user's history are excluded.
* **Candidates.** Injection Oracle: the target plus the 19 highest-scoring other items. Realistic Retrieval: BPR's
  top 20; for users whose target is among them (r_u = 1) the two sets are identical, so the Injection Oracle run is
  reused and realistic NDCG = injected NDCG × r_u. The candidates are shuffled once per user (`candidates.shuffle_seed`)
  and this order defines the identifiers [1]–[20]. Candidate files are frozen and checked by sha256 at every load.
* **History.** The titles of the 10 most recent interactions before the target, oldest first.
* **Long-history sample (RQ2).** The 500 users per domain with the longest histories; their shortest history has
  439 (ML-1M), 48 (Games) and 110 (Yelp) items, so Games is run up to 40 items (`rq_csv/rq2_history_length_caps.json`).
* **Popularity** = the number of training interactions (all but each user's test target).

## LLM output parsing

`src/dm4recsys/models/llm/parser.py`: items are read in the order written, by bracketed ID (bare numbers only if no
bracketed ID appears) or by title; the first mention counts, invalid IDs are dropped, and candidates the model did
not name are appended as **one tied block** at the end, which tie-aware NDCG treats as equally likely at all of
those ranks. A response is **compliant** if it names at least 5 distinct candidates (so the top 5 always come from
the model); **complete** if it names all 20 exactly once with nothing invalid.

| LLM | compliant | complete |
|---|---|---|
| GPT-6 Luna (on) | 99.4 % | 99.4 % |
| GPT-6 Luna (off) | 100 % | 99.9 % |
| Qwen3.8-27B | 100 % | 98.2 % |
| Qwen3.5-9B | 100 % | 91.6 % |
| Qwen3.5-4B | 100 % | 90.6 % |
| Qwen3.5-4B-Base | 69.5 % | 55.8 % |

(4,500 responses each; `text_numbers.json` → `llm_compliance_rate`; per row `compliant` and `info`.)

## Sampling, runs and determinism

* **LLM rankers:** temperature 0.7, top-p 0.8, top-k 20 (Qwen, thinking off), request seed 2027 + run;
  GPT-6 Luna with reasoning at the provider's default temperature (only value accepted), max 4,000 completion tokens.
* **Decision models** return probabilities; candidates are ranked by them, and equal probabilities are tied (Jev
  rounds to two decimals). If the API names its own choice and that choice is within 0.01 of the highest
  probability, it is ranked first.
* **Three runs per user** for every zero-shot ranker; NDCG@5 is averaged per user over the runs before any test.
  Run-to-run agreement of the top-1 item (pooled, `text_numbers.json` → `run_to_run_stability`): OpenAI Decisions
  and the three open decision models 100 % (identical rankings); Jev 84 %, MS-Decision-1 87 %, i.e. these two APIs
  are not deterministic, which is why RQ2/RQ3 compare every change with a plain rerun.
* No API call was refused in RQ1 (`text_numbers.json` → `api_refusals_and_checks`); RQ2 refusals are in
  `rq_csv/rq_refusal_share.csv` (a refused call scores 0).

## Statistics

Paired t-tests over users on the run-averaged NDCG@5, two-sided, α = .05, Holm correction within each family:

| Family | Tests | File |
|---|---|---|
| Best vs. second-best ranker | 3 domains | `text_numbers.json` → `injected_per_domain` |
| Best decision model vs. best LLM / PLM | 6 (3 domains × 2) | `rq1_class_tests.csv` (`best_vs_best`) |
| Class means (decision / LLM / PLM) | 9 (3 domains × 3) | `rq1_class_tests.csv` (`class_average`) |
| Class means, retrieved users | 3 | `rq1_class_tests_retrieved.csv` |
| Each decision model vs. Qwen3.8-27B / GPT-6 Luna on / off | 18 per protocol | `rq1_dm_vs_large_llm.csv` |
| Open decision model vs. its backbone | 3 per scope | `rq1_dm_vs_backbone.csv` |
| Within the decision models | 5 (Jev), 5 (OpenAI Decisions), 3 domains (best vs. second; OpenAI Decisions vs. Random) | `rq1_within_dm_tests.csv` |
| Ranker vs. expected random NDCG@5 (one-sample t, 0.1474) | 16 rankers per domain | `rq1_vs_expected_random.csv` |
| Ranker vs. BPR, retrieved users | 15 | `rq1_vs_bpr_retrieved.csv` |
| Sequential vs. best decision model | 6 shares per domain × model × protocol | `rq1_seq_crossover_*.csv` |
| Open vs. closed decision models | 3 domains (pooled separately) | `rq1_open_vs_closed_dm.csv` |
| RQ2 condition vs. Real | 60 (12 models × 5 conditions) | `rq_csv/rq2_tests_conditions.csv` |
| RQ2 no history vs. Random | 12 | `rq2_nohistory_vs_random.csv` |
| RQ2 5 vs. 40 history items | 12 | `rq_csv/rq2_tests_history_length.csv` |
| RQ3 position (Friedman over users) / labels (McNemar) / popularity (paired t) | 12 each | `rq_csv/rq3_tests_*.csv` |
| RQ3 decision models vs. LLM rankers | 6 measures | `rq_csv/rq3_class_tests.csv` |

Bootstrap CIs: 1,000 resamples over users, seed 2027. The tests written by `report/posthoc.py` (the files listed in
its header) were added after the main results to back statements in the text; they use the same per-user data.

## RQ2 and RQ3 conditions

Injection Oracle candidates; the reference "Real" is RQ1 run 0 and every condition changes one thing
(`src/dm4recsys/rq/conditions.py`). A plain **rerun** of the unchanged input is the noise floor. The six decision
models and all six LLM rankers run every condition; Figures 2 and 3 show the six decision models, GPT-6 Luna (on) and
Qwen3.8-27B, while the text values and CSVs cover all twelve.

* **History content:** another test user's history from the same domain that does not contain the target
  (`other_user`), random catalogue items (`random_fake`), no history (`no_history`; the history block is removed).
* **History order:** `shuffled`, `reversed` (the prompt still says "oldest to newest").
* **History length:** the last L ∈ {5, 10, 20, 30, 40} items on the long-history sample.
* **Position:** 19 cyclic shifts of the candidate list (with Real, every target at each of the 20 positions);
  identifiers move with their items; Friedman test over users.
* **Labels:** the identifiers are reassigned by a derangement while items keep their positions; the new top-1 is
  the same item, the item now carrying the old identifier, or another item; McNemar test per model on the change
  rate vs. the rerun.
* **Popularity:** within-domain percentile rank of the training interaction count (ties averaged); the top-1 is
  compared with the true target and with the mean over the 20 candidates (which includes the injected target).
* **Seeds:** seeded LLMs keep Real's request seed in every condition; the rerun uses seed 2027 + 3, so it is a fresh
  sample. For them, change measured against the rerun is therefore conservative.

## Latency

`latency.csv`: per call, the successful attempt only (`latency_s`; retries and back-off excluded), over all
4,500 RQ1 calls per model, with median, mean, p90, concurrency and deployment. Local models ran on one GPU server
(NVIDIA GB10, bf16; vLLM 0.33.0 for the Qwen3.5 LLMs, System One servers for the decision models). Calls in flight
slow each other on that server, so `median_one_at_a_time_s` also reports the calls run one at a time (Qwen3.5-9B,
Qwen3.5-4B-Base). `latency_factors.csv`: the backbone LLM's median over its decision model's.

| Model | median (s) | Model | median (s) |
|---|---|---|---|
| Jev | 0.74 | GPT-6 Luna (on) | 9.42 |
| OpenAI Decisions | 0.23 | GPT-6 Luna (off) | 2.26 |
| MS-Decision-1 | 0.34 | Qwen3.8-27B | 2.29 |
| Clef-flash | 0.34 | Qwen3.5-9B | 6.04 |
| Perplexity Decider | 0.20 | Qwen3.5-4B | 3.65 |
| Kev-4B | 0.13 | Qwen3.5-4B-Base | 3.54 |

## NDCG@10 and NDCG@20

All results are recomputed at NDCG@10 and @20 from the stored tie ranks, without new model calls
(`results/full/k10/`, `results/full/k20/`, same file names). With 20 candidates, @20 covers the whole list
(Hit@20 = 1, a random ranking reaches .355), which is why the paper reports @5. The main conclusions hold at both
cutoffs; the crossover thresholds differ (`rq1_seq_crossover_thresholds.csv`; pooled Realistic vs. Jev):

| Cutoff | SASRec | GRU4Rec | BERT4Rec |
|---|---|---|---|
| NDCG@5 | 25 % | 50 % | 75 % |
| NDCG@10 | 25 % | 50 % | 100 % |
| NDCG@20 | 10 % | 50 % | 100 % |

## Layout

```
configs/      base.yaml (protocol, report settings), server.yaml, datasets/*.yaml, models/*.yaml
src/dm4recsys/
  data/       download, loaders, preprocessing (5-core, order), splits
  retrieval/  BPR, frozen candidates
  prompts/    history/candidate transforms, prompt texts
  models/     decision/, llm/ (client, parser), plm/, sequential/, reference/ (BPR order, Popularity, Random)
  run/        resumable runner, raw logs, latency, repeats
  eval/       metrics (tie-aware NDCG), stats, loading
  report/     Table 1, Figure 1, text numbers (build.py), post-hoc tests (posthoc.py), figure style
  rq/         RQ2/RQ3 conditions, runner, analysis and Figures 2/3 (fig3/fig4)
scripts/      one CLI per step (prepare_data, make_candidates, train_sequential, run_model, run_conditions,
              make_report, local_serve.sh)
results/full/ outputs (see the mapping above)
```
