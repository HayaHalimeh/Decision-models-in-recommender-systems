"""Post-hoc analyses (added after the main results; LOCKED_SETTINGS.md marks them [paper]), written as CSVs to
results/{mode}/ by `make posthoc` (also part of `make report`). No model is called; everything is read from runs/.

  rq1_dm_vs_large_llm.csv        each decision model vs. Qwen3.8-27B and GPT-6 Luna (thinking on / off); Injection
                                 Oracle over all test users, Realistic Retrieval over the users whose target BPR
                                 retrieved (r_u = 1); Holm over the 18 pairs per protocol
  rq1_within_dm_tests.csv        Jev vs. each other decision model, OpenAI Decisions vs. each other decision model
                                 (pooled, Holm over 5 each), OpenAI Decisions vs. Random (pooled; per domain, Holm
                                 over 3), best vs. second-best decision model per domain (Holm over 3)
  rq1_seq_crossover_tests.csv    per domain: each sequential model at each training share vs. the domain's best
                                 decision model (Injection Oracle: all users; Realistic Retrieval: r_u = 1), Holm
                                 over the six shares; at 100 % also vs. Random (Injection Oracle)
  rq1_seq_crossover_pooled.csv   Realistic Retrieval, pooled: each sequential model and their per-user average vs.
                                 the best pooled decision model, on all retrieved users and on the Games + Yelp ones
  rq1_seq_crossover_thresholds.csv  the smallest training share from which a sequential model is significantly
                                 above the decision model at every larger share (from the two files above)
  rq1_vs_expected_random.csv     Injection Oracle, per domain: each ranker vs. the expected NDCG@k of a random
                                 ranking (one-sample t), Holm over the rankers per domain
  rq1_open_vs_closed_dm.csv      mean of the open-weight vs. mean of the closed-source decision models per user
                                 (paired t), per domain and pooled
  rq1_dm_vs_backbone.csv         each open decision model vs. its backbone LLM (pair_llm), Injection Oracle, pooled
                                 and per domain, Holm over the pairs per scope
  rq1_class_tests_retrieved.csv  Realistic Retrieval on the users with r_u = 1: class means (decision vs. LLM vs.
                                 PLM) per user, paired t, Holm over the three comparisons
  rq1_vs_bpr_retrieved.csv       Realistic Retrieval on the users with r_u = 1: every zero-shot ranker vs. BPR's own
                                 order (= eta vs. BPR's eta), Holm over the rankers
  rq2_nohistory_vs_random.csv    RQ2 "no history" vs. Random (Injection Oracle, paired over users), Holm over models
  latency.csv                    per-call latency (successful attempt only) per model, with deployment/hardware,
                                 and latency_factors.csv: LLM backbone vs. its decision model
At the robustness cutoffs (report.robustness_k, e.g. NDCG@10/@20) the three crossover files are also written to
results/{mode}/k{k}/.
"""
from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd
import scipy.stats as st

from ..config import model_cfg
from ..eval.metrics import ndcg_at_k
from ..eval.stats import holm, paired_t
from ..run.runner import n_repeats, result_path
from ..utils import read_jsonl
from .build import Report

log = logging.getLogger(__name__)


def expected_random_ndcg(n: int, k: int) -> float:
    """NDCG@k of a uniformly random ranking of n options with one relevant item."""
    return float(np.mean([ndcg_at_k(r, k) for r in range(1, n + 1)]))


def _holm_into(rows: list[dict], alpha: float, sig: str = "significant") -> list[dict]:
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = p
        r[sig] = bool(p < alpha)
    return rows


def first_share(fracs: list, rows: pd.DataFrame) -> object:
    """Smallest training share from which the sequential model is significantly ABOVE the decision model at that
    and every larger share ('never' if it is not significantly above at 100 %)."""
    ok = {int(r.frac): bool(r.sig and r["diff"] > 0) for _, r in rows.iterrows()}
    out = "never"
    for f in sorted(fracs, reverse=True):
        if not ok.get(int(f), False):
            break
        out = int(f)
    return out


class PostHoc(Report):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.retrieved = self.cands.set_index(["dataset", "user"])["r_u"]

    # ------------------------------------------------------------------ helpers
    def vals(self, m: str, d: str | None = None, retrieved_only: bool = False) -> pd.Series:
        """Per-user NDCG@k (mean over runs), Injection Oracle; retrieved_only: users with r_u = 1 (where Realistic
        Retrieval = Injection Oracle)."""
        v = self.user_vals(self.zs[m], d)
        return v[self.retrieved.loc[v.index].values == 1] if retrieved_only else v

    def seq_vals(self, s: str, frac, d: str | None = None, retrieved_only: bool = False) -> pd.Series:
        """Per-user NDCG@k of a sequential model at one training share, mean over the seeds."""
        x = self.seq[s]
        x = x[(x["frac"] == frac) & ((x["dataset"] == d) if d else True)]
        v = x.groupby(["dataset", "user"])["ndcg"].mean()
        v.index = list(v.index)
        return v[self.retrieved.loc[v.index].values == 1] if retrieved_only else v

    def ttest(self, a: pd.Series, b: pd.Series) -> dict:
        idx = a.index.intersection(b.index)
        return paired_t(a.loc[idx].values, b.loc[idx].values)

    def best_dm(self, d: str | None, realistic: bool) -> str:
        score = lambda m: float(np.mean(self.user_vals(self.zs[m], d, real=realistic)))
        return max(self.decision, key=score)

    # ------------------------------------------------------------------ 1. decision models vs. large LLMs
    def dm_vs_large_llm(self) -> pd.DataFrame:
        llms = [m for m in self.cfg["report"]["posthoc"]["large_llms"] if m in self.zs]
        out = []
        for protocol, ret in (("injection_oracle", False), ("realistic_retrieved_users", True)):
            rows = []
            for dm in self.decision:
                for llm in llms:
                    a, b = self.vals(dm, retrieved_only=ret), self.vals(llm, retrieved_only=ret)
                    t = self.ttest(a, b)
                    rows.append({"protocol": protocol, "decision_model": dm, "llm": llm, "mean_dm": float(a.mean()),
                                 "mean_llm": float(b.mean()), "diff": t["mean_diff"], "n": t["n"], "p": t["p"]})
            out += _holm_into(rows, self.alpha)
        return pd.DataFrame(out)

    # ------------------------------------------------------------------ 2. within the decision models
    def within_dm(self) -> pd.DataFrame:
        def row(family, scope, a, b, d=None):
            va, vb = self.vals(a, d), self.vals(b, d)
            t = self.ttest(va, vb)
            return {"family": family, "scope": scope, "a": a, "b": b, "mean_a": float(va.mean()),
                    "mean_b": float(vb.mean()), "diff": t["mean_diff"], "p": t["p"]}

        top, low = "jev", "openai_decisions"
        out = _holm_into([row("jev_vs_dm", "pooled", top, m) for m in self.decision if m != top], self.alpha)
        out += _holm_into([row("openai_vs_dm", "pooled", low, m) for m in self.decision if m != low], self.alpha)
        out += _holm_into([row("openai_vs_random", "pooled", low, "random")], self.alpha)
        bs = []
        for d in self.datasets:
            ranked = sorted(self.decision, key=lambda m: -float(self.vals(m, d).mean()))
            bs.append(row("best_vs_second_dm", d, ranked[0], ranked[1], d))
        out += _holm_into(bs, self.alpha)
        out += _holm_into([row("openai_vs_random_dom", d, low, "random", d) for d in self.datasets], self.alpha)
        return pd.DataFrame(out)

    # ------------------------------------------------------------------ 3. sequential crossover
    def seq_crossover(self) -> pd.DataFrame:
        fracs = list(self.cfg["sequential"]["fractions"])
        out = []
        for d in self.datasets:
            for protocol, ret in (("oracle", False), ("realistic", True)):
                dm = self.best_dm(d, realistic=ret)
                for s in self.seq:
                    rows = []
                    for f in fracs:
                        t = self.ttest(self.seq_vals(s, f, d, ret), self.vals(dm, d, ret))
                        rows.append({"protocol": protocol, "dataset": d, "seq": s, "frac": f, "vs": dm,
                                     "diff": t["mean_diff"], "n": t["n"], "p": t["p"]})
                    out += _holm_into(rows, self.alpha, "sig")
            for s in self.seq:  # at 100 %: vs. Random (single test per dataset and model)
                t = self.ttest(self.seq_vals(s, 100, d), self.vals("random", d))
                out.append({"protocol": "oracle_vs_random", "dataset": d, "seq": s, "frac": 100, "vs": "random",
                            "diff": t["mean_diff"], "n": t["n"], "p": t["p"], "p_holm": t["p"],
                            "sig": bool(t["p"] < self.alpha)})
        return pd.DataFrame(out)

    def seq_crossover_pooled(self) -> pd.DataFrame:
        fracs = list(self.cfg["sequential"]["fractions"])
        dm = self.best_dm(None, realistic=True)
        scopes = {"pooled": list(self.datasets),
                  "games+yelp": [d for d in self.datasets if d in ("games", "yelp")]}
        out = []
        for scope, ds in scopes.items():
            if not ds:
                continue
            base = pd.concat([self.vals(dm, d, True) for d in ds])
            for s in list(self.seq) + ["avg"]:
                rows = []
                for f in fracs:
                    if s == "avg":
                        v = pd.concat([pd.concat([self.seq_vals(x, f, d, True) for d in ds]) for x in self.seq],
                                      axis=1).mean(axis=1)
                    else:
                        v = pd.concat([self.seq_vals(s, f, d, True) for d in ds])
                    t = self.ttest(v, base)
                    rows.append({"scope": scope, "n": t["n"], "best_dm": dm, "seq": s, "frac": f,
                                 "diff": t["mean_diff"], "p": t["p"]})
                out += _holm_into(rows, self.alpha, "sig")
        return pd.DataFrame(out)

    def thresholds(self, dom: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
        fracs = list(self.cfg["sequential"]["fractions"])
        out = []
        for (protocol, d, s), g in dom[dom["protocol"] != "oracle_vs_random"].groupby(["protocol", "dataset", "seq"]):
            out.append({"k": self.k, "protocol": protocol, "scope": d, "seq": s, "vs": g["vs"].iloc[0],
                        "significantly_above_from_share": first_share(fracs, g)})
        for (scope, s), g in pooled.groupby(["scope", "seq"]):
            out.append({"k": self.k, "protocol": "realistic", "scope": scope, "seq": s, "vs": g["best_dm"].iloc[0],
                        "significantly_above_from_share": first_share(fracs, g)})
        return pd.DataFrame(out)

    # ------------------------------------------------------------------ 5. rankers vs. a random ranking
    def vs_expected_random(self) -> pd.DataFrame:
        n_opt = int(self.cfg["candidates"]["n"])
        mu = expected_random_ndcg(n_opt, self.k)
        rankers = [m for m in self.cfg["report"]["table1_rows"] if m in self.zs and m != "random"]
        out = []
        for d in self.datasets:
            rows = []
            for m in rankers:
                v = self.vals(m, d).values
                t, p = st.ttest_1samp(v, mu)
                rows.append({"dataset": d, "model": m, "group": model_cfg(self.cfg, m)["group"], "mean": float(v.mean()),
                             "expected_random": mu, "diff": float(v.mean() - mu), "n": len(v), "t": float(t),
                             "p": float(p)})
            out += _holm_into(rows, self.alpha)
        return pd.DataFrame(out)

    # ------------------------------------------------------------------ 6. open vs. closed decision models
    def open_vs_closed(self) -> pd.DataFrame:
        grp = {g: [m for m in self.decision if bool(model_cfg(self.cfg, m).get("open")) == (g == "open")]
               for g in ("closed", "open")}
        rows = []
        for scope in self.datasets + ["pooled"]:
            d = None if scope == "pooled" else scope
            mean = {g: pd.concat([self.vals(m, d) for m in ms], axis=1).mean(axis=1) for g, ms in grp.items()}
            t = self.ttest(mean["closed"], mean["open"])
            rows.append({"scope": scope, "closed_models": " ".join(grp["closed"]), "open_models": " ".join(grp["open"]),
                         "mean_closed": float(mean["closed"].mean()), "mean_open": float(mean["open"].mean()),
                         "diff": t["mean_diff"], "n": t["n"], "t": t["t"], "p": t["p"]})
        dom = _holm_into([r for r in rows if r["scope"] != "pooled"], self.alpha)  # Holm over the domains
        for r in dom:
            r["p_holm_domains"] = r.pop("p_holm")
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ backbones, retrieved users, BPR
    def dm_vs_backbone(self) -> pd.DataFrame:
        pairs = [(m, model_cfg(self.cfg, m)["pair_llm"]) for m in self.decision
                 if model_cfg(self.cfg, m).get("pair_llm") in self.zs]
        out = []
        for scope in ["pooled"] + self.datasets:
            d = None if scope == "pooled" else scope
            rows = []
            for dm, llm in pairs:
                a, b = self.vals(dm, d), self.vals(llm, d)
                t = self.ttest(a, b)
                rows.append({"scope": scope, "decision_model": dm, "backbone_llm": llm, "mean_dm": float(a.mean()),
                             "mean_llm": float(b.mean()), "diff": t["mean_diff"], "n": t["n"], "p": t["p"]})
            out += _holm_into(rows, self.alpha)
        return pd.DataFrame(out)

    def class_tests_retrieved(self) -> pd.DataFrame:
        groups = {g: [m for m in self.cfg["report"]["table1_rows"] if m in self.zs and model_cfg(self.cfg, m)["group"] == g]
                  for g in ("decision", "llm", "plm")}
        mean = {g: pd.concat([self.vals(m, retrieved_only=True) for m in ms], axis=1).mean(axis=1)
                for g, ms in groups.items()}
        rows = []
        for a, b in (("decision", "llm"), ("decision", "plm"), ("llm", "plm")):
            t = self.ttest(mean[a], mean[b])
            rows.append({"comparison": f"{a} vs. {b}", "users": "r_u=1", "mean_a": float(mean[a].mean()),
                         "mean_b": float(mean[b].mean()), "diff": t["mean_diff"], "n": t["n"], "p": t["p"]})
        return pd.DataFrame(_holm_into(rows, self.alpha))

    def vs_bpr_retrieved(self) -> pd.DataFrame:
        rankers = [m for m in self.cfg["report"]["table1_rows"]
                   if m in self.zs and model_cfg(self.cfg, m)["group"] in ("decision", "llm", "plm")]
        bpr = self.vals("bpr", retrieved_only=True)
        rows = []
        for m in rankers:
            v = self.vals(m, retrieved_only=True)
            t = self.ttest(v, bpr)
            rows.append({"model": m, "eta": float(v.mean()), "eta_bpr": float(bpr.mean()), "diff": t["mean_diff"],
                         "n": t["n"], "p": t["p"]})
        return pd.DataFrame(_holm_into(rows, self.alpha))

    # ------------------------------------------------------------------ 4. RQ2 no history vs. Random
    def nohistory_vs_random(self) -> pd.DataFrame:
        from ..rq.collect import RQData

        data = RQData(self.cfg, self.datasets)
        rnd = self.vals("random")
        rows = []
        for m in self.cfg["rq"]["models"]:
            df = data.get(m, "no_history")
            if df.empty:
                continue
            v = df.groupby(["dataset", "user"])["ndcg"].mean()
            v.index = list(v.index)
            t = self.ttest(v, rnd)
            idx = v.index.intersection(rnd.index)
            rows.append({"model": m, "n": t["n"], "nohist": float(v.loc[idx].mean()), "random": float(rnd.loc[idx].mean()),
                         "diff": t["mean_diff"], "p": t["p"]})
        return pd.DataFrame(_holm_into(rows, self.alpha, "sig"))

    # ------------------------------------------------------------------ 7. latency
    def latency(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        lc = self.cfg["report"]["posthoc"]["latency"]
        rows, med, med1 = [], {}, {}
        for m in lc["models"]:
            mc = model_cfg(self.cfg, m)
            calls = []
            for d in self.datasets:
                for r in range(n_repeats(self.cfg, mc)):
                    calls += [(x["latency_s"], x.get("concurrency")) for x in read_jsonl(result_path(self.cfg, d, m, r))]
            lat = np.array([c[0] for c in calls], float)
            one = np.array([c[0] for c in calls if c[1] in (None, 1)], float)  # rows without the field ran one at a time
            conc = sorted({c[1] or 1 for c in calls})
            med[m] = float(np.median(lat))
            med1[m] = float(np.median(one)) if len(one) else np.nan
            rows.append({"model": m, "display": mc.get("display", m), "group": mc["group"],
                         "deployment": mc.get("deployment"), "hardware": lc["hardware"].get(m, ""),
                         "concurrency": "/".join(map(str, conc)), "n_calls": len(lat),
                         "median_s": med[m], "mean_s": float(lat.mean()), "p90_s": float(np.quantile(lat, 0.9)),
                         "n_calls_one_at_a_time": len(one),
                         "median_one_at_a_time_s": med1[m]})
        pairs = [{"decision_model": dm, "llm": llm, "same": why, "median_dm_s": med[dm], "median_llm_s": med[llm],
                  "factor": med[llm] / med[dm], "factor_one_at_a_time": med1[llm] / med1[dm]} for dm, llm, why in lc["pairs"] if dm in med and llm in med]
        return pd.DataFrame(rows), pd.DataFrame(pairs)

    # ------------------------------------------------------------------
    def build(self) -> None:
        out = self.out
        files = {"rq1_dm_vs_large_llm.csv": self.dm_vs_large_llm(), "rq1_within_dm_tests.csv": self.within_dm()}
        files.update(self.crossover_files())
        files["rq1_vs_expected_random.csv"] = self.vs_expected_random()
        files["rq1_open_vs_closed_dm.csv"] = self.open_vs_closed()
        files["rq1_dm_vs_backbone.csv"] = self.dm_vs_backbone()
        files["rq1_class_tests_retrieved.csv"] = self.class_tests_retrieved()
        files["rq1_vs_bpr_retrieved.csv"] = self.vs_bpr_retrieved()
        files["rq2_nohistory_vs_random.csv"] = self.nohistory_vs_random()
        files["latency.csv"], files["latency_factors.csv"] = self.latency()
        for name, df in files.items():
            df.to_csv(out / name, index=False)
        log.info("Post-hoc CSVs written to %s: %s", out, ", ".join(files))

    def crossover_files(self) -> dict[str, pd.DataFrame]:
        dom, pooled = self.seq_crossover(), self.seq_crossover_pooled()
        return {"rq1_seq_crossover_tests.csv": dom, "rq1_seq_crossover_pooled.csv": pooled,
                "rq1_seq_crossover_thresholds.csv": self.thresholds(dom, pooled)}


def build_posthoc(cfg: dict) -> None:
    PostHoc(cfg).build()
    base = cfg["paths"]["results"]
    for k in cfg["report"].get("robustness_k", []):  # crossover tests at NDCG@10 / @20 -> results/{mode}/k{k}/
        c = copy.deepcopy(cfg)
        c["eval"]["k"] = int(k)
        c["paths"]["results"] = f"{base}/k{k}"
        p = PostHoc(c)
        for name, df in p.crossover_files().items():
            df.to_csv(p.out / name, index=False)
        log.info("Crossover tests at NDCG@%d written to %s", k, p.out)
