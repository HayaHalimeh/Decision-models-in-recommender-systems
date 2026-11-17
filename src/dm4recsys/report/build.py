"""Build Table 1, Figure 1 and text_numbers.json from the per-user results."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import model_cfg, models_in_group, path  # noqa: E402
from ..data.store import DatasetStore  # noqa: E402
from ..eval import collect  # noqa: E402
from ..eval.crossover import crossover, curve  # noqa: E402
from ..eval.metrics import eta, realistic, recall  # noqa: E402
from ..eval.stats import bootstrap_ci, holm, paired_t, paired_wilcoxon  # noqa: E402
from ..utils import write_json  # noqa: E402
from . import figstyle  # noqa: E402

log = logging.getLogger(__name__)
# names of the two candidate settings in tables, captions and figures
SETTING = {"injected": "Injection Oracle", "realistic": "Realistic Retrieval"}


def _fmt(x: float, nd: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _tex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def check_bpr_injected_equals_realistic(df: pd.DataFrame) -> None:
    """Per user: BPR's injected NDCG equals its realistic NDCG (= injected x r_u)."""
    bad = df[~np.isclose(df["ndcg"].values, realistic(df["ndcg"].values, df["r_u"].values))]
    if len(bad):
        raise AssertionError(f"BPR injected != realistic NDCG for {len(bad)} users, e.g. {bad.iloc[0].to_dict()}")


class Report:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.datasets = list(cfg["run_datasets"])
        self.out = path(cfg, "results")
        self.out.mkdir(parents=True, exist_ok=True)
        self.k = cfg["eval"]["k"]
        self.B, self.bseed, self.alpha = cfg["eval"]["bootstrap_resamples"], cfg["eval"]["bootstrap_seed"], cfg["eval"]["alpha"]
        self.cands = collect.candidate_frame(cfg, self.datasets)
        self.dnames = {d: cfg["datasets"][d]["display"] for d in self.datasets}
        self.zs: dict[str, pd.DataFrame] = {}
        self.rows_all = list(dict.fromkeys(cfg["report"]["table1_rows"] + ["bpr"]))  # BPR: Realistic Retrieval baseline
        for m in self.rows_all:
            if m in cfg["models"] and cfg["models"][m].get("enabled", True):
                df = collect.zero_shot(cfg, m, self.datasets, self.cands)
                if df is not None:
                    self.zs[m] = df
        self.seq: dict[str, pd.DataFrame] = {}
        for m in cfg["sequential"]["models"]:
            df = collect.sequential(cfg, m, self.datasets, self.cands)
            if df is not None:
                self.seq[m] = df
        if "bpr" in self.zs:
            check_bpr_injected_equals_realistic(self.zs["bpr"])
        self.missing = [m for m in self.rows_all if m not in self.zs] + \
                       [m for m in cfg["sequential"]["models"] if m not in self.seq]
        self.decision = [m for m in models_in_group(cfg, "decision") if m in self.zs]

    def meta(self, m):
        return model_cfg(self.cfg, m)

    def display(self, m):
        return self.meta(m).get("display", m)

    def user_vals(self, df: pd.DataFrame, d: str | None = None, real: bool = False) -> pd.Series:
        x = df if d is None else df[df["dataset"] == d]
        x = x.sort_values(["dataset", "user"])
        v = realistic(x["ndcg"].values, x["r_u"].values) if real else x["ndcg"].values
        return pd.Series(v, index=list(zip(x["dataset"], x["user"])))

    def best_vs_second(self) -> dict:
        """Injection Oracle, per domain: mean NDCG@k per model, best vs. second best (paired t-test, Holm over the
        domains) and the widths of the 95 % bootstrap CIs. For the text (text_numbers.json), not a table."""
        rows = [m for m in self.cfg["report"]["table1_rows"] if m in self.zs]
        means = {m: {d: float(self.user_vals(self.zs[m], d).mean()) for d in self.datasets} for m in rows}
        widths, tests = [], {}
        for m in rows:
            for d in self.datasets:
                lo, hi = bootstrap_ci(self.user_vals(self.zs[m], d).values, self.B, self.bseed, self.alpha)
                widths.append(hi - lo)
        best, second = {}, {}
        for d in self.datasets:
            ranked = sorted(rows, key=lambda m: -means[m][d])
            best[d] = ranked[0] if ranked else None
            second[d] = ranked[1] if len(ranked) > 1 else None
            if best[d] and second[d]:
                tests[d] = paired_t(self.user_vals(self.zs[best[d]], d).values, self.user_vals(self.zs[second[d]], d).values)
                tests[d].update(best=best[d], second=second[d])
        ds_t = [d for d in self.datasets if d in tests]
        for d, p in zip(ds_t, holm([tests[d]["p"] for d in ds_t])):
            tests[d]["p_holm"] = p
            tests[d]["significant"] = bool(p < self.alpha and tests[d]["mean_diff"] > 0)
        return {"means": means, "tests": tests, "ci_width_max": max(widths) if widths else None,
                "ci_width_median": float(np.median(widths)) if widths else None}

    def _runs_text(self, rows) -> str:
        """Runs per user behind the table's means (from the result files)."""
        runs = {m: int(self.zs[m]["n_runs"].max()) for m in rows}
        if len(set(runs.values())) == 1:
            return f"Mean over {next(iter(runs.values()))} runs per user."
        by_n = {}
        for m, n in runs.items():
            by_n.setdefault(n, []).append(_tex_escape(self.display(m)))
        return "Mean over runs per user (" + "; ".join(f"{n}: {', '.join(ms)}" for n, ms in sorted(by_n.items())) + ")."

    def _n_users_text(self):
        n = self.cands.groupby("dataset").size()
        if n.nunique() == 1:
            return f"{int(n.iloc[0])} test users per domain"
        return ", ".join(f"{self.dnames[d]} $n={int(n[d])}$" for d in self.datasets)

    def bpr_recall(self) -> dict:
        r = {d: recall(self.cands.loc[self.cands["dataset"] == d, "r_u"].values) for d in self.datasets}
        r["pooled"] = recall(self.cands["r_u"].values)
        return r

    def pooled(self) -> dict:
        """Pooled over all test users: Injection Oracle and Realistic Retrieval NDCG@k and eta per model (BPR
        included), and the best zero-shot model vs. BPR on Realistic Retrieval. For the text, not a table."""
        rows = [m for m in self.cfg["report"]["table1_rows"] + ["bpr"] if m in self.zs]
        res = {}
        for m in rows:
            df = self.zs[m].sort_values(["dataset", "user"])
            res[m] = {"injected": float(df["ndcg"].mean()),
                      "realistic": float(realistic(df["ndcg"], df["r_u"]).mean()),
                      "eta": eta(df["ndcg"].values, df["r_u"].values)}
        rec = self.bpr_recall()
        zero_shot = [m for m in rows if self.meta(m)["group"] != "reference"]
        test = None
        if zero_shot and "bpr" in self.zs:
            # test only on users whose target BPR retrieved (r_u = 1); for all others every model
            # scores 0, so their paired difference is 0. Equivalent to testing eta.
            best = max(zero_shot, key=lambda m: res[m]["realistic"])
            a, b = self.user_vals(self.zs[best], real=True), self.user_vals(self.zs["bpr"], real=True)
            ru = self.cands.set_index(["dataset", "user"])["r_u"]
            keep = [i for i in a.index if ru.loc[i] == 1]
            test = paired_t(a.loc[keep].values, b.loc[keep].values)
            test.update(best=best, users="r_u=1", n_retrieved=len(keep), n_users=int(len(self.cands)))
        return {"pooled": res, "bpr_recall": rec, "test_best_vs_bpr": test}

    def figure1(self) -> dict:
        """Sequential models' NDCG@k vs. share of training users (equally spaced; mean over seeds, band = min..max over
        the seeds), with every decision model as a dashed horizontal line. One row at printed size (7 x 1.75 in): the
        three domains under Injection Oracle, a gap, the three under Realistic Retrieval (= injected x r_u)."""
        nd = len(self.datasets)
        fracs = list(self.cfg["sequential"]["fractions"])
        pos = {f: i for i, f in enumerate(fracs)}  # equally spaced training shares
        cmap = plt.get_cmap("tab10")  # sequential models: colours 0-2; decision models: 3 onwards
        rc = {**figstyle.RC, "font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5, "legend.fontsize": 6,
              "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "xtick.major.size": 2, "ytick.major.size": 2,
              "axes.titlepad": 2, "lines.linewidth": 0.4}
        with plt.rc_context(rc):
            # one row: Injection Oracle (3 domains), a gap, Realistic Retrieval (3 domains)
            fig = plt.figure(figsize=(figstyle.PAGE_W, 1.75))
            gs = fig.add_gridspec(1, 2 * nd + 1, width_ratios=[1] * nd + [0.18] + [1] * nd, wspace=0.42)
            groups = []
            for g, real in enumerate((False, True)):
                axes = [fig.add_subplot(gs[0, g * (nd + 1) + i]) for i in range(nd)]
                groups.append(axes)
                for ax, d in zip(axes, self.datasets):
                    for i, m in enumerate(self.seq):
                        sd = self.seq[m][self.seq[m]["dataset"] == d].copy()
                        sd["v"] = realistic(sd["ndcg"], sd["r_u"]) if real else sd["ndcg"]
                        c = curve(sd, "v")
                        x = [pos[f] for f in c["frac"]]
                        ax.plot(x, c["mean"], color=cmap(i), label=self.display(m))
                        ax.fill_between(x, c["min"], c["max"], color=cmap(i), alpha=0.2, lw=0)  # pale band
                    for j, m in enumerate(self.decision):
                        # dashes staggered between models, so two models with the same score both stay visible
                        ax.axhline(float(self.user_vals(self.zs[m], d, real).mean()), ls=(3 * (j % 2), (3, 3)),
                                   lw=0.4, color=cmap(3 + j), label=figstyle.name(self.cfg, m))
                    ax.set_xticks(range(len(fracs)), [str(f) for f in fracs])  # "%" is in the axis label
                    ax.set_xlim(-0.3, len(fracs) - 0.7)
                    ax.set_title(self.dnames[d])
                    ax.grid(lw=0.4)
                axes[0].set_ylabel(f"NDCG@{self.k}\n({SETTING['realistic' if real else 'injected']})")
            fig.subplots_adjust(left=0.07, right=0.995, bottom=0.3, top=0.9)
            for axes in groups:  # shared x label below each group of three panels
                l, r = axes[0].get_position().x0, axes[-1].get_position().x1
                fig.text((l + r) / 2, 0.14, "Percentage of Training Data (%)", ha="center", va="bottom", fontsize=6.5)
            h, lab = groups[0][0].get_legend_handles_labels()
            fig.legend(h, lab, loc="lower center", ncol=len(lab), bbox_to_anchor=(0.5, -0.02), handlelength=1.8,
                       columnspacing=1.0)
            for ext in ("pdf", "png"):
                fig.savefig(self.out / f"fig1.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
        out = {"injected": {}, "realistic": {}}
        # crossovers per domain (each seq model vs each decision model) and pooled (vs best decision model)
        for setting, real in (("injected", False), ("realistic", True)):
            per_dom = {}
            for d in self.datasets:
                per_dom[d] = {}
                for s in self.seq:
                    sd = self.seq[s][self.seq[s]["dataset"] == d].copy()
                    sd["v"] = realistic(sd["ndcg"], sd["r_u"]) if real else sd["ndcg"]
                    c = curve(sd, "v")
                    per_dom[d][s] = {m: crossover(c, float(self.user_vals(self.zs[m], d, real).mean())) for m in self.decision}
            pooled = {}
            if self.decision:
                best = max(self.decision, key=lambda m: self.user_vals(self.zs[m], real=real).mean())
                thr = float(self.user_vals(self.zs[best], real=real).mean())
                pooled = {"best_decision_model": best, "best_decision_ndcg": thr}
                for s in self.seq:
                    sd = self.seq[s].copy()
                    sd["v"] = realistic(sd["ndcg"], sd["r_u"]) if real else sd["ndcg"]
                    c = curve(sd, "v")
                    pooled[s] = crossover(c, thr)
                    pooled[f"{s}_curve"] = dict(zip(map(str, c["frac"]), c["mean"].round(4)))
            out[setting] = {"pooled": pooled, "per_domain": per_dom}
        return out

    def pooled_injected(self) -> dict:
        """Injected NDCG@k pooled over domains per model; sequential models at 100 % training users (mean over seeds)."""
        out = {m: float(df["ndcg"].mean()) for m, df in self.zs.items()}
        for m, df in self.seq.items():
            full = df[df["frac"] == max(self.cfg["sequential"]["fractions"])]
            out[m] = float(full.groupby("seed")["ndcg"].mean().mean())
        return out

    def tie_robustness(self) -> dict:
        """Effect of tied scores: tie-aware NDCG@k (the main metric, Table 1: expected value over all orders of
        each tied block) vs. plain NDCG@k of the one order our seeded tie-break produced, per model with score
        vectors and per LLM (unnamed options = one tied block), per domain and pooled, injected and realistic; plus, for decision models, the share of users with
        fewer than k non-zero options and the share whose target scored exactly 0. Writes tie_robustness.csv."""
        rows, out = [], {}
        for m, df in self.zs.items():
            if not df["has_scores"].all() and self.meta(m)["group"] != "llm":  # Random: a full permutation, no ties
                continue
            out[m] = {}
            for d in [*self.datasets, None]:
                x = df if d is None else df[df["dataset"] == d]
                tie, order = x["ndcg"].values, x["ndcg_order"].values
                r = {"model": m, "dataset": d or "pooled", "n_users": len(x),
                     "ndcg_tie_aware": float(tie.mean()), "ndcg_seeded_tiebreak": float(order.mean()),
                     "diff": float(tie.mean() - order.mean()),
                     "max_abs_user_diff": float(np.abs(tie - order).max()),
                     "share_users_changed": float((np.abs(tie - order) > 1e-12).mean()),
                     "ndcg_tie_aware_realistic": float(realistic(tie, x["r_u"].values).mean()),
                     "ndcg_seeded_tiebreak_realistic": float(realistic(order, x["r_u"].values).mean()),
                     "share_lt_k_nonzero": float(x["lt_k_nonzero"].mean()),
                     "share_target_zero": float(x["target_zero"].mean())}
                rows.append(r)
                out[m][d or "pooled"] = {k: v for k, v in r.items() if k not in ("model", "dataset")}
        if rows:
            pd.DataFrame(rows).to_csv(self.out / "tie_robustness.csv", index=False)
        return out

    def stability(self) -> dict:
        """Run-to-run stability of every model with more than one run (hosted APIs), per domain and pooled: mean over
        users of the SD of NDCG@k across runs, share of users with an identical top 1 / identical full ranking in all
        runs. With score vectors, rankings are compared as blocks of tied options (our seeded tie-break left out),
        so a model whose API returns the same scores every run counts as identical. Writes stability.csv."""
        rows, out = [], {}
        for m, df in self.zs.items():
            if df["n_runs"].max() < 2:
                continue
            out[m] = {}
            for d in [*self.datasets, None]:
                x = df if d is None else df[df["dataset"] == d]
                r = {"model": m, "dataset": d or "pooled", "n_users": len(x), "runs": int(x["n_runs"].max()),
                     "mean_ndcg_sd": float(x["ndcg_sd"].mean()), "share_top1_identical": float(x["top1_same"].mean()),
                     "share_ranking_identical": float(x["ranking_same"].mean())}
                rows.append(r)
                out[m][d or "pooled"] = {k: v for k, v in r.items() if k not in ("model", "dataset")}
        if rows:
            pd.DataFrame(rows).to_csv(self.out / "stability.csv", index=False)
        return out

    def api_health(self) -> dict:
        """Per hosted model and domain: calls (users x runs), OpenAI refusals, failed answer checks."""
        out = {}
        for m, df in self.zs.items():
            if self.meta(m).get("deployment") != "api":
                continue
            out[m] = {d: {"calls": int(df.loc[df["dataset"] == d, "n_runs"].sum()),
                          "refusals": int(df.loc[df["dataset"] == d, "refusals"].sum()),
                          "answer_checks_failed": int(df.loc[df["dataset"] == d, "check_failed"].sum())}
                      for d in self.datasets}
        return out

    def compliance(self) -> dict:
        out = {}
        for m in models_in_group(self.cfg, "llm"):
            if m in self.zs:
                out[m] = float(self.zs[m]["compliant"].mean())
        return out

    def class_tests(self) -> dict:
        """RQ1 tests between model classes on Injection Oracle NDCG@k.

        (a) best vs. best, per domain: the best decision model vs. the best LLM ranker and vs. the best PLM ranker
            ("best" = highest mean NDCG@k in that domain, chosen from the results);
        (b) class averages, per domain: per user, the mean NDCG@k over all models of a class, compared for
            decision vs. LLM, decision vs. PLM and LLM vs. PLM (no model is selected).
        Paired t-test over users, Holm-corrected within (a) and within (b) separately; with each test the 95 %
        bootstrap CI of the mean difference (resampling users, as for all CIs), the shares of users for whom
        a / b is better, and the Wilcoxon signed-rank p (reported for transparency; it tests the typical user,
        not the mean, and can disagree when gains are concentrated in few users). Written to rq1_class_tests.csv and text_numbers.json: class_tests.
        """
        rows = [m for m in self.cfg["report"]["table1_rows"] if m in self.zs]
        classes = {g: [m for m in rows if self.meta(m)["group"] == g] for g in ("decision", "llm", "plm")}
        classes = {g: ms for g, ms in classes.items() if ms}
        vals = {d: {m: self.user_vals(self.zs[m], d) for m in rows} for d in self.datasets}

        def test(a: pd.Series, b: pd.Series) -> dict:
            a, b = a.align(b, join="inner")
            r = paired_t(a.values, b.values)
            lo, hi = bootstrap_ci(a.values - b.values, self.B, self.bseed, self.alpha)
            r.update(mean_a=float(a.mean()), mean_b=float(b.mean()), ci_low=lo, ci_high=hi,
                     share_a_better=float((a.values > b.values).mean()), share_b_better=float((a.values < b.values).mean()),
                     p_wilcoxon=paired_wilcoxon(a.values, b.values))
            return r

        best_vs_best, class_avg, best = [], [], {}
        for d in self.datasets:
            best[d] = {g: max(ms, key=lambda m: float(vals[d][m].mean())) for g, ms in classes.items()}
            for other in ("llm", "plm"):
                if "decision" in classes and other in classes:
                    r = test(vals[d][best[d]["decision"]], vals[d][best[d][other]])
                    r.update(family="best_vs_best", dataset=d, a=best[d]["decision"], b=best[d][other],
                             comparison=f"best decision vs. best {other}")
                    best_vs_best.append(r)
            avg = {g: pd.concat([vals[d][m] for m in ms], axis=1).mean(axis=1) for g, ms in classes.items()}
            for ga, gb in (("decision", "llm"), ("decision", "plm"), ("llm", "plm")):
                if ga in avg and gb in avg:
                    r = test(avg[ga], avg[gb])
                    r.update(family="class_average", dataset=d, a=f"{ga} (mean of {len(classes[ga])})",
                             b=f"{gb} (mean of {len(classes[gb])})", comparison=f"{ga} vs. {gb}")
                    class_avg.append(r)
        for fam in (best_vs_best, class_avg):
            for r, p in zip(fam, holm([r["p"] for r in fam])):
                r["p_holm"] = p
                r["significant"] = bool(p < self.alpha)
                r["a_better"] = bool(r["mean_diff"] > 0)
        cols = ["family", "dataset", "comparison", "a", "b", "mean_a", "mean_b", "mean_diff", "t", "n", "p",
                "p_holm", "significant", "a_better", "ci_low", "ci_high", "share_a_better", "share_b_better", "p_wilcoxon"]
        pd.DataFrame(best_vs_best + class_avg)[cols].to_csv(self.out / "rq1_class_tests.csv", index=False)
        return {"classes": classes, "best_per_class": best, "best_vs_best": best_vs_best, "class_average": class_avg,
                "holm_families": {"best_vs_best": len(best_vs_best), "class_average": len(class_avg)}}

    def _n_users_plain(self) -> str:
        """Test users per domain as a plain number when equal across domains (e.g. "500")."""
        n = self.cands.groupby("dataset").size()
        return str(int(n.iloc[0])) if n.nunique() == 1 else "the"

    def _n_runs_word(self, rows) -> str:
        """Number of runs per user as a word (e.g. "three")."""
        n = {int(self.zs[m]["n_runs"].max()) for m in rows if "n_runs" in self.zs[m]}
        k = n.pop() if len(n) == 1 else 3
        return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}.get(k, str(k))

    def table1(self) -> None:
        """Table 1, the only RQ1 table (Overleaf-ready, one column by default, needs booktabs): NDCG@k under Injection Oracle and
        Realistic Retrieval (x100) per domain and pooled, plus pooled ceiling utilisation eta, for every ranker and
        the references (BPR left out; its Recall@n is the last row, the upper bound of Realistic Retrieval).
        Bold / underline: best / second best among the rankers (decision, LLM, PLM). Significance tests are
        reported in the text (class_tests)."""
        rc = self.cfg["report"]
        rows = [m for m in rc["table1_rows"] if m in self.zs and m != "bpr"]
        rankers = [m for m in rows if self.meta(m)["group"] in ("decision", "llm", "plm")]
        short, sub = rc.get("per_domain_short_display", {}), rc.get("per_domain_subrows", {})
        one_col = rc.get("table1_width", "column") == "column"
        dabbr = rc.get("table1_domain_abbrev", {}) if one_col else {}
        n_c, nd = self.cfg["candidates"]["n"], len(self.datasets)
        scopes = self.datasets + [None]  # None = pooled over domains
        rec = self.bpr_recall()
        val = {}
        for m in rows:
            for d in scopes:
                o, r = self.user_vals(self.zs[m], d).values, self.user_vals(self.zs[m], d, real=True).values
                val[(m, d, "o")], val[(m, d, "r")] = float(o.mean()), float(r.mean())
            df = self.zs[m]
            val[(m, None, "eta")] = eta(df["ndcg"].values, df["r_u"].values)
        cols = [(d, "o") for d in scopes] + [(d, "r") for d in scopes] + [(None, "eta")]
        mark = {}
        for c in cols:
            ranked = sorted(rankers, key=lambda m: -val[(m, *c)])
            if ranked:
                mark[(ranked[0], *c)] = "b"
            if len(ranked) > 1:
                mark[(ranked[1], *c)] = "u"

        def cell(m, c):
            x = val[(m, *c)]
            s = f"{100 * x:.2f}" if c[1] == "r" else _fmt(x).replace("0.", ".", 1)
            k = mark.get((m, *c))
            return r"\textbf{" + s + "}" if k == "b" else r"\underline{" + s + "}" if k == "u" else s

        def name(m):
            if m in sub:
                return r"\quad " + _tex_escape(sub[m][1])
            return _tex_escape(short.get(m, self.meta(m).get("display", m)))

        heading = {"decision": "Decision models", "llm": "LLM rankers", "plm": "PLM rankers", "reference": "References"}
        ncol = len(cols)
        n_ret = {d: int(self.cands.loc[self.cands["dataset"] == d, "r_u"].sum()) for d in self.datasets}
        n_ret_txt = ", ".join(f"{self.dnames[d]} {n_ret[d]}" for d in self.datasets)
        runs = self._runs_text(rows)
        dom_hdr = " & ".join(dabbr.get(d, self.dnames[d]) for d in self.datasets) + (" & All" if one_col else " & Pooled")
        env = "table" if one_col else "table*"
        if one_col:
            # short caption without colons or semicolons, the details are in the text and the repository
            caption = (rf"NDCG@{self.k} for {self._n_users_plain()} test users per domain, averaged over "
                       rf"{self._n_runs_word(rows)} runs. {SETTING['realistic']} scores and BPR's Recall@{n_c} are "
                       rf"$\times 100$. $\eta$ is the pooled NDCG@{self.k} over users whose target BPR retrieved. "
                       r"All is pooled over domains. Best ranker in bold, second underlined.")
            size = [r"\footnotesize", r"\setlength{\tabcolsep}{1.4pt}"]
            spec = r"@{}l" + "c" * (nd + 1) + r"@{\hspace{5pt}}" + "c" * (nd + 1) + r"@{\hspace{5pt}}c@{}"
            real_hdr = rf"Realistic ($\times$100)"
        else:
            caption = None
            size = [r"\small"]
            spec = r"@{}l" + "c" * (nd + 1) + "c" * (nd + 1) + "c@{}"
            real_hdr = rf"{SETTING['realistic']} ($\times$100)"
        lines = [r"% Table 1 (RQ1) -- requires \usepackage{booktabs}; written by make rq1-report (Report.table1)",
                 rf"\begin{{{env}}}[t]", r"\centering",
                 r"\caption{" + caption + "}" if one_col else r"\caption{" +
                 rf"NDCG@{self.k} ({self._n_users_text()}; {runs[0].lower() + runs[1:-1]}). "
                 rf"{SETTING['injected']}: the target is always among the {n_c} candidates. "
                 rf"{SETTING['realistic']} ($\times 100$): BPR's top {n_c} without injection, scored 0 when the target is "
                 rf"not retrieved; the last row gives BPR's Recall@{n_c} ($\times 100$), the upper bound for any ranker "
                 rf"(target retrieved for {n_ret_txt} users). $\eta$: ceiling utilisation, mean NDCG@{self.k} over users "
                 rf"whose target BPR retrieved, pooled. Bold: best; underlined: second best among the rankers.}}",
                 r"\label{tab:rq1}", *size,
                 r"\begin{tabular}{" + spec + "}", r"\toprule",
                 rf" & \multicolumn{{{nd + 1}}}{{c}}{{{SETTING['injected']}}} & "
                 rf"\multicolumn{{{nd + 1}}}{{c}}{{{real_hdr}}} & \\",
                 rf"\cmidrule(lr){{2-{nd + 2}}}\cmidrule(lr){{{nd + 3}-{2 * nd + 3}}}",
                 rf"Model & {dom_hdr} & {dom_hdr} & $\eta$ \\", r"\midrule"]
        prev, parent = None, None
        for m in rows:
            g = self.meta(m)["group"]
            if g != prev:
                if prev is not None:
                    lines.append(r"\midrule")
                lines.append(rf"\multicolumn{{{1 + ncol}}}{{@{{}}l}}{{\textit{{{heading.get(g, g)}}}}} \\")
                parent = None
            if m in sub and sub[m][0] != parent:
                parent = sub[m][0]
                lines.append(_tex_escape(parent) + " &" * ncol + r" \\")
            lines.append(name(m) + " & " + " & ".join(cell(m, c) for c in cols) + r" \\")
            prev = g
        lines += [r"\midrule",
                  rf"\textit{{BPR Recall@{n_c}}}" + " &" * (nd + 2) + " "
                  + " & ".join(f"{100 * rec[d]:.1f}" for d in self.datasets) + f" & {100 * rec['pooled']:.1f} & " + r" \\",
                  r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
        (self.out / "table1.tex").write_text("\n".join(lines) + "\n")

    def build(self) -> dict:
        self.table1()
        t1, t2, f1 = self.best_vs_second(), self.pooled(), self.figure1()
        ct = self.class_tests()
        comp = self.compliance()
        tn = {
            "mode": self.cfg["mode"],
            "n_test_users": {d: int((self.cands["dataset"] == d).sum()) for d in self.datasets},
            "dropped_untitled_before_kcore": {d: DatasetStore.load(self.cfg, d).info.get("dropped_untitled")
                                              for d in self.datasets},
            "missing_models": self.missing,
            "crossover_pooled": {"injected": f1["injected"]["pooled"], "realistic": f1["realistic"]["pooled"]},
            "crossover_per_domain": {"injected": f1["injected"]["per_domain"], "realistic": f1["realistic"]["per_domain"]},
            "bpr_recall_at_20": t2["bpr_recall"],
            "llm_compliance_rate": comp,
            "llm_compliance_below_100": {m: v for m, v in comp.items() if v < 1.0},
            "pooled_ndcg_injected": self.pooled_injected(),
            "injected_per_domain": {"means": t1["means"], "tests": t1["tests"], "ci_width_max": t1["ci_width_max"],
                       "ci_width_median": t1["ci_width_median"]},
            "pooled": {"models": t2["pooled"], "test_best_vs_bpr": t2["test_best_vs_bpr"]},
            "class_tests": ct,
            "tie_aware_ndcg": self.tie_robustness(),
            "run_to_run_stability": self.stability(),
            "api_refusals_and_checks": self.api_health(),
        }
        write_json(self.out / "text_numbers.json", tn)
        log.info("Report written to %s", self.out)
        return tn


def build_report(cfg: dict) -> dict:
    return Report(cfg).build()
