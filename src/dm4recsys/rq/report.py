"""Figures 3 (RQ2) and 4 (RQ3), repository CSVs, captions and text numbers."""
from __future__ import annotations



import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ..config import model_cfg, path  # noqa: E402
from ..data.store import DatasetStore  # noqa: E402
from ..eval import collect as rq1_collect  # noqa: E402
from ..eval.stats import bootstrap_ci, holm, paired_t, paired_wilcoxon  # noqa: E402
from ..retrieval.candidates import load_candidates  # noqa: E402
from ..report import figstyle  # noqa: E402
from ..utils import read_json, write_json  # noqa: E402
from .collect import RQData  # noqa: E402
from .conditions import CONTENT, LABELS, ORDER, REAL, RERUN, RQ2_HISTORY, all_conditions, rotation_name  # noqa: E402
from .metrics import KEY, change, friedman, label_outcome, mcnemar, paired, popularity_percentiles  # noqa: E402
from .runner import longhist_lengths  # noqa: E402

POOLED = "pooled"


class RQReport:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.datasets = list(cfg["run_datasets"])
        self.k = cfg["eval"]["k"]
        self.B, self.bseed, self.alpha = cfg["eval"]["bootstrap_resamples"], cfg["eval"]["bootstrap_seed"], cfg["eval"]["alpha"]
        self.out = path(cfg, "results")
        self.out.mkdir(parents=True, exist_ok=True)
        self.csv = self.out / "rq_csv"
        self.csv.mkdir(exist_ok=True)
        self.data = RQData(cfg, self.datasets)
        self.colors = cfg["rq"]["colors"]
        self.dnames = {d: cfg["datasets"][d]["display"] for d in self.datasets}
        order = list(cfg["report"]["table1_rows"])  # same model order as Table 1
        self.models = sorted([m for m in cfg["rq"]["models"] if self._has(m, REAL)],
                             key=lambda m: order.index(m) if m in order else len(order))
        self.missing_models = [m for m in cfg["rq"]["models"] if m not in self.models]
        self.table1_pooled = self.table1_scores()
        # Figures 3 and 4 (report.figure_models), in Table 1 order: decision models first, then the LLM rerankers
        self.fig_models = sorted(figstyle.paper_models(cfg, self.table1_pooled), key=self.models.index)

    def table1_scores(self) -> dict[str, float]:
        """Table 1's pooled Injection Oracle NDCG@k (RQ1 runs, mean over runs per user, all test users) of the
        models with RQ2/RQ3 results; picks the figures' models."""
        cands = rq1_collect.candidate_frame(self.cfg, self.datasets)
        out = {}
        for m in self.models:
            df = rq1_collect.zero_shot(self.cfg, m, self.datasets, cands)
            if df is not None:
                out[m] = float(df["ndcg"].mean())
        return out

    def figure_models_text(self) -> str:
        dec = [m for m in self.fig_models if model_cfg(self.cfg, m)["group"] == "decision"]
        llm = [m for m in self.fig_models if model_cfg(self.cfg, m)["group"] == "llm"]
        n_dec_all = sum(model_cfg(self.cfg, m)["group"] == "decision" for m in self.models)
        words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
        parts = []
        if dec:
            parts.append(f"the {words.get(len(dec), len(dec))} decision models" if len(dec) == n_dec_all
                         else ", ".join(self.short(m) for m in dec))
        parts += [self.short(m) for m in llm]
        text = ", ".join(parts[:-1]) + ", and " + parts[-1] if len(parts) > 2 else " and ".join(parts)
        return text + ". Results for all rankers are in the CSV files"

    def _has(self, m, c) -> bool:
        try:
            return len(self.data.get(m, c)) > 0
        except FileNotFoundError:
            return False

    def short(self, m) -> str:
        return figstyle.name(self.cfg, m)

    def scopes(self):
        return self.datasets + [POOLED]

    def sel(self, df: pd.DataFrame, scope: str) -> pd.DataFrame:
        return df if scope == POOLED else df[df["dataset"] == scope]

    def avail(self, m, c) -> pd.DataFrame | None:
        if not self._has(m, c):
            return None
        self.data.complete(m, c)
        return self.data.get(m, c)

    def caption(self, name: str, text: str):
        (self.out / f"{name}_caption.txt").write_text(text + "\n")

    def ci(self, x):
        return bootstrap_ci(np.asarray(x, float), self.B, self.bseed, self.alpha)

    def noise(self, m) -> dict:
        """Rerun vs Real, per scope."""
        rr = self.avail(m, RERUN)
        real = self.data.get(m, REAL)
        return {s: (change(self.sel(rr, s), self.sel(real, s), self.k) if rr is not None else None) for s in self.scopes()}

    def rq2(self) -> dict:
        conds = [REAL] + RQ2_HISTORY
        rows, tests = [], []
        for m in self.models:
            real = self.data.get(m, REAL)
            nz = self.noise(m)
            for c in RQ2_HISTORY + [RERUN]:
                df = self.avail(m, c)
                if df is None:
                    continue
                for s in self.scopes():
                    ch = change(self.sel(df, s), self.sel(real, s), self.k)
                    lo, hi = self.ci(self.sel(df, s)["ndcg"])
                    rows.append({"scope": s, "model": m, "condition": c, **ch, "ndcg_ci_lo": lo, "ndcg_ci_hi": hi,
                                 "noise_top1_change": (nz[s] or {}).get("top1_change"),
                                 "noise_mean_abs_dndcg": (nz[s] or {}).get("mean_abs_dndcg"),
                                 "noise_hit_flip": (nz[s] or {}).get("hit_flip")})
                if c != RERUN:
                    a, b = paired(df, real)
                    tests.append({"model": m, "condition": c, **paired_t(a, b)})
            for s in self.scopes():
                lo, hi = self.ci(self.sel(real, s)["ndcg"])
                rows.append({"scope": s, "model": m, "condition": REAL, "n": int(len(self.sel(real, s))),
                             "ndcg": float(self.sel(real, s)["ndcg"].mean()), "ndcg_ci_lo": lo, "ndcg_ci_hi": hi})
        if tests:
            for t, p in zip(tests, holm([t["p"] for t in tests])):  # Holm over all tests in panel (a)
                t["p_holm"] = p
                t["significant"] = bool(p < self.alpha)
        table = pd.DataFrame(rows)
        table.to_csv(self.csv / "rq2_change_table.csv", index=False)
        pd.DataFrame(tests).to_csv(self.csv / "rq2_tests_conditions.csv", index=False)

        # relative NDCG drop under no history, in %
        drop = []
        for m in self.models:
            for s in self.scopes():
                r = table[(table.model == m) & (table.scope == s) & (table.condition == REAL)]
                n = table[(table.model == m) & (table.scope == s) & (table.condition == "no_history")]
                if len(r) and len(n):
                    rv, nv = float(r.ndcg.iloc[0]), float(n.ndcg.iloc[0])
                    drop.append({"scope": s, "model": m, "ndcg_real": rv, "ndcg_no_history": nv,
                                 "relative_drop_pct": 100.0 * (rv - nv) / rv if rv else np.nan})
        pd.DataFrame(drop).to_csv(self.csv / "rq2_no_history_drop.csv", index=False)

        pairing = self.pairing_stats()
        hl = self.history_length()
        self.figure3(table, hl)
        self.caption("fig3", self._fig3_caption(table, hl))
        tn = {"conditions": conds, "change_table_pooled": table[table.scope == POOLED].to_dict("records"),
              "tests_conditions": tests, "no_history_drop_pct": drop, "other_user_pairing": pairing,
              "history_length": hl["summary"], "missing_models": self.missing_models,
              "shuffle_identity_users": self.shuffle_identity(), **self.figure_numbers()}
        write_json(self.out / "text_numbers_rq2.json", tn)
        return tn

    def pairing_stats(self) -> dict:
        out = {}
        for m in self.models[:1]:  # the pairing is model-independent
            df = self.avail(m, "other_user")
            if df is None:
                return {}
            lg = pd.DataFrame([dict(x, dataset=d) for x, d in zip(df["transform_log"], df["dataset"])])
            for s in self.scopes():
                x = lg if s == POOLED else lg[lg["dataset"] == s]
                out[s] = {"users": int(len(x)), "mean_redraws": float(x["redraws"].mean()),
                          "share_redrawn": float((x["redraws"] > 0).mean()),
                          "mean_overlap_candidates": float(x["overlap_candidates"].mean()),
                          "share_overlap_candidates": float((x["overlap_candidates"] > 0).mean()),
                          "mean_overlap_own_history": float(x["overlap_own_history"].mean())}
        pd.DataFrame([{"scope": s, **v} for s, v in out.items()]).to_csv(self.csv / "rq2_other_user_pairing.csv", index=False)
        return out

    def shuffle_identity(self) -> dict:
        if not self.models or self.avail(self.models[0], "shuffled") is None:
            return {}
        df = self.data.get(self.models[0], "shuffled")
        lg = df["transform_log"]
        return {"len1_identity": int(sum(bool(x.get("shuffle_identity_len1")) for x in lg)),
                "identity_by_chance": int(sum(bool(x.get("shuffle_identity")) for x in lg)), "users": int(len(df))}

    def history_length(self) -> dict:
        lens = {d: longhist_lengths(self.cfg, d) for d in self.datasets}
        metas = {d: load_candidates(self.cfg, d, "longhist")[1]["longhist"] for d in self.datasets}
        pooled_lengths = sorted(set.intersection(*[set(v) for v in lens.values()])) if lens else []
        rows, tests = [], []
        for m in self.models:
            per = {}
            for d in self.datasets:
                for L in lens[d]:
                    df = self.avail(m, f"length_{L:02d}")
                    if df is None:
                        continue
                    x = df[df["dataset"] == d]
                    if len(x) == 0:
                        continue
                    per[(d, L)] = x
                    lo, hi = self.ci(x["ndcg"])
                    rows.append({"scope": d, "model": m, "length": L, "n": len(x), "ndcg": float(x["ndcg"].mean()),
                                 "ndcg_ci_lo": lo, "ndcg_ci_hi": hi})
            for L in pooled_lengths:
                parts = [per[(d, L)] for d in self.datasets if (d, L) in per]
                if len(parts) == len(self.datasets):
                    x = pd.concat(parts)
                    lo, hi = self.ci(x["ndcg"])
                    rows.append({"scope": POOLED, "model": m, "length": L, "n": len(x), "ndcg": float(x["ndcg"].mean()),
                                 "ndcg_ci_lo": lo, "ndcg_ci_hi": hi})
            if len(pooled_lengths) >= 2 and all((d, L) in per for d in self.datasets for L in (pooled_lengths[0], pooled_lengths[-1])):
                a = pd.concat([per[(d, pooled_lengths[0])] for d in self.datasets])
                b = pd.concat([per[(d, pooled_lengths[-1])] for d in self.datasets])
                x, y = paired(a, b)
                tests.append({"model": m, "short": pooled_lengths[0], "long": pooled_lengths[-1], **paired_t(x, y)})
        if tests:
            for t, p in zip(tests, holm([t["p"] for t in tests])):
                t["p_holm"] = p
                t["significant"] = bool(p < self.alpha)
        curves = pd.DataFrame(rows)
        curves.to_csv(self.csv / "rq2_history_length_curves.csv", index=False)
        for d in self.datasets:
            curves[curves.get("scope", pd.Series(dtype=str)) == d].to_csv(self.csv / f"rq2_history_length_{d}.csv", index=False)
        pd.DataFrame(tests).to_csv(self.csv / "rq2_tests_history_length.csv", index=False)
        summary = {"caps": {d: metas[d]["cap"] for d in self.datasets},
                   "lengths_per_dataset": lens, "lengths_dropped": {d: metas[d]["lengths_dropped"] for d in self.datasets},
                   "pooled_lengths": pooled_lengths, "tests": tests}
        write_json(self.csv / "rq2_history_length_caps.json", summary)
        return {"curves": curves, "summary": summary}

    # Figures 3 and 4: one finding per panel, models on the x axis (decision models, a gap, then the LLM rerankers)
    FIG_SIZE = (figstyle.PAGE_W, 1.7)
    FIG_RC = {**figstyle.RC, "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "legend.fontsize": 6,
              "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "xtick.major.size": 2, "ytick.major.size": 2,
              "axes.titlepad": 4, "legend.handlelength": 1.2, "legend.columnspacing": 0.9, "legend.borderaxespad": 0.2}
    DARK, MID, LIGHT, GREY, RED = "#1f3b73", "#4f86c6", "#b9d3ee", "#9ca3af", "#dc2626"

    def _xpos(self) -> dict:
        """x position per figure model, with a gap between the decision models and the LLM rerankers."""
        return {m: i + (0.8 if model_cfg(self.cfg, m)["group"] != "decision" else 0) for i, m in enumerate(self.fig_models)}

    def _model_axis(self, ax, ylabel: str) -> None:
        X, ms = self._xpos(), self.fig_models
        ax.set_xticks([X[m] for m in ms], [self.short(m) for m in ms], rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, X[ms[-1]] + 0.6)
        dec = [m for m in ms if model_cfg(self.cfg, m)["group"] == "decision"]
        if dec and len(dec) < len(ms):
            ax.axvline((X[dec[-1]] + X[ms[len(dec)]]) / 2, color="#d1d5db", lw=0.4)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", lw=0.4)

    def _save_fig(self, fig, name: str) -> None:
        fig.tight_layout(pad=0.3, w_pad=1.2)
        for ext in ("pdf", "png"):
            fig.savefig(self.out / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def figure3(self, table: pd.DataFrame, hl: dict):
        """RQ2 at printed size (7 x 1.7 in), models on the x axis in every panel:
        (a) NDCG@k with the own history, another user's history, random items and no history, against a random
        ranking; (b) share of users whose top-1 changes when the history is shuffled or reversed, against a plain
        rerun of the same input; (c) NDCG@k over history length, one small line per model (long-history sample)."""
        pooled = table[table.scope == POOLED]
        ms, X = self.fig_models, self._xpos()
        xs = np.array([X[m] for m in ms])
        val = lambda m, c, col: float(pooled[(pooled.model == m) & (pooled.condition == c)][col].iloc[0])  # noqa: E731
        rand = self.table1_random()
        with plt.rc_context(self.FIG_RC):
            fig, axes = plt.subplots(1, 3, figsize=self.FIG_SIZE, gridspec_kw={"width_ratios": [1.35, 1.15, 1.0]})
            # (a) content of the history
            ax = axes[0]
            bars = ((REAL, "own", self.DARK), ("other_user", "other user's", self.MID),
                    ("random_fake", "random items", "#94a3b8"), ("no_history", "none", self.LIGHT))
            w = 0.8 / len(bars)
            for j, (c, lab, col) in enumerate(bars):
                ys = np.array([val(m, c, "ndcg") for m in ms])
                lo = np.array([val(m, c, "ndcg_ci_lo") for m in ms])
                hi = np.array([val(m, c, "ndcg_ci_hi") for m in ms])
                ax.bar(xs - 0.4 + w * (j + 0.5), ys, w, color=col, label=lab, lw=0, yerr=np.vstack([ys - lo, hi - ys]),
                       error_kw={"lw": 0.4, "capsize": 0, "ecolor": "#374151"})
            if rand is not None:
                ax.axhline(rand, color=self.RED, ls="--", lw=0.5)
                dec = [m for m in ms if model_cfg(self.cfg, m)["group"] == "decision"]
                gap = (X[dec[-1]] + X[ms[len(dec)]]) / 2 if 0 < len(dec) < len(ms) else X[ms[-1]]
                ax.text(gap, rand, "random\nranking", color=self.RED, fontsize=5.2, ha="center", va="bottom",
                        linespacing=0.9, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3})
            self._model_axis(ax, f"NDCG@{self.k}")
            ax.set_title("(a) Accuracy by history content", pad=27)
            ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), title="history:", title_fontsize=6,
                      handlelength=1.0, columnspacing=0.7)
            ax.get_legend()._legend_box.align = "left"
            # (b) order of the history vs. run-to-run noise
            ax = axes[1]
            bars = (("shuffled", "shuffled", "#0f766e"), ("reversed", "reversed", "#5eead4"),
                    (RERUN, "same input again", self.GREY))
            w = 0.8 / len(bars)
            for j, (c, lab, col) in enumerate(bars):
                ax.bar(xs - 0.4 + w * (j + 0.5), [val(m, c, "top1_change") for m in ms], w, color=col, lw=0, label=lab)
            self._model_axis(ax, "top-ranked item changed")
            ax.set_ylim(0, 1)
            ax.set_title("(b) Effect of history order", pad=27)
            ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
            # (c) history length: per model a small line over all lengths (left to right: 5 ... 40 recent items)
            ax = axes[2]
            lens = hl["summary"]["pooled_lengths"]
            cur = hl["curves"]
            cp = cur[cur.scope == POOLED].set_index(["model", "length"])["ndcg"] if len(cur) else None
            if cp is not None and lens:
                shades = plt.get_cmap("Blues")(np.linspace(0.3, 1.0, len(lens)))
                dx = np.linspace(-0.32, 0.32, len(lens))
                for m in ms:
                    y = np.array([cp.get((m, L), np.nan) for L in lens])
                    ax.plot(X[m] + dx, y, color=self.GREY, lw=0.4, zorder=1)
                    ax.scatter(X[m] + dx, y, s=7, c=shades, edgecolors="none", zorder=2)
                for L, col in zip(lens, shades):
                    ax.scatter([], [], s=9, color=col, label=str(L))
            self._model_axis(ax, f"NDCG@{self.k}")
            ax.set_title("(c) Effect of history length", pad=27)
            ax.legend(ncol=len(lens), loc="lower center", bbox_to_anchor=(0.5, 1.0), title="most recent items:",
                      title_fontsize=6, handlelength=0.8, handletextpad=0.2, columnspacing=0.6)
            ax.get_legend()._legend_box.align = "left"
            self._save_fig(fig, "fig3")

    def table1_random(self) -> float | None:
        """Random's pooled Injection Oracle NDCG@k from the RQ1 text numbers (None before the RQ1 report)."""
        try:
            return float(read_json(self.out / "text_numbers.json")["pooled"]["models"]["random"]["injected"])
        except (FileNotFoundError, KeyError, TypeError):
            return None

    def figure_numbers(self) -> dict:
        rule = self.cfg["report"].get("figure_models", {})
        return {"figure_models": self.fig_models, "figure_models_rule": rule,
                "figure_models_table1_pooled_ndcg": {m: self.table1_pooled[m] for m in self.fig_models}}

    def _fig3_caption(self, table, hl) -> str:
        n = sum(len(self.data.recs[d]) for d in self.datasets)
        s = hl["summary"]
        return (f"RQ2: {self.figure_models_text()}. Pooled over {n} test users with Injection Oracle candidates. "
                f"(a) NDCG@{self.k} with the user's own history, another test user's history, random catalogue items "
                f"and no history (95% bootstrap CIs); dashed: a random ranking. (b) Share of users whose top-ranked "
                f"item changes when the history is shuffled or reversed, and when the same input is simply sent again "
                f"(the model's own run-to-run noise). (c) NDCG@{self.k} with the most recent "
                f"{', '.join(map(str, s['pooled_lengths']))} history items (left to right within each model), on the "
                f"{self.cfg['longhist']['n_users']} users per domain with the longest histories.")

    def rq3(self) -> dict:
        tn = {"missing_models": self.missing_models, **self.figure_numbers()}
        tn["position"] = self.position()
        tn["popularity"] = self.popularity()
        tn["label"] = self.label()
        tn["class_tests"] = self.class_tests_rq3()
        self.figure4(tn)
        self.caption("fig4", (f"RQ3: {self.figure_models_text()}. Pooled over all test users with Injection Oracle "
                              f"candidates. (a) NDCG@{self.k} with the target listed first, the mean over the other "
                              f"positions, and the range over all {self.cfg['candidates']['n']} positions (cyclic "
                              f"rotations of the candidate list; identifiers move with their items). (b) Top-ranked item "
                              f"after the identifiers are reassigned: the same item, the item that now carries the old "
                              f"top item's identifier, or another item; black lines: share keeping the same top item on a plain "
                              f"rerun of the unchanged input. (c) Popularity percentile of the top-ranked item with the "
                              f"user's own history and with no history, against the mean over all candidates and over "
                              f"the true targets."))
        write_json(self.out / "text_numbers_rq3.json", tn)
        return tn

    def position(self) -> dict:
        n = int(self.cfg["rq"]["n_rotations"])
        rows, rot_rows, tests = [], [], []
        for m in self.models:
            real = self.data.get(m, REAL)
            frames = [real.assign(rotation=0)]
            for k in range(1, n):
                df = self.avail(m, rotation_name(k))
                if df is None:
                    continue
                frames.append(df.assign(rotation=k))
                for s in self.scopes():
                    rot_rows.append({"scope": s, "model": m, "rotation": k,
                                     **change(self.sel(df, s), self.sel(real, s), self.k)})
            allr = pd.concat(frames, ignore_index=True)
            for s in self.scopes():
                x = self.sel(allr, s)
                g = x.groupby("target_position")["ndcg"]
                for pos, v in g:
                    lo, hi = self.ci(v)
                    rows.append({"scope": s, "model": m, "target_position": int(pos), "n": len(v), "ndcg": float(v.mean()),
                                 "ndcg_ci_lo": lo, "ndcg_ci_hi": hi})
            piv = allr.pivot_table(index=KEY, columns="target_position", values="ndcg", aggfunc="first")
            piv = piv.dropna()  # users with all positions observed
            tests.append({"model": m, **friedman(piv.to_numpy()), "positions": int(piv.shape[1])})
        if tests:
            for t, p in zip(tests, holm([t["p"] if np.isfinite(t["p"]) else 1.0 for t in tests])):
                t["p_holm"] = p
                t["significant"] = bool(p < self.alpha)
        pos = pd.DataFrame(rows)
        pos.to_csv(self.csv / "rq3_position_ndcg.csv", index=False)
        rot = pd.DataFrame(rot_rows)
        rot.to_csv(self.csv / "rq3_rotation_change_per_rotation.csv", index=False)
        summ = (rot.groupby(["scope", "model"])[["top1_change", "mean_abs_dndcg", "hit_flip"]].mean().reset_index()
                if len(rot) else pd.DataFrame())
        summ.to_csv(self.csv / "rq3_rotation_change_summary.csv", index=False)
        pd.DataFrame(tests).to_csv(self.csv / "rq3_tests_position.csv", index=False)
        self._pos = pos
        return {"tests_friedman": tests, "rotation_change_mean_over_rotations": summ.to_dict("records")}

    def popularity(self) -> dict:
        pct = {}
        for d in self.datasets:
            st = DatasetStore.load(self.cfg, d)
            pct[d] = popularity_percentiles(st.popularity, st.all_items)
        nc = self.cfg["candidates"]["n"]
        rows, tests, text = [], [], {}
        for m in self.models:
            real = self.data.get(m, REAL)
            # tie-aware: rank r gets the mean popularity of the tie block that covers it,
            # i.e. the expected value under a random order of tied options (rq/collect._blocks)
            P = np.array([[float(np.mean([pct[d][i] for i in b])) for b in blocks for _ in b]
                          for d, blocks in zip(real["dataset"], real["blocks"])])
            tp = np.array([pct[d][t] for d, t in zip(real["dataset"], real["target"])])
            for s in self.scopes():
                mask = np.ones(len(real), bool) if s == POOLED else (real["dataset"] == s).to_numpy()
                for r in range(nc):
                    rows.append({"scope": s, "model": m, "rank": r + 1, "mean_popularity": float(P[mask, r].mean()),
                                 "target_mean_popularity": float(tp[mask].mean()), "n": int(mask.sum())})
            tests.append({"model": m, "top1_popularity": float(P[:, 0].mean()), "target_popularity": float(tp.mean()),
                          **paired_t(P[:, 0], tp)})
            nh = self.avail(m, "no_history")
            if nh is not None:
                nh = nh[nh["top1"].notna()]  # refused users have no top-1 (counted in the refusal share instead)
                nhp = nh.assign(pop=[pct[d][i] for d, i in zip(nh["dataset"], nh["top1"])])
                rp = real.assign(pop=P[:, 0])
                a, b = paired(nhp, rp, "pop")
                text[m] = {"top1_popularity_no_history": float(a.mean()), "top1_popularity_real": float(b.mean()),
                           **{f"t_{k}": v for k, v in paired_t(a, b).items()}}
            else:
                text[m] = None
        if tests:
            for t, p in zip(tests, holm([t["p"] for t in tests])):
                t["p_holm"] = p
                t["significant"] = bool(p < self.alpha)
        pop = pd.DataFrame(rows)
        pop.to_csv(self.csv / "rq3_popularity_by_rank.csv", index=False)
        pd.DataFrame(tests).to_csv(self.csv / "rq3_tests_popularity.csv", index=False)
        self._pop = pop
        return {"tests_top1_vs_target": tests, "no_history_vs_real_top1": text}

    def label(self) -> dict:
        rows, tests, pooled = [], [], {}
        for m in self.models:
            sw = self.avail(m, "label_swap")
            if sw is None:
                continue
            real = self.data.get(m, REAL)
            x = sw[KEY + ["top1", "options"]].merge(real[KEY + ["top1", "top1_id"]], on=KEY, suffixes=("", "_real"))
            x["outcome"] = [label_outcome(a, b, c, o) for a, b, c, o in zip(x["top1_real"], x["top1_id"], x["top1"], x["options"])]
            rr = self.avail(m, RERUN)
            if rr is not None:
                x = x.merge(rr[KEY + ["top1"]].rename(columns={"top1": "top1_rerun"}), on=KEY)
                x["changed_swap"] = x["top1"] != x["top1_real"]
                x["changed_rerun"] = x["top1_rerun"] != x["top1_real"]
                tests.append({"model": m, **mcnemar(x["changed_swap"], x["changed_rerun"])})
            for s in self.scopes():
                y = self.sel(x, s)
                shares = y["outcome"].value_counts(normalize=True).reindex(["same_item", "same_label", "other"]).fillna(0)
                row = {"scope": s, "model": m, "n": len(y), **shares.to_dict(),
                       "rerun_top1_change": float(y["changed_rerun"].mean()) if "changed_rerun" in y else None}
                rows.append(row)
                if s == POOLED:
                    pooled[m] = row
        if tests:
            for t, p in zip(tests, holm([t["p"] for t in tests])):
                t["p_holm"] = p
                t["significant"] = bool(p < self.alpha)
        pd.DataFrame(rows).to_csv(self.csv / "rq3_label_swap.csv", index=False)
        pd.DataFrame(tests).to_csv(self.csv / "rq3_tests_label.csv", index=False)
        return {"pooled": pooled, "tests_mcnemar": tests}

    def refusals(self, rq: str = "all") -> list[dict]:
        """Refusal share per model x condition x scope : calls the decision API still refused
        after runs.refusal_retries reruns. They are scored NDCG@k = 0 with the target ranked last and no top-1, so a
        condition's NDCG and change measures mix refusals and rankings; this table shows how much is refusal.
        Written to rq_csv/rq_refusal_share.csv; Real (RQ1 run) and the rerun included."""
        names = [REAL, RERUN] + [c["name"] for c in all_conditions(self.cfg, rq) if c["name"] not in (REAL, RERUN)]
        rows = []
        for m in self.models:
            for c in names:
                if not self._has(m, c):
                    continue
                df = self.data.get(m, c)
                for s in self.scopes():
                    x = self.sel(df, s)
                    if not len(x):
                        continue
                    n_ref = int(x["refused"].astype(bool).sum())
                    rows.append({"scope": s, "model": m, "condition": c, "n": int(len(x)), "n_refused": n_ref,
                                 "refusal_share": n_ref / len(x)})
        pd.DataFrame(rows, columns=["scope", "model", "condition", "n", "n_refused", "refusal_share"]).to_csv(
            self.csv / "rq_refusal_share.csv", index=False)
        write_json(self.out / "text_numbers_refusals.json", [r for r in rows if r["n_refused"]])
        return rows

    # RQ3 class-level tests: per user, each measure is averaged over the models of a class; decision models vs. LLM
    # rerankers are then compared on the same users (pooled over domains).
    RQ3_CLASS_MEASURES = {
        "position_change_excess": "top-1 change under rotation (mean over the rotations) minus top-1 change on a rerun",
        "primacy_gap": "NDCG@k with the target listed first minus the mean NDCG@k at the other positions",
        "label_change_excess": "top-1 change after the label swap minus top-1 change on a rerun",
        "follows_label": "top-1 after the label swap is the item that now carries the old top-1's ID (0/1)",
        "popularity_vs_candidates": "popularity percentile of the top-1 minus the mean over the user's candidates",
        "popularity_no_history_shift": "popularity percentile of the top-1 with no history minus with the own history",
    }

    def _rq3_user_measures(self, m: str, pct: dict) -> pd.DataFrame:
        """One row per (dataset, user) with the RQ3_CLASS_MEASURES of model m (NaN where a condition is missing).
        A refused call has no top-1 and counts as changed, as in rq/metrics.change."""
        real = self.data.get(m, REAL).set_index(KEY)
        out = pd.DataFrame(index=real.index)
        changed = lambda a, b: ((a != b) | a.isna() | b.isna()).astype(float)  # noqa: E731
        rr = self.avail(m, RERUN)
        rerun_ch = changed(rr.set_index(KEY)["top1"].reindex(real.index), real["top1"]) if rr is not None else np.nan
        rot = [self.avail(m, rotation_name(k)) for k in range(1, int(self.cfg["rq"]["n_rotations"]))]
        rot = [r.set_index(KEY) for r in rot if r is not None]
        if rot:
            out["position_change_excess"] = np.mean([changed(r["top1"].reindex(real.index), real["top1"]) for r in rot],
                                                    axis=0) - rerun_ch
            allr = pd.concat([real[["ndcg", "target_position"]]] + [r[["ndcg", "target_position"]] for r in rot])
            piv = allr.reset_index().pivot_table(index=KEY, columns="target_position", values="ndcg", aggfunc="first")
            if 1 in piv.columns:
                out["primacy_gap"] = (piv[1] - piv.drop(columns=1).mean(axis=1)).reindex(real.index)
        sw = self.avail(m, "label_swap")
        if sw is not None:
            sw = sw.set_index(KEY).reindex(real.index)
            out["label_change_excess"] = changed(sw["top1"], real["top1"]) - rerun_ch
            out["follows_label"] = [float(label_outcome(a, b, c, o) == "same_label") if isinstance(o, dict) else np.nan
                                    for a, b, c, o in zip(real["top1"], real["top1_id"], sw["top1"], sw["options"])]
        pop = lambda d, i: pct[d].get(i, np.nan) if isinstance(i, str) else np.nan  # noqa: E731
        ds = real.index.get_level_values("dataset")
        top1_pop = np.array([pop(d, i) for d, i in zip(ds, real["top1"])])
        cand_pop = np.array([np.mean([pct[d][i] for i in o.values()]) for d, o in zip(ds, real["options"])])
        out["popularity_vs_candidates"] = top1_pop - cand_pop
        nh = self.avail(m, "no_history")
        if nh is not None:
            nh_top1 = nh.set_index(KEY)["top1"].reindex(real.index)
            out["popularity_no_history_shift"] = np.array([pop(d, i) for d, i in zip(ds, nh_top1)]) - top1_pop
        return out

    def class_tests_rq3(self) -> dict:
        """Decision models vs. LLM rerankers on each RQ3 measure (RQ3_CLASS_MEASURES): per user the mean over all
        models of a class, then a paired t-test over users (pooled over domains), Holm-corrected over the measures;
        with the 95 % bootstrap CI of the mean difference, the shares of users for whom each class is higher, and the
        Wilcoxon signed-rank p. The excess measures subtract each model's own rerun change, so deterministic and
        sampling models are compared beyond their noise. Written to rq_csv/rq3_class_tests.csv."""
        groups = {g: [m for m in self.models if model_cfg(self.cfg, m)["group"] == g] for g in ("decision", "llm")}
        if not all(groups.values()):
            return {}
        pct = {}
        for d in self.datasets:
            st = DatasetStore.load(self.cfg, d)
            pct[d] = popularity_percentiles(st.popularity, st.all_items)
        per_model = {m: self._rq3_user_measures(m, pct) for ms in groups.values() for m in ms}
        rows, model_rows = [], []
        for meas, desc in self.RQ3_CLASS_MEASURES.items():
            cls = {}
            for g, ms in groups.items():
                cols = [per_model[m][meas] for m in ms if meas in per_model[m]]
                cls[g] = pd.concat(cols, axis=1).mean(axis=1) if cols else None
                for m in ms:
                    if meas in per_model[m]:
                        model_rows.append({"measure": meas, "model": m, "group": g,
                                           "mean": float(per_model[m][meas].mean())})
            if cls["decision"] is None or cls["llm"] is None:
                continue
            a, b = cls["decision"].align(cls["llm"], join="inner")
            ok = a.notna() & b.notna()
            a, b = a[ok].to_numpy(float), b[ok].to_numpy(float)
            r = paired_t(a, b)
            lo, hi = bootstrap_ci(a - b, self.B, self.bseed, self.alpha)
            r.update(measure=meas, description=desc, mean_decision=float(a.mean()), mean_llm=float(b.mean()),
                     ci_low=lo, ci_high=hi, share_decision_higher=float((a > b).mean()),
                     share_llm_higher=float((a < b).mean()), p_wilcoxon=paired_wilcoxon(a, b),
                     n_decision_models=len(groups["decision"]), n_llm_models=len(groups["llm"]))
            rows.append(r)
        for r, p in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = p
            r["significant"] = bool(p < self.alpha)
        cols = ["measure", "mean_decision", "mean_llm", "mean_diff", "ci_low", "ci_high", "t", "n", "p", "p_holm",
                "significant", "share_decision_higher", "share_llm_higher", "p_wilcoxon", "n_decision_models",
                "n_llm_models", "description"]
        pd.DataFrame(rows)[cols].to_csv(self.csv / "rq3_class_tests.csv", index=False)
        pd.DataFrame(model_rows).to_csv(self.csv / "rq3_class_measures_per_model.csv", index=False)
        return {"tests": rows, "classes": groups, "per_model_means": model_rows}

    def figure4(self, tn):
        """RQ3 at printed size (7 x 1.7 in): (a) NDCG@k with the target listed first, against the mean over the other
        positions and the range over all positions; (b) top-1 after the identifiers are reassigned; (c) popularity
        percentile of the top-1, against all candidates and the true targets."""
        ms, X = self.fig_models, self._xpos()
        xs = np.array([X[m] for m in ms])
        pos = self._pos[self._pos.scope == POOLED]
        lab = tn["label"]["pooled"]
        pop = self._pop[self._pop.scope == POOLED]
        top1 = {t["model"]: t["top1_popularity"] for t in tn["popularity"]["tests_top1_vs_target"]}
        with plt.rc_context(self.FIG_RC):
            fig, axes = plt.subplots(1, 3, figsize=self.FIG_SIZE)
            ax = axes[0]
            for m in ms:
                p = pos[pos.model == m].set_index("target_position")["ndcg"]
                ax.vlines(X[m], p.min(), p.max(), color=self.GREY, lw=1.5, alpha=0.5, zorder=1)
                ax.plot(X[m], p.drop(1).mean(), "o", color=self.LIGHT, mec=self.DARK, mew=0.6, ms=3.8, zorder=2)
                ax.plot(X[m], p.loc[1], "o", color=self.DARK, ms=3.8, zorder=3)
            self._model_axis(ax, f"NDCG@{self.k}")
            ax.set_title("(a) Effect of target position", pad=27)
            ax.legend(handles=[Line2D([], [], marker="o", ls="", color=self.DARK, ms=3.8, label="listed first"),
                               Line2D([], [], marker="o", ls="", color=self.LIGHT, mec=self.DARK, mew=0.6, ms=3.8,
                                      label="other positions"),
                               Line2D([], [], color=self.GREY, lw=1.5, alpha=0.5, label="range")],
                      ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
            ax = axes[1]
            bottom = np.zeros(len(ms))
            for o, lb in (("same_item", "same item"), ("same_label", "old label"), ("other", "other")):
                v = np.array([lab[m][o] if m in lab else 0.0 for m in ms])
                ax.bar(xs, v, 0.6, bottom=bottom, color=self.colors[o], label=lb, lw=0)
                bottom += v
            keep = np.array([1 - lab[m]["rerun_top1_change"] if m in lab and lab[m].get("rerun_top1_change") is not None
                             else np.nan for m in ms])  # share keeping the same top item on a plain rerun
            ax.hlines(keep, xs - 0.36, xs + 0.36, color="black", lw=0.6, label="rerun")
            self._model_axis(ax, "share of users")
            ax.set_ylim(0, 1)
            ax.set_title("(b) Top-ranked item after relabelling", pad=27)
            h, l = ax.get_legend_handles_labels()
            order = [l.index(x) for x in ("same item", "old label", "other", "rerun")]
            ax.legend([h[i] for i in order], [l[i] for i in order], ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
            ax = axes[2]
            cand = float(pop["mean_popularity"].mean())  # every model ranks the same candidates
            target = float(pop["target_mean_popularity"].iloc[0])
            nh = tn["popularity"]["no_history_vs_real_top1"]
            ax.plot(xs, [top1[m] for m in ms], "o", color=self.DARK, ms=3.8, label="own history")
            ax.plot(xs, [(nh.get(m) or {}).get("top1_popularity_no_history", np.nan) for m in ms], "o", color="white",
                    mec=self.DARK, mew=0.8, ms=3.8, label="no history")
            ax.axhline(cand, color=self.GREY, lw=0.5, label=f"candidates ({cand:.2f})")
            ax.axhline(target, color=self.RED, ls="--", lw=0.5, label=f"targets ({target:.2f})")
            self._model_axis(ax, "popularity percentile")
            ax.set_ylim(np.floor(min(target, *top1.values()) * 20) / 20 - 0.03, 1.0)
            ax.set_title("(c) Popularity of the top-ranked item", pad=27)
            ax.set_ylabel("popularity pct.")
            ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
            self._save_fig(fig, "fig4")


def build_rq_report(cfg: dict, rq: str = "all") -> dict:
    r = RQReport(cfg)
    out = {}
    if rq in ("rq2", "all"):
        out["rq2"] = r.rq2()
    if rq in ("rq3", "all"):
        out["rq3"] = r.rq3()
    out["refusals"] = r.refusals("all")  # one file for RQ2 and RQ3, whichever report runs
    return out
