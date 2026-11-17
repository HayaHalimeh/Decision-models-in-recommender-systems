"""Shared look of the paper figures (Figure 1 for RQ1, Figures 3 and 4 for RQ2/RQ3).

Every figure is drawn at its printed size (full text width), with the same serif font, boxed panels, a dashed
grid and the model names of Table 1.
Which models Figures 3 and 4 draw is a rule (report.figure_models), applied to Table 1's pooled scores.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from ..config import model_cfg

PAGE_W = 7.0   # in, full text width (figure*)
COL_W = 3.33   # in, one column
RC = {"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
      "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": True, "axes.spines.right": True,
      "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
      "mathtext.fontset": "stix", "grid.linestyle": "--", "grid.color": "#b0b0b0",
      "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
      "lines.markersize": 3, "legend.frameon": False, "legend.handlelength": 2.0, "legend.columnspacing": 1.2,
      "legend.handletextpad": 0.4, "pdf.fonttype": 42}


def name(cfg: dict, m: str) -> str:
    """Model name as in Table 1 (its short name where one is set)."""
    return cfg["report"].get("per_domain_short_display", {}).get(m, model_cfg(cfg, m).get("display", m))


def paper_models(cfg: dict, pooled: dict[str, float]) -> list[str]:
    """The models the figures draw, from Table 1's pooled Injection Oracle NDCG@k (`pooled`: model -> score):
    the report.figure_models.n_decision best decision models, the best LLM reranker, then `extra`."""
    rule = cfg["report"].get("figure_models", {})
    by_group = lambda g: sorted((m for m in pooled if model_cfg(cfg, m)["group"] == g), key=lambda m: -pooled[m])
    out = by_group("decision")[: int(rule.get("n_decision", 2))]
    if rule.get("best_llm", True) and by_group("llm"):
        out.append(by_group("llm")[0])
    out += [m for m in rule.get("extra", []) if m in pooled and m not in out]
    return out


def legend_row(fig, handles, labels, ncol: int | None = None, y: float = 1.0) -> None:
    """The figure's one legend: a row above the panels."""
    fig.legend(handles, labels, loc="lower center", ncol=ncol or len(labels), bbox_to_anchor=(0.5, y))


def save(fig, out_dir, stem: str) -> None:
    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
