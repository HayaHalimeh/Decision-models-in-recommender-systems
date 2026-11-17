from __future__ import annotations

import math

import numpy as np


def ndcg_at_k(rank: int, k: int = 5) -> float:
    """One relevant item at 1-based `rank`: 1/log2(rank+1) if rank <= k else 0."""
    if rank < 1:
        raise ValueError("rank is 1-based")
    return 1.0 / math.log2(rank + 1) if rank <= k else 0.0


def tie_rank(scores: dict[str, float], target_id: str, first: str | None = None) -> tuple[int, int]:
    """(h, t) for the target: h options are ranked above it for certain, and it is equally likely at ranks
    h+1 .. h+t (t = options sharing its score, incl. itself). `first`: an option fixed at rank 1 regardless of its
    score (the decision API's own choice under the near-tie rule); the other options are ranked below it."""
    if first is not None:
        if target_id == first:
            return 0, 1
        h, t = tie_rank({o: v for o, v in scores.items() if o != first}, target_id)
        return h + 1, t
    s = scores[target_id]
    vals = list(scores.values())
    return sum(1 for v in vals if v > s), sum(1 for v in vals if v == s)


def tie_aware_ndcg(scores: dict[str, float], target_id: str, k: int = 5, first: str | None = None) -> float:
    """Tie-aware NDCG@k (McSherry & Najork 2008, ECIR) for one relevant item: the expected NDCG@k over all orders of
    the options that share the target's score, i.e. the mean of NDCG@k at ranks h+1 .. h+t (see tie_rank). Without
    ties this equals ndcg_at_k(h+1). The main metric: Jev and OpenAI Decisions
    round probabilities to 2 decimals, so the seeded tie-break would otherwise add noise of our own."""
    return expected_ndcg(*tie_rank(scores, target_id, first), k)


def expected_ndcg(h: int, t: int, k: int = 5) -> float:
    """Mean NDCG@k of one relevant item that is equally likely at ranks h+1 .. h+t."""
    return float(np.mean([ndcg_at_k(h + j, k) for j in range(1, t + 1)]))


def list_tie_rank(order: list[str], target_id: str, n_ranked: int | None) -> tuple[int, int]:
    """(h, t) for a ranked list without scores (LLMs): the model itself ranked the first `n_ranked` options; the
    options it did not name (appended by the parser in a seeded random order) form one tied block at the end.
    n_ranked None or >= len(order) - 1: no ties."""
    rank = order.index(target_id) + 1
    if n_ranked is None or rank <= n_ranked:
        return rank - 1, 1
    return n_ranked, len(order) - n_ranked


def list_tie_groups(order: list[str], n_ranked: int | None) -> tuple[frozenset, ...]:
    """tie_groups for a ranked list without scores: one block per ranked option, then the unnamed options as one block."""
    n = len(order) if n_ranked is None else n_ranked
    head = tuple(frozenset([o]) for o in order[:n])
    return head + ((frozenset(order[n:]),) if order[n:] else ())


def tie_groups(scores: dict[str, float], first: str | None = None) -> tuple[frozenset, ...]:
    """The ranking a score vector fixes, without our tie-break: blocks of equally scored options, best first
    (`first` as its own leading block). Used to compare runs (stability) independently of the tie-break."""
    rest = {o: v for o, v in scores.items() if o != first}
    blocks = [frozenset(o for o, v in rest.items() if v == s) for s in sorted(set(rest.values()), reverse=True)]
    return ((frozenset([first]),) if first is not None else ()) + tuple(blocks)


def realistic(ndcg: np.ndarray, r_u: np.ndarray) -> np.ndarray:
    """Per-user realistic NDCG: the injected value if BPR retrieved the target (r_u = 1), else 0."""
    return np.asarray(ndcg, float) * np.asarray(r_u, float)


def eta(ndcg: np.ndarray, r_u: np.ndarray) -> float:
    """Ceiling utilisation: mean NDCG over users with r_u = 1 (= realistic NDCG / recall@20)."""
    r = np.asarray(r_u, bool)
    return float(np.mean(np.asarray(ndcg, float)[r])) if r.any() else float("nan")


def recall(r_u: np.ndarray) -> float:
    return float(np.mean(r_u))


def with_cutoff(rows: list[dict], k: int) -> list[dict]:
    """Rows carry ndcg{k} for the run's cutoff (eval.k, 5). For another cutoff the same tie-aware value follows from
    the stored tie ranks (h, t) and the saved rank, so reports at @10 / @20 need no new model call:
    ndcg{k} = expected_ndcg(h, t, k), ndcg{k}_order = ndcg_at_k(rank, k); a refused row stays 0."""
    for r in rows:
        if f"ndcg{k}" in r or "tie" not in r:
            continue
        refused = r.get("refusal_scored_zero") or (r.get("info") or {}).get("api_refusal")
        r[f"ndcg{k}"] = 0.0 if refused else expected_ndcg(int(r["tie"]["higher"]), int(r["tie"]["tied"]), k)
        r[f"ndcg{k}_order"] = ndcg_at_k(int(r["rank"]), k)
    return rows
