from ..base import RankRequest, RankResult, Ranker, rank_from_scores


class Popularity(Ranker):
    """Popularity reference: an item's score = its number of interactions in the whole dataset
    except every user's test target (= the final BPR's training data). The 20 candidates are sorted by that count;
    ties get a random order seeded per user (`seed`, dataset, user), so the ranking is fixed across reruns."""

    def __init__(self, mcfg, cfg, popularity: dict[str, dict[str, int]]):
        super().__init__(mcfg, cfg)
        self.pop = popularity  # dataset -> item -> count

    def rank(self, req: RankRequest) -> RankResult:
        p = self.pop[req.dataset]
        scores = {o: float(p.get(i, 0)) for o, i in zip(req.option_ids, req.cand_items)}
        order = rank_from_scores(scores, req.option_ids, self.name, self.mcfg.get("seed", 2027), req.dataset, req.user)
        return RankResult(order=order, scores=scores, model_version="popularity")
