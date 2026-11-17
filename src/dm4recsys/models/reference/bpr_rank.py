from ..base import RankRequest, RankResult, Ranker, rank_from_scores


class BPRRank(Ranker):
    """The retriever's own ranking of the 20 frozen candidates (stored BPR scores)."""

    def rank(self, req: RankRequest) -> RankResult:
        scores = dict(zip(req.option_ids, req.bpr_scores))
        return RankResult(order=rank_from_scores(scores, req.option_ids, self.name, req.dataset, req.user),
                          scores=scores, model_version="bpr-retriever")
