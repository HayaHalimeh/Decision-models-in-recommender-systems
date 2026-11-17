from ...utils import rng_for
from ..base import RankRequest, RankResult, Ranker


class RandomRank(Ranker):
    def rank(self, req: RankRequest) -> RankResult:
        rng = rng_for("random", self.mcfg.get("seed", 2027), req.dataset, req.user, req.run)
        return RankResult(order=[req.option_ids[i] for i in rng.permutation(len(req.option_ids))],
                          model_version="random")
