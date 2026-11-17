"""EmbSim: cosine similarity between the history embedding and each candidate-title embedding.
Model: Qwen3-Embedding-0.6B. Input format of the model
card: the query gets 'Instruct: {instruction}\nQuery:' in front, the documents (titles) get no prompt. Query = the
history block every other model gets; instruction = the retrieval instruction of configs/models/plm.yaml."""
from __future__ import annotations

import torch

from ...utils import sync_device
from .common import PLMRanker, torch_dtype


def query_prompt(instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery:"


class EmbSim(PLMRanker):
    def __init__(self, mcfg, cfg):
        super().__init__(mcfg, cfg)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(mcfg["hf_model_id"], revision=mcfg.get("revision"), device=self.device,
                                         trust_remote_code=mcfg.get("trust_remote_code", False),
                                         model_kwargs={"torch_dtype": torch_dtype(mcfg, self.device)})

    @torch.no_grad()
    def score(self, req, cands):
        history, instruction = self.texts(req)
        q = self.model.encode(history, prompt=query_prompt(instruction), convert_to_tensor=True, show_progress_bar=False)
        c = self.model.encode(cands, convert_to_tensor=True, show_progress_bar=False)
        s = torch.nn.functional.cosine_similarity(q.float().unsqueeze(0), c.float(), dim=-1)
        sync_device(self.device)
        return s.cpu().tolist()
