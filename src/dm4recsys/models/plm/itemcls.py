"""ItemCLS: NLI entailment with the history as premise and the candidate as hypothesis
(DeBERTa-v3-large fine-tuned on NLI).
Premise = the history block every other model gets; NLI takes no instruction, and the
hypothesis must be a statement: "The user will {verb} {item} next." (verb per dataset). Score = softmax over
[contradiction, entailment] logits, entailment component. Label indices are read from the model config.
DeBERTa takes 512 tokens: if a (premise, hypothesis) pair is longer, the OLDEST history items are dropped until
every pair fits (the lead-in sentence and the most recent items are kept); the number dropped is logged."""
from __future__ import annotations

import torch

from ...utils import sync_device
from .common import PLMRanker, torch_dtype


def nli_label_indices(id2label: dict) -> tuple[int, int]:
    lab = {int(k): str(v).lower() for k, v in id2label.items()}
    c = next(k for k, v in lab.items() if v.startswith("contra"))
    e = next(k for k, v in lab.items() if v.startswith("entail"))
    return c, e


class ItemCLS(PLMRanker):
    def __init__(self, mcfg, cfg):
        super().__init__(mcfg, cfg)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        rev = mcfg.get("revision")
        self.tok = AutoTokenizer.from_pretrained(mcfg["hf_model_id"], revision=rev)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            mcfg["hf_model_id"], revision=rev, torch_dtype=torch_dtype(mcfg, self.device)).to(self.device).eval()
        self.c_idx, self.e_idx = nli_label_indices(self.model.config.id2label)
        self.max_len = min(self.tok.model_max_length, self.mcfg.get("max_length", 512))

    def hypotheses(self, req, cands):
        verb = self.cfg["datasets"][req.dataset]["itemcls_verb"]
        return [self.mcfg["hypothesis_template"].format(verb=verb, item=c) for c in cands]

    def fitted_premise(self, req, hyps) -> tuple[str, int]:
        """History block with as many of the most recent items as fit next to the longest hypothesis."""
        titles = list(req.history_titles)
        longest = max(hyps, key=lambda h: len(self.tok(h, add_special_tokens=False)["input_ids"]))
        for drop in range(len(titles) + 1):
            premise, _ = self.texts(req, titles[drop:])
            if len(self.tok(premise, longest)["input_ids"]) <= self.max_len or drop == len(titles):
                return premise, drop
        raise AssertionError

    @torch.no_grad()
    def score(self, req, cands):
        hyps = self.hypotheses(req, cands)
        premise, dropped = self.fitted_premise(req, hyps)
        self.last_info["itemcls_history_items_dropped"] = dropped
        enc = self.tok([premise] * len(hyps), hyps, return_tensors="pt", padding=True, truncation="only_second",
                       max_length=self.max_len)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits.float()
        p = torch.softmax(logits[:, [self.c_idx, self.e_idx]], -1)[:, 1]
        sync_device(self.device)
        return p.cpu().tolist()
