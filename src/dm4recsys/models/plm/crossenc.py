"""CrossEnc: cross-encoder relevance of each candidate title to the history.
Model: Qwen3-Ranker-0.6B, used exactly as in its model card: system prompt + '<Instruct>: ...
<Query>: {history} <Document>: {title}', empty thinking block, score = P("yes") = softmax over the "no"/"yes"
logits at the last position. All 20 pairs of one user in one left-padded batch. Query = the history block every
other model gets; <Instruct> = the retrieval instruction of configs/models/plm.yaml; Document = the plain title."""
from __future__ import annotations

import torch

from ...utils import sync_device
from .common import PLMRanker, torch_dtype

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
          "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_pair(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


class CrossEnc(PLMRanker):
    def __init__(self, mcfg, cfg):
        super().__init__(mcfg, cfg)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        rev = mcfg.get("revision")
        self.tok = AutoTokenizer.from_pretrained(mcfg["hf_model_id"], revision=rev, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            mcfg["hf_model_id"], revision=rev, torch_dtype=torch_dtype(mcfg, self.device)).to(self.device).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix = self.tok.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tok.encode(SUFFIX, add_special_tokens=False)

    def inputs(self, req, cands: list[str]):
        history, instr = self.texts(req)
        pairs = [format_pair(instr, history, c) for c in cands]
        budget = self.mcfg.get("max_length", 8192) - len(self.prefix) - len(self.suffix)
        enc = self.tok(pairs, padding=False, truncation="longest_first", return_attention_mask=False,
                       max_length=budget)
        enc["input_ids"] = [self.prefix + ids + self.suffix for ids in enc["input_ids"]]
        enc = self.tok.pad(enc, padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def score(self, req, cands):
        logits = self.model(**self.inputs(req, cands)).logits[:, -1, :].float()
        p = torch.softmax(logits[:, [self.no, self.yes]], -1)[:, 1]
        sync_device(self.device)
        return p.cpu().tolist()
