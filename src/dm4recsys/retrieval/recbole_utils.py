"""Thin layer over RecBole: atomic files, config, a training loop that keeps the best state in memory
(no torch.load of RecBole checkpoints, which breaks with newer torch), and scoring helpers."""
from __future__ import annotations

import copy
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def write_atomic(out_dir: Path, ds_name: str, sequences: dict[str, list[str]], all_items: list[str]) -> Path:
    """Write <ds>.inter (timestamp = position, so RecBole's ordering equals ours exactly) and <ds>.item
    (the full catalogue, so the item vocabulary is always the full dataset's)."""
    d = Path(out_dir) / ds_name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{ds_name}.inter", "w") as f:
        f.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for u, seq in sequences.items():
            for p, it in enumerate(seq):
                f.write(f"{u}\t{it}\t{p}\n")
    with open(d / f"{ds_name}.item", "w") as f:
        f.write("item_id:token\n")
        for it in all_items:
            f.write(f"{it}\n")
    return Path(out_dir)


def recbole_config(model: str, data_path: Path, ds_name: str, params: dict, run_cfg: dict, split: str,
                   seed: int, device: str):
    from recbole.config import Config

    is_seq = model in ("SASRec", "BERT4Rec", "GRU4Rec")
    cd = {
        "data_path": str(data_path),
        "dataset": ds_name,
        "load_col": {"inter": ["user_id", "item_id", "timestamp"], "item": ["item_id"]},
        "user_inter_num_interval": "[0,inf)",
        "item_inter_num_interval": "[0,inf)",
        "eval_args": {"split": {"LS": split}, "order": "TO", "group_by": "user", "mode": "full"},
        "metrics": ["NDCG", "Recall"],
        "topk": [10, 20],
        "valid_metric": run_cfg.get("valid_metric", "NDCG@10"),
        "epochs": run_cfg["max_epochs"],
        "stopping_step": run_cfg["patience"],
        "train_batch_size": run_cfg["train_batch_size"],
        "eval_batch_size": run_cfg["eval_batch_size"],
        "seed": seed,
        "reproducibility": True,
        "show_progress": False,
        "state": "ERROR",
        "use_gpu": device.startswith("cuda"),
        # RecBole writes gpu_id into CUDA_VISIBLE_DEVICES; keep the GPU the job was started with
        "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "checkpoint_dir": str(Path(data_path) / "_ckpt"),
        "save_dataset": False,
        "save_dataloaders": False,
    }
    if is_seq:
        cd["MAX_ITEM_LIST_LENGTH"] = run_cfg.get("max_seq_length", 50)
        cd["train_neg_sample_args"] = None
    else:
        cd["train_neg_sample_args"] = {"distribution": "uniform", "sample_num": 1}
    cd.update(params)
    argv, sys.argv = sys.argv, sys.argv[:1]  # RecBole parses sys.argv; keep our CLI flags away from it
    try:
        config = Config(model=model, config_dict=cd)
    finally:
        sys.argv = argv
    logging.getLogger().setLevel(logging.WARNING)  # RecBole logs to the root logger
    if device == "mps":
        config["device"] = torch.device("mps")
    elif device == "cpu":
        config["device"] = torch.device("cpu")
    return config


def env_info(device: str | None) -> dict:
    """Software/hardware that trained a model (needed to rebuild results bit-exactly)."""
    import platform

    import recbole

    env = {"device": device, "torch": torch.__version__, "recbole": recbole.__version__,
           "python": platform.python_version(), "platform": platform.platform()}
    if device and str(device).startswith("cuda") and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
    return env


def topk_metric(rank: torch.Tensor, metric: str) -> torch.Tensor:
    """Per-user metric from the 0-based rank of the single target: NDCG@k or Recall@k (= Hit@k for one target)."""
    name, k = metric.split("@")
    hit = rank < int(k)
    if name.lower() == "ndcg":
        return torch.where(hit, 1.0 / torch.log2(rank.float() + 2.0), torch.zeros_like(rank, dtype=torch.float))
    if name.lower() in ("recall", "hit"):
        return hit.float()
    raise ValueError(f"unsupported validation metric {metric!r} (NDCG@k or Recall@k)")


def seen_index(dataset, seen: dict[str, list[str]]) -> dict[int, torch.Tensor]:
    """user token -> its seen item tokens, as RecBole internal ids."""
    uf, itf = dataset.uid_field, dataset.iid_field
    return {int(dataset.token2id(uf, u)): torch.as_tensor(np.atleast_1d(dataset.token2id(itf, items)), dtype=torch.long)
            for u, items in seen.items() if items}


def mask_and_rank(scores: torch.Tensor, targets: torch.Tensor, uids: list[int], seen: dict[int, torch.Tensor]):
    """Set [PAD] (column 0) and each row's seen items to -inf, then return the 0-based rank of each target
    (number of items scored strictly higher)."""
    scores = scores.clone()
    scores[:, 0] = -np.inf
    rows, cols = [], []
    for r, u in enumerate(uids):
        ids = seen.get(int(u))
        if ids is not None and len(ids):
            rows.append(torch.full_like(ids, r))
            cols.append(ids)
    if rows:
        scores[torch.cat(rows).to(scores.device), torch.cat(cols).to(scores.device)] = -np.inf
    t = scores.gather(1, targets.view(-1, 1).to(scores.device))
    return (scores > t).sum(1)


@torch.no_grad()
def masked_valid(model, valid_data, dataset, seen: dict[int, torch.Tensor], metric: str) -> float:
    """Sequential models: full ranking of the validation target with the user's earlier items excluded (as BPR's
    RecBole evaluation and our test-time candidates do; RecBole masks nothing for sequential models)."""
    model.eval()
    dev = next(model.parameters()).device
    vals = []
    for inter, _, _, pos_i in valid_data:
        inter = inter.to(dev)
        s = model.full_sort_predict(inter).view(len(pos_i), -1)
        rank = mask_and_rank(s, pos_i, inter[dataset.uid_field].tolist(), seen)
        vals.append(topk_metric(rank, metric))
    return float(torch.cat(vals).mean())


def train(model_name: str, data_path: Path, ds_name: str, params: dict, run_cfg: dict, split: str,
          seed: int, device: str, fixed_epochs: int | None = None, seen: dict[str, list[str]] | None = None) -> dict:
    """Train a RecBole model. With split='valid_and_test': early stopping on the validation target,
    returns the best state. With split='test_only' and fixed_epochs: train exactly that many epochs.
    seen (sequential models): user -> items before the validation target, excluded from the validation ranking."""
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_seed

    config = recbole_config(model_name, data_path, ds_name, params, run_cfg, split, seed, device)
    init_seed(seed, True)
    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)
    init_seed(seed, True)
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], model_name)(config, model)
    seen_ids = seen_index(dataset, seen) if seen is not None else None
    best_score, best_epoch, best_state, history, bad = -np.inf, -1, None, [], 0
    n_epochs = fixed_epochs if fixed_epochs is not None else run_cfg["max_epochs"]
    for ep in range(n_epochs):
        model.train()
        loss = trainer._train_epoch(train_data, ep, show_progress=False)
        loss = float(sum(loss)) if isinstance(loss, tuple) else float(loss)
        if fixed_epochs is None:
            if seen_ids is not None:
                score = masked_valid(model, valid_data, dataset, seen_ids, config["valid_metric"])
            else:
                score, res = trainer._valid_epoch(valid_data, show_progress=False)
            history.append({"epoch": ep, "loss": loss, "valid": float(score)})
            if score > best_score:
                best_score, best_epoch, bad = float(score), ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= run_cfg["patience"]:
                    break
        else:
            history.append({"epoch": ep, "loss": loss})
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = n_epochs - 1
    model.eval()
    return {"model": model, "dataset": dataset, "config": config, "best_valid": best_score,
            "best_epoch": best_epoch, "history": history, "valid_masked_seen": seen_ids is not None}


def load_trained(model_name: str, data_path: Path, ds_name: str, params: dict, run_cfg: dict, split: str,
                 seed: int, device: str, state_path: Path) -> dict:
    """Rebuild the RecBole dataset/model exactly as in train() and load saved weights (our own state_dict)."""
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, init_seed

    config = recbole_config(model_name, data_path, ds_name, params, run_cfg, split, seed, device)
    init_seed(seed, True)
    dataset = create_dataset(config)
    train_data, _, _ = data_preparation(config, dataset)
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    model.load_state_dict(torch.load(state_path, map_location=config["device"], weights_only=True))
    model.eval()
    return {"model": model, "dataset": dataset, "config": config}


@torch.no_grad()
def bpr_full_scores(model, dataset, user_tokens: list[str]) -> np.ndarray:
    from recbole.data.interaction import Interaction

    uid = torch.tensor(dataset.token2id("user_id", user_tokens), device=next(model.parameters()).device)
    s = model.full_sort_predict(Interaction({"user_id": uid}))
    return s.view(len(user_tokens), -1).float().cpu().numpy()


@torch.no_grad()
def seq_full_scores(model, dataset, item_seq_tokens: list[str], max_len: int) -> np.ndarray:
    """Score all items given a raw item-token sequence (oldest -> newest), batch size 1."""
    from recbole.data.interaction import Interaction

    dev = next(model.parameters()).device
    ids = dataset.token2id("item_id", item_seq_tokens[-max_len:]) if item_seq_tokens else []
    ids = np.atleast_1d(ids)
    seq = torch.zeros(1, max_len, dtype=torch.long, device=dev)
    if len(ids):
        seq[0, : len(ids)] = torch.as_tensor(ids, device=dev)
    length = torch.tensor([max(1, len(ids))], device=dev)
    inter = Interaction({"item_id_list": seq, "item_length": length})
    s = model.full_sort_predict(inter)
    return s.view(-1).float().cpu().numpy()


def item_index(dataset, items: list[str]) -> np.ndarray:
    return np.atleast_1d(dataset.token2id("item_id", items))
