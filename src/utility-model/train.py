#!/usr/bin/env python3
"""Train utility-model v1 (TreeLSTM + reuse/opt_vec + listwise+huber)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


def _load_local_module(module_name: str, filename: str):
    path = os.path.join(THIS_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_data = _load_local_module("utility_v1_data", "data_utils.py")
_model = _load_local_module("utility_v1_model", "model.py")
_losses = _load_local_module("utility_v1_losses", "losses.py")

PlanTreeEncoder = _data.PlanTreeEncoder
fit_bao_hybrid_artifacts_from_rows = _data.fit_bao_hybrid_artifacts_from_rows
encode_row_opt_vec = _data.encode_row_opt_vec
encode_row_reuse = _data.encode_row_reuse
load_jsonl = _data.load_jsonl
split_by_target = _data.split_by_target
SourceTargetTreeModel = _model.SourceTargetTreeModel
total_loss = _losses.total_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UtilityDataset(Dataset):
    def __init__(self, rows: List[Dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]


class GroupedBatchSampler(BatchSampler):
    """Yield one target group per batch for listwise loss."""

    def __init__(self, rows: List[Dict], min_group_candidates: int = 2, shuffle: bool = True, seed: int = 42):
        self.rows = rows
        self.min_group_candidates = int(min_group_candidates)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        grouped: Dict[str, List[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            grouped[str(row["target_id"])].append(idx)
        self.groups = {k: v for k, v in grouped.items() if len(v) >= self.min_group_candidates}
        self.keys = sorted(self.groups.keys())

    def __iter__(self) -> Iterator[List[int]]:
        rnd = random.Random(self.seed + self.epoch)
        keys = list(self.keys)
        if self.shuffle:
            rnd.shuffle(keys)
        for k in keys:
            idxs = list(self.groups[k])
            if self.shuffle:
                rnd.shuffle(idxs)
            yield idxs

    def __len__(self) -> int:
        return len(self.keys)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def enrich_rows(rows: List[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        rr = dict(r)
        rr["target_id"] = str(rr.get("target_id", ""))
        rr["source_id"] = str(rr.get("source_id", ""))
        rr["weight"] = float(rr.get("weight", 1.0))
        rr["y"] = float(rr.get("y", 0.0))
        out.append(rr)
    return out


def pairwise_accuracy(target_ids: List[str], y_true: np.ndarray, y_pred: np.ndarray, min_delta: float = 0.0) -> float:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, tid in enumerate(target_ids):
        groups[tid].append(i)
    total = 0
    correct = 0
    for _, idxs in groups.items():
        for ai in range(len(idxs)):
            i = idxs[ai]
            for bi in range(ai + 1, len(idxs)):
                j = idxs[bi]
                if abs(y_true[i] - y_true[j]) <= min_delta:
                    continue
                total += 1
                truth = 1 if y_true[i] > y_true[j] else -1
                pred = 1 if y_pred[i] > y_pred[j] else -1
                if truth == pred:
                    correct += 1
    return (correct / total) if total > 0 else 0.0


def sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 0.01) -> float:
    """Sign consistency accuracy with tolerance band around zero."""
    if len(y_true) == 0:
        return 0.0

    def _sign_bucket(x: float) -> int:
        if x > eps:
            return 1
        if x < -eps:
            return -1
        return 0

    correct = 0
    total = 0
    for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
        total += 1
        if _sign_bucket(float(yt)) == _sign_bucket(float(yp)):
            correct += 1
    return (correct / total) if total > 0 else 0.0


def top1_metrics(rows: List[Tuple[str, float, float]], sign_eps: float = 0.05) -> Dict[str, float]:
    groups: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for tid, y, p in rows:
        groups[tid].append((y, p))
    if not groups:
        return {
            "top1_true_utility_mean": 0.0,
            "top1_negative_rate": 0.0,
            "top1_positive_rate": 0.0,
            "observed_top1_hit": 0.0,
        }

    util = []
    neg = 0
    pos = 0
    hit = 0
    for _, vals in groups.items():
        ys = np.array([x[0] for x in vals], dtype=np.float32)
        ps = np.array([x[1] for x in vals], dtype=np.float32)
        pred_top = int(np.argmax(ps))
        true_top = int(np.argmax(ys))
        if pred_top == true_top:
            hit += 1
        u = float(ys[pred_top])
        util.append(u)
        if u < -sign_eps:
            neg += 1
        if u > sign_eps:
            pos += 1

    n = float(len(groups))
    return {
        "top1_true_utility_mean": float(np.mean(util)) if util else 0.0,
        "top1_negative_rate": neg / n,
        "top1_positive_rate": pos / n,
        "observed_top1_hit": hit / n,
    }


def build_collate_fn(tree_encoder: PlanTreeEncoder, device: str):
    def collate_fn(batch: List[Dict]) -> Dict:
        source_features, source_children = [], []
        target_features, target_children = [], []
        reuse_list = []
        opt_vec_list = []
        y, w = [], []
        target_ids, source_ids = [], []

        for row in batch:
            sf, sc = tree_encoder.encode_path(row["source_plan_json"])
            tf, tc = tree_encoder.encode_path(row["target_plan_json"])
            source_features.append(sf.to(device))
            source_children.append(sc.to(device))
            target_features.append(tf.to(device))
            target_children.append(tc.to(device))
            reuse_list.append(encode_row_reuse(row).to(device))
            opt_vec_list.append(encode_row_opt_vec(row).to(device))
            y.append(float(row["y"]))
            w.append(float(row.get("weight", 1.0)))
            target_ids.append(str(row["target_id"]))
            source_ids.append(str(row["source_id"]))

        return {
            "source_features": source_features,
            "source_children": source_children,
            "target_features": target_features,
            "target_children": target_children,
            "reuse": torch.stack(reuse_list, dim=0),
            "opt_vec": torch.stack(opt_vec_list, dim=0),
            "y": torch.tensor(y, dtype=torch.float32, device=device),
            "weight": torch.tensor(w, dtype=torch.float32, device=device),
            "target_ids": target_ids,
            "source_ids": source_ids,
        }

    return collate_fn


def run_epoch(model: SourceTargetTreeModel, loader: DataLoader, optimizer: AdamW | None, args: argparse.Namespace) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    pair_losses = []
    sign_losses = []
    reg_losses = []

    all_rows: List[Tuple[str, float, float]] = []
    all_y = []
    all_pred = []
    all_tid = []

    for batch in loader:
        with torch.set_grad_enabled(is_train):
            out = model(
                source_features=batch["source_features"],
                source_children=batch["source_children"],
                target_features=batch["target_features"],
                target_children=batch["target_children"],
                reuse=batch["reuse"],
                opt_vec=batch["opt_vec"],
            )

            ld = total_loss(
                logits=out["logit"],
                scores=out["score"],
                labels=batch["y"],
                weight=batch["weight"],
                target_ids=batch["target_ids"],
                lambda_pair=args.lambda_pair,
                lambda_sign=args.lambda_sign,
                lambda_reg=args.lambda_reg,
                rank_min_delta=args.rank_min_delta,
                rank_temperature=args.rank_temperature,
                sign_eps=args.sign_eps,
                sign_temperature=args.sign_temperature,
            )
            loss = ld["total"]

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        pair_losses.append(float(ld["pair"].detach().cpu()))
        sign_losses.append(float(ld["sign"].detach().cpu()))
        reg_losses.append(float(ld["reg"].detach().cpu()))

        y_np = batch["y"].detach().cpu().numpy().tolist()
        p_np = out["score"].detach().cpu().numpy().tolist()
        for tid, yv, pv in zip(batch["target_ids"], y_np, p_np):
            all_rows.append((tid, float(yv), float(pv)))
            all_y.append(float(yv))
            all_pred.append(float(pv))
            all_tid.append(tid)

    y_true = np.asarray(all_y, dtype=np.float32)
    y_pred = np.asarray(all_pred, dtype=np.float32)
    top1 = top1_metrics(all_rows)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "pair_loss": float(np.mean(pair_losses)) if pair_losses else 0.0,
        "sign_loss": float(np.mean(sign_losses)) if sign_losses else 0.0,
        "reg_loss": float(np.mean(reg_losses)) if reg_losses else 0.0,
        "reg_mae": float(np.mean(np.abs(y_true - y_pred))) if len(y_true) > 0 else 0.0,
        "pair_acc": pairwise_accuracy(all_tid, y_true, y_pred),
        "sign_acc": sign_accuracy(y_true, y_pred, eps=args.sign_eps),
        "top1_true_utility_mean": top1["top1_true_utility_mean"],
        "top1_negative_rate": top1["top1_negative_rate"],
        "top1_positive_rate": top1["top1_positive_rate"],
        "observed_top1_hit": top1["observed_top1_hit"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train utility-model v1")
    parser.add_argument("--train_jsonl", type=str, default=os.path.join(ROOT_DIR, "outputs", "demo_pool", "utility_trials.jsonl"))
    parser.add_argument("--artifacts_dir", type=str, default=os.path.join(ROOT_DIR, "models", "cardinality_bias"))
    parser.add_argument("--predicate_fit_dir", type=str, default=os.path.join(ROOT_DIR, "outputs", "demo_pool"))
    parser.add_argument("--encoder_mode", type=str, choices=["current", "bao_hybrid"], default="current")
    parser.add_argument(
        "--encoder_artifacts_dir",
        type=str,
        default="",
        help="Path to encoder artifacts file for bao_hybrid. Default: <out_dir>/encoder_artifacts.json",
    )
    parser.add_argument("--use_predicate_pca", dest="use_predicate_pca", action="store_true")
    parser.add_argument("--no_use_predicate_pca", dest="use_predicate_pca", action="store_false")
    parser.set_defaults(use_predicate_pca=None)
    parser.add_argument("--db_name", type=str, default="")
    parser.add_argument("--out_dir", type=str, default=os.path.join(ROOT_DIR, "models", "utility"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text_device", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.2)

    parser.add_argument("--tree_hidden_dim", type=int, default=128)
    parser.add_argument("--scorer_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lambda_pair", type=float, default=1.0)
    parser.add_argument("--lambda_sign", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=0.0)
    parser.add_argument("--rank_min_delta", type=float, default=0.0)
    parser.add_argument("--rank_temperature", type=float, default=1.0)
    parser.add_argument("--sign_eps", type=float, default=0.01, help="Tolerance band for sign consistency.")
    parser.add_argument("--sign_temperature", type=float, default=1.0)

    parser.add_argument("--min_group_candidates", type=int, default=2)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--verbose_metrics", action="store_true", help="Print full metric set each epoch.")
    return parser.parse_args()


def save_ckpt(path: str, model: SourceTargetTreeModel, args: argparse.Namespace, metrics: Dict[str, float], epoch: int, feature_dim: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "feature_dim": int(feature_dim),
                "tree_hidden_dim": int(args.tree_hidden_dim),
                "scorer_hidden_dim": int(args.scorer_hidden_dim),
                "dropout": float(args.dropout),
                "opt_vec_dim": 4,
                "reuse_dim": 1,
                "encoder_mode": str(args.encoder_mode),
                "encoder_artifacts_dir": str(args.encoder_artifacts_dir),
                "use_predicate_pca": bool(args.use_predicate_pca),
            },
            "metrics": metrics,
            "epoch": int(epoch),
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = enrich_rows(load_jsonl(args.train_jsonl))
    train_rows, valid_rows = split_by_target(rows, valid_ratio=args.valid_ratio, seed=args.seed)

    if args.use_predicate_pca is None:
        args.use_predicate_pca = (str(args.encoder_mode).lower() == "current")

    if str(args.encoder_mode).lower() == "bao_hybrid":
        encoder_artifacts_path = args.encoder_artifacts_dir.strip() if args.encoder_artifacts_dir.strip() else os.path.join(args.out_dir, "encoder_artifacts.json")
        args.encoder_artifacts_dir = encoder_artifacts_path
        fit_bao_hybrid_artifacts_from_rows(
            rows=train_rows,
            output_path=encoder_artifacts_path,
            use_predicate_pca=bool(args.use_predicate_pca),
            model_name=os.environ.get("DIAGHINT_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            text_device=args.text_device,
        )

    print(f"rows={len(rows)} train={len(train_rows)} valid={len(valid_rows)}")
    valid_target_ids = sorted({str(r.get("target_id", "")) for r in valid_rows if str(r.get("target_id", "")).strip()})
    print(f"valid targets ({len(valid_target_ids)}): {', '.join(valid_target_ids)}")

    tree_encoder = PlanTreeEncoder(
        artifacts_dir=args.artifacts_dir,
        predicate_fit_dir=args.predicate_fit_dir,
        model_name=os.environ.get("DIAGHINT_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        text_device=args.text_device,
        db_name=(args.db_name.strip() or None),
        encoder_mode=args.encoder_mode,
        encoder_artifacts_dir=args.encoder_artifacts_dir,
        use_predicate_pca=bool(args.use_predicate_pca),
    )
    feature_dim = int(tree_encoder.feature_dim)
    print(
        f"feature_dim={feature_dim} "
        f"(encoder_mode={args.encoder_mode}, use_predicate_pca={bool(args.use_predicate_pca)})"
    )

    if train_rows:
        print("opt_vec preview (first 5 train rows):")
        for row in train_rows[:5]:
            ov = encode_row_opt_vec(row).tolist()
            rv = encode_row_reuse(row).item()
            print(f"  source={row['source_id']} target={row['target_id']} reuse={rv:.0f} opt_vec={ov}")

    collate_fn = build_collate_fn(tree_encoder, args.device)

    train_loader = DataLoader(
        UtilityDataset(train_rows),
        batch_sampler=GroupedBatchSampler(train_rows, min_group_candidates=args.min_group_candidates, shuffle=True, seed=args.seed),
        num_workers=0,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        UtilityDataset(valid_rows),
        batch_sampler=GroupedBatchSampler(valid_rows, min_group_candidates=1, shuffle=False, seed=args.seed),
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = SourceTargetTreeModel(
        node_input_dim=feature_dim,
        tree_hidden_dim=args.tree_hidden_dim,
        scorer_hidden_dim=args.scorer_hidden_dim,
        dropout=args.dropout,
    ).to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metrics = None
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        train_m = run_epoch(model, train_loader, optimizer, args)
        valid_m = run_epoch(model, valid_loader, None, args)

        history.append({"epoch": epoch, "train": train_m, "valid": valid_m})

        if args.verbose_metrics:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_m['loss']:.5f} "
                f"valid_loss={valid_m['loss']:.5f} "
                f"valid_pair_loss={valid_m['pair_loss']:.5f} "
                f"valid_sign_loss={valid_m['sign_loss']:.5f} "
                f"valid_pair={valid_m['pair_acc']:.4f} "
                f"valid_sign={valid_m['sign_acc']:.4f} "
                f"valid_reg_mae={valid_m['reg_mae']:.4f} "
                f"top1_true_u={valid_m['top1_true_utility_mean']:.3f} "
                f"top1_neg={valid_m['top1_negative_rate']:.3f} "
                f"top1_pos={valid_m['top1_positive_rate']:.3f}"
            )
        else:
            print(
                f"epoch={epoch:03d} "
                f"valid_pair={valid_m['pair_acc']:.4f} "
                f"valid_sign={valid_m['sign_acc']:.4f} "
                f"valid_loss={valid_m['loss']:.5f}"
            )

        improved = False
        if best_metrics is None:
            improved = True
        else:
            if valid_m["pair_acc"] != best_metrics["pair_acc"]:
                improved = valid_m["pair_acc"] > best_metrics["pair_acc"]
            elif valid_m["sign_acc"] != best_metrics["sign_acc"]:
                improved = valid_m["sign_acc"] > best_metrics["sign_acc"]
            else:
                improved = valid_m["loss"] < best_metrics["loss"]

        if improved:
            best_metrics = dict(valid_m)
            no_improve = 0
            save_ckpt(os.path.join(args.out_dir, "best.pt"), model, args, valid_m, epoch, feature_dim=feature_dim)
            print(
                f"new_best: epoch={epoch:03d} "
                f"pair={valid_m['pair_acc']:.3f} "
                f"sign={valid_m['sign_acc']:.3f} "
                f"reg_mae={valid_m['reg_mae']:.4f}"
            )
        else:
            no_improve += 1

        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"early_stop: epoch={epoch:03d} patience={args.early_stop_patience}")
            break

    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"saved checkpoint: {os.path.join(args.out_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
