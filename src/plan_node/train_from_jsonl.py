# train_from_jsonl.py
# Unified training script:
# 1) extract node samples from pool directory (dedupe by hint per parent), or
# 2) load existing jsonl
# then split by template_id, encode with new structured features, and train.
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directories to path for direct script execution compatibility.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plan_node.encoder import (
    JOIN_NODE_TYPES,
    SCAN_NODE_TYPES,
    build_table_vocab_from_plan_files,
    collect_plan_files,
    encode_node_features,
    parse_plan_node,
    safe_float,
)
from plan_node.model import NodeEstimatorNet
from plan_node.utils import qerror_and_bucket, bucket2id

# Disable tokenizer parallelism warning (safe no-op if tokenizer not used)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

NUM_BUCKETS = 5


def _log_transform(x: float) -> float:
    import math
    return math.log(max(x, 0.001) + 0.001)


def _extract_template_id(parent_id: str) -> str:
    m = re.match(r"(\d+)", str(parent_id))
    return m.group(1) if m else ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _collect_plan_paths_with_hint_dedupe(plan_dir: str) -> Tuple[List[Path], Dict[str, int]]:
    """Collect plan.json paths from pool dir and dedupe by hint within each parent dir.

    Expected structure: <plan_dir>/<parent>/<idx>/plan.json
    """
    base = Path(plan_dir)
    if not base.exists():
        raise FileNotFoundError(f"plan_dir does not exist: {plan_dir}")

    selected: List[Path] = []
    total_candidates = 0
    dedup_skipped = 0

    for parent in sorted(base.iterdir()):
        if not parent.is_dir():
            continue

        seen_hints = set()
        for child in sorted(parent.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            plan_path = child / "plan.json"
            if not plan_path.exists():
                continue

            total_candidates += 1
            hint_text = _read_text(child / "hint.txt")
            # Deduplicate by hint within the same parent; keep first seen.
            if hint_text in seen_hints:
                dedup_skipped += 1
                continue
            seen_hints.add(hint_text)
            selected.append(plan_path)

    stats = {
        "total_candidates": total_candidates,
        "selected_after_hint_dedupe": len(selected),
        "dedup_skipped": dedup_skipped,
    }
    return selected, stats


def _extract_nodes_from_plan_obj(
    plan_obj: Dict,
    check_actual_rows: bool = True,
) -> Tuple[List[Dict], bool]:
    root = plan_obj.get("Plan", plan_obj)
    nodes, has_actual_rows = parse_plan_node(root, depth=0, check_actual_rows=check_actual_rows)
    return nodes, has_actual_rows


def extract_nodes_from_pool(plan_dir: str, check_actual_rows: bool = True) -> Tuple[List[Dict], Dict[str, int]]:
    """Extract node samples from pool plan directory with hint dedupe per parent."""
    selected_paths, pick_stats = _collect_plan_paths_with_hint_dedupe(plan_dir)

    nodes: List[Dict] = []
    files_missing_actual = 0
    files_failed = 0

    for plan_path in tqdm(selected_paths, desc="Extracting nodes"):
        try:
            obj = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception:
            files_failed += 1
            continue

        extracted, has_actual = _extract_nodes_from_plan_obj(obj, check_actual_rows=check_actual_rows)
        if check_actual_rows and not has_actual:
            files_missing_actual += 1
            continue

        rel = plan_path.relative_to(Path(plan_dir))
        # Expected: parent/idx/plan.json
        parent_id = rel.parts[0] if len(rel.parts) >= 3 else ""
        plan_index = rel.parts[1] if len(rel.parts) >= 3 else ""
        template_id = _extract_template_id(parent_id)

        for n in extracted:
            # Keep only join/scan nodes (parse_plan_node already does this; defensive guard)
            if n.get("node_type") not in JOIN_NODE_TYPES and n.get("node_type") not in SCAN_NODE_TYPES:
                continue
            sample = dict(n)
            sample["parent_id"] = parent_id
            sample["template_id"] = template_id
            sample["plan_index"] = str(plan_index)
            sample["query_id"] = parent_id  # compatibility for old analytics
            sample["plan_file"] = str(plan_path)
            nodes.append(sample)

    stats = {
        **pick_stats,
        "files_missing_actual": files_missing_actual,
        "files_failed": files_failed,
        "nodes_extracted": len(nodes),
    }
    return nodes, stats


def load_nodes_from_jsonl(jsonl_path: str) -> List[Dict]:
    """Load node data from jsonl file (compat mode)."""
    nodes: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                n = json.loads(line)
            except Exception as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue

            # Normalize legacy fields into new encoder schema.
            normalized = {
                "node_type": n.get("node_type", ""),
                "depth": safe_float(n.get("plan_depth", n.get("depth", 0.0))),
                "self_cost": safe_float(n.get("self_cost", n.get("total_cost", 0.0))),
                "total_cost": safe_float(n.get("total_cost", 0.0)),
                "plan_rows": safe_float(n.get("plan_rows", 0.0)),
                "plan_width": safe_float(n.get("plan_width", 0.0)),
                "actual_rows": safe_float(n.get("actual_rows", 0.0)),
                "table_name": n.get("table_name", ""),
                "alias": n.get("alias", ""),
                "predicates": n.get("predicates", [n.get("filter", "")] if n.get("filter") else []),
                "columns": n.get("columns", []),
            }

            alias = str(normalized.get("alias", "") or "").strip()
            rel = str(normalized.get("table_name", "") or "").strip()
            tok = alias or rel
            normalized["table_set"] = [tok] if tok else []

            # total_rows for selectivity in new encoding.
            # If unavailable, fallback to plan_rows to keep selectivity in sane range.
            normalized["total_rows"] = safe_float(
                n.get("total_rows", n.get("plan_rows", 0.0))
            )

            parent_id = str(n.get("parent_id", n.get("query_id", "")))
            template_id = str(n.get("template_id", _extract_template_id(parent_id)))
            normalized["parent_id"] = parent_id
            normalized["template_id"] = template_id
            normalized["plan_index"] = str(n.get("plan_index", ""))
            normalized["query_id"] = str(n.get("query_id", parent_id))
            normalized["plan_file"] = str(n.get("plan_file", ""))

            # Skip unusable supervised labels.
            if normalized["actual_rows"] <= 0:
                continue
            nodes.append(normalized)

    print(f"Loaded {len(nodes)} nodes from {jsonl_path}")
    return nodes


def split_nodes_by_template(nodes: List[Dict], train_ratio: float = 0.8, seed: int = 42):
    """Split by template_id (numeric prefix), avoiding template leakage."""
    groups: Dict[str, List[Dict]] = {}
    for n in nodes:
        tid = str(n.get("template_id", ""))
        if not tid:
            pid = str(n.get("parent_id", n.get("query_id", "")))
            tid = _extract_template_id(pid) or "unknown"
            n["template_id"] = tid
        groups.setdefault(tid, []).append(n)

    template_ids = sorted(groups.keys(), key=lambda x: (x == "unknown", int(x) if x.isdigit() else 10**9, x))
    n_templates = len(template_ids)
    if n_templates == 0:
        return [], [], set(), set()

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_templates)
    train_n = max(1, int(n_templates * train_ratio)) if n_templates > 1 else 1
    train_n = min(train_n, n_templates - 1) if n_templates > 1 else 1

    train_tids = set(template_ids[i] for i in perm[:train_n])
    valid_tids = set(template_ids[i] for i in perm[train_n:])
    if not valid_tids:
        # fallback for tiny datasets
        t = next(iter(train_tids))
        train_tids.remove(t)
        valid_tids.add(t)

    train_nodes: List[Dict] = []
    valid_nodes: List[Dict] = []
    for n in nodes:
        if n.get("template_id") in train_tids:
            train_nodes.append(n)
        else:
            valid_nodes.append(n)

    return train_nodes, valid_nodes, train_tids, valid_tids


def compute_norm_stats(nodes: List[Dict]):
    self_costs = []
    plan_rows_list = []
    total_costs = []
    plan_widths = []
    depths = []

    for node in nodes:
        self_costs.append(safe_float(node.get("self_cost", 0.0)))
        plan_rows_list.append(safe_float(node.get("plan_rows", 0.0)))
        total_costs.append(safe_float(node.get("total_cost", 0.0)))
        plan_widths.append(safe_float(node.get("plan_width", 0.0)))
        depths.append(safe_float(node.get("depth", 0.0)))

    self_costs_log = np.array([_log_transform(x) for x in self_costs])
    plan_rows_log = np.array([_log_transform(x) for x in plan_rows_list])
    total_costs_log = np.array([_log_transform(x) for x in total_costs])
    plan_widths_log = np.array([_log_transform(x) for x in plan_widths])
    depths_arr = np.array(depths)

    norm_stats = (
        np.min(self_costs_log), np.max(self_costs_log),
        np.min(plan_rows_log), np.max(plan_rows_log),
        np.min(total_costs_log), np.max(total_costs_log),
        np.min(plan_widths_log), np.max(plan_widths_log),
        np.min(depths_arr), np.max(depths_arr),
    )

    print("Normalization statistics:")
    print(f"  Self cost: [{norm_stats[0]:.4f}, {norm_stats[1]:.4f}]")
    print(f"  Plan rows: [{norm_stats[2]:.4f}, {norm_stats[3]:.4f}]")
    print(f"  Total cost: [{norm_stats[4]:.4f}, {norm_stats[5]:.4f}]")
    print(f"  Plan width: [{norm_stats[6]:.4f}, {norm_stats[7]:.4f}]")
    print(f"  Depth: [{norm_stats[8]:.4f}, {norm_stats[9]:.4f}]")
    return norm_stats


def encode_nodes(nodes: List[Dict], table_to_idx: Dict[str, int], norm_stats):
    X_list = []
    y_list = []
    q_list = []
    node_types = []

    for n in tqdm(nodes, desc="Encoding nodes"):
        feat = encode_node_features(
            n,
            encoder=None,
            norm_stats=norm_stats,
            table_to_idx=table_to_idx,
        )

        est = safe_float(n.get("plan_rows", 0.0))
        act = safe_float(n.get("actual_rows", 0.0))
        if act <= 0:
            continue

        bucket_name, q = qerror_and_bucket(est, act)
        y_list.append(bucket2id[bucket_name])
        q_list.append(q)
        X_list.append(feat)
        node_types.append(n.get("node_type", ""))

    if not X_list:
        return None, None, None, None

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    qerrors = np.array(q_list, dtype=np.float32)
    node_types = np.array(node_types, dtype=object)
    return X, y, qerrors, node_types


class SimpleDataset:
    def __init__(self, X, y, logq):
        self.X = X
        self.y = y
        self.logq = logq

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "feat": self.X[idx],
            "bucket": self.y[idx],
            "log_q": self.logq[idx],
        }


def collate_fn(batch):
    feats = np.stack([b["feat"] for b in batch]).astype(np.float32)
    feats = torch.from_numpy(feats)
    buckets = torch.tensor([b["bucket"] for b in batch], dtype=torch.long)
    logq = torch.tensor([b["log_q"] for b in batch], dtype=torch.float32)
    return feats, buckets, logq


def compute_directional_lite_accuracy(labels, preds, qerrors):
    correct = 0
    for label, pred, q in zip(labels, preds, qerrors):
        if label == pred:
            correct += 1
        elif label == 2:
            if pred in [0, 1] and q < 1:
                correct += 1
            elif pred in [3, 4] and q > 1:
                correct += 1
    return correct / len(labels) if len(labels) > 0 else 0.0


def train_epoch(model, loader, opt, scaler, device, args):
    model.train()
    losses = []
    all_preds = []
    all_labels = []
    bucket_to_metaclass = torch.tensor([0, 0, 1, 2, 2], device=device)

    for step, (feats, buckets, logq) in enumerate(tqdm(loader, desc="train", position=0, leave=False)):
        feats = feats.to(device)
        buckets = buckets.to(device)
        logq = logq.to(device)

        with torch.amp.autocast("cuda", enabled=args.amp):
            logits, pred_logq = model(feats)
            c_loss = F.cross_entropy(logits, buckets)
            r_loss = F.mse_loss(pred_logq, logq)

            probs = F.softmax(logits, dim=1)
            true_metaclasses = bucket_to_metaclass[buckets]
            all_metaclasses = bucket_to_metaclass.unsqueeze(0).expand(logits.size(0), -1)
            true_metaclasses_expanded = true_metaclasses.unsqueeze(1).expand(-1, NUM_BUCKETS)
            penalty_mask = (all_metaclasses != true_metaclasses_expanded).float()
            wrong_metaclass_probs = probs * penalty_mask
            penalty_per_sample = torch.sum(wrong_metaclass_probs, dim=1)
            penalty_loss = penalty_per_sample.mean()

            loss = c_loss + args.lambda_reg * r_loss + args.gamma * penalty_loss

        scaler.scale(loss).backward()

        if (step + 1) % args.grad_accum == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        losses.append(loss.item())
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(buckets.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return float(np.mean(losses)), acc, f1


@torch.no_grad()
def eval_epoch(model, loader, device, args):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    all_qerrors = []

    for feats, buckets, logq in tqdm(loader, desc="eval", position=1, leave=False):
        feats = feats.to(device)
        buckets = buckets.to(device)
        logq = logq.to(device)

        logits, pred_logq = model(feats)
        c_loss = F.cross_entropy(logits, buckets)
        r_loss = F.mse_loss(pred_logq, logq)
        loss = c_loss + args.lambda_reg * r_loss
        losses.append(loss.item())

        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(buckets.detach().cpu().numpy().tolist())

        qerrors = torch.expm1(logq).detach().cpu().numpy()
        all_qerrors.extend(qerrors.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    lite_acc = compute_directional_lite_accuracy(all_labels, all_preds, all_qerrors)
    lite_recall = lite_acc
    return float(np.mean(losses)), acc, f1, lite_acc, lite_recall


def _print_split_report(train_nodes, valid_nodes, train_tids, valid_tids):
    overlap = train_tids.intersection(valid_tids)
    print("\nTemplate split report:")
    print(f"  Train templates: {len(train_tids)}")
    print(f"  Valid templates: {len(valid_tids)}")
    print(f"  Template overlap size: {len(overlap)}")
    print(f"  Train nodes: {len(train_nodes)}")
    print(f"  Valid nodes: {len(valid_nodes)}")
    if overlap:
        print(f"  Overlap templates (unexpected): {sorted(list(overlap))}")


def _build_table_vocab_from_nodes(nodes: List[Dict]) -> Dict[str, int]:
    tables = set()
    for n in nodes:
        for tok in n.get("table_set", []) or []:
            tok = str(tok).strip()
            if tok:
                tables.add(tok)
    ordered = sorted(tables)
    return {t: i for i, t in enumerate(ordered)}


def main():
    parser = argparse.ArgumentParser(description="Train node classification model from plan_dir or jsonl")
    parser.add_argument("--input", type=str, default=None, help="Input jsonl file path (compat mode)")
    parser.add_argument("--plan_dir", type=str, default=None, help="Plan pool directory (preferred)")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio by template")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--gamma", type=float, default=0.3, help="Penalty loss weight")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lambda_reg", type=float, default=0.1, help="Regression loss weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--check_actual_rows", action="store_true", default=True,
                        help="Require Actual Rows for supervision when extracting from plan_dir")
    parser.add_argument("--no_check_actual_rows", dest="check_actual_rows", action="store_false",
                        help="Disable Actual Rows requirement when extracting from plan_dir")
    parser.add_argument("--dump_extracted_jsonl", type=str, default=None,
                        help="Optional path to dump extracted/normalized node jsonl")
    args = parser.parse_args()

    if not args.plan_dir and not args.input:
        parser.error("one of --plan_dir or --input is required")

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Training Node Classification Model (Unified Pipeline)")
    print("=" * 60)

    # Step 1: Load/extract nodes
    print("\n[Step 1] Loading/extracting nodes...")
    extraction_stats = {}
    if args.plan_dir:
        nodes, extraction_stats = extract_nodes_from_pool(
            args.plan_dir,
            check_actual_rows=args.check_actual_rows,
        )
        print(f"  plan_dir: {args.plan_dir}")
        print(f"  total plan candidates: {extraction_stats.get('total_candidates', 0)}")
        print(f"  selected after hint dedupe: {extraction_stats.get('selected_after_hint_dedupe', 0)}")
        print(f"  dedup skipped: {extraction_stats.get('dedup_skipped', 0)}")
        print(f"  files missing actual rows: {extraction_stats.get('files_missing_actual', 0)}")
        print(f"  extracted nodes: {extraction_stats.get('nodes_extracted', 0)}")
    else:
        nodes = load_nodes_from_jsonl(args.input)

    if len(nodes) == 0:
        print("Error: No nodes available after extraction/loading.")
        return

    if args.dump_extracted_jsonl:
        dump_path = Path(args.dump_extracted_jsonl)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as f:
            for n in nodes:
                f.write(json.dumps(n, ensure_ascii=False) + "\n")
        print(f"  dumped extracted nodes to: {dump_path}")

    # Step 2: Split by template_id
    print(f"\n[Step 2] Splitting by template_id (train_ratio={args.train_ratio})...")
    train_nodes, valid_nodes, train_tids, valid_tids = split_nodes_by_template(
        nodes,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    _print_split_report(train_nodes, valid_nodes, train_tids, valid_tids)

    if len(train_nodes) == 0 or len(valid_nodes) == 0:
        print("Error: train/valid split produced empty split.")
        return

    # Step 3: Build table vocab from training templates only
    print("\n[Step 3] Building table vocabulary from training split...")
    table_to_idx = _build_table_vocab_from_nodes(train_nodes)
    print(f"  table vocab size (train-only): {len(table_to_idx)}")

    # Step 4: Compute train-only normalization stats
    print("\n[Step 4] Computing normalization stats from training split...")
    norm_stats = compute_norm_stats(train_nodes)

    # Step 5: Encode
    print("\n[Step 5] Encoding train/valid with unified structured features...")
    X_train, y_train, qerrors_train, _ = encode_nodes(train_nodes, table_to_idx, norm_stats)
    X_valid, y_valid, qerrors_valid, _ = encode_nodes(valid_nodes, table_to_idx, norm_stats)

    if X_train is None or X_valid is None:
        print("Error: encoding produced empty tensors.")
        return

    q_train = np.log1p(qerrors_train)
    q_valid = np.log1p(qerrors_valid)

    print(f"  Train encoded: {X_train.shape[0]} x {X_train.shape[1]}")
    print(f"  Valid encoded: {X_valid.shape[0]} x {X_valid.shape[1]}")

    # Step 6: Save artifacts for online prediction
    print("\n[Step 6] Saving encoding/training artifacts...")
    config = {
        "model_name": "structured_table_features_v1",
        "feature_dim": int(X_train.shape[1]),
        "table_vocab_size": len(table_to_idx),
        "table_vocab": sorted(table_to_idx.keys(), key=lambda x: table_to_idx[x]),
        "norm_stats": [float(x) for x in norm_stats],
        "split_mode": "template_id",
        "train_ratio": args.train_ratio,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(args.output_dir, "norm_stats.npy"), np.array(norm_stats, dtype=np.float32))

    # Step 7: Dataloaders
    print("\n[Step 7] Creating dataloaders...")
    train_ds = SimpleDataset(X_train, y_train, q_train)
    valid_ds = SimpleDataset(X_valid, y_valid, q_valid)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # Step 8: Model
    print("\n[Step 8] Creating model...")
    input_dim = int(X_train.shape[1])
    model = NodeEstimatorNet(
        input_dim=input_dim,
        hidden_dim=128,
        shared_dim=128,
        n_buckets=NUM_BUCKETS,
        dropout=0.2,
    ).to(args.device)
    print(f"  Model: NodeEstimatorNet(input_dim={input_dim}, hidden_dim=128, shared_dim=128)")

    # Step 9: Optimizer
    print("\n[Step 9] Creating optimizer...")
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=args.amp)

    # Step 10: Training
    print("\n[Step 10] Training...")
    print("=" * 60)
    best_val_f1 = -1.0

    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, opt, scaler, args.device, args)
        val_loss, val_acc, val_f1, val_lite_acc, val_lite_recall = eval_epoch(model, valid_loader, args.device, args)

        if epoch % 10 == 0 or (epoch + 1) == args.epochs:
            print(f"\nEpoch {epoch}:")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}, f1={train_f1:.4f}")
            print(f"  Valid: loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}")
            print(f"         lite_acc={val_lite_acc:.4f}, lite_recall={val_lite_recall:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            with open(os.path.join(args.output_dir, "model_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "input_dim": input_dim,
                        "hidden_dim": 128,
                        "shared_dim": 128,
                        "n_buckets": NUM_BUCKETS,
                        "dropout": 0.2,
                        "best_val_f1": float(best_val_f1),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    # Final summary
    print("\n[Step 11] Saving results...")
    print(f"  Best validation F1: {best_val_f1:.4f}")
    print(f"  Model saved to: {os.path.join(args.output_dir, 'best_model.pt')}")
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
