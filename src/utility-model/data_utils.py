#!/usr/bin/env python3
"""Data loading and feature encoding for utility-model v1."""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from plan_node.embedding import TextEncoder
from plan_node.encoder import (
    JOIN_NODE_TYPES,
    NODE_TYPES,
    SCAN_NODE_TYPES,
    TableRowsProvider,
    build_tree_from_plan_json,
)
from plan_node.predicate import extract_predicates_from_node

OPT_VEC_DIM = 4
PRED_EMBED_DIM = 384
PRED_PCA_DIM = 16

JOIN_HINT_HEADS = ("NESTLOOP", "HASHJOIN", "MERGEJOIN")
SCAN_HINT_HEADS = ("SEQSCAN", "INDEXSCAN", "INDEXONLYSCAN")


BAO_WRAPPER_TYPES = {
    "Sort",
    "Aggregate",
    "Materialize",
    "Gather",
    "Gather Merge",
    "Limit",
    "Result",
    "Unique",
    "Memoize",
}

BAO_OP_VOCAB = [
    "Nested Loop",
    "Hash Join",
    "Merge Join",
    "Seq Scan",
    "Index Scan",
    "Index Only Scan",
    "Bitmap Heap Scan",
    "Bitmap Index Scan",
    "CTE Scan",
    "Subquery Scan",
    "Append",
    "Merge Append",
    "Sort",
    "Aggregate",
    "Materialize",
    "Gather",
    "Other",
]


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split_by_target(rows: List[Dict], valid_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    def template_of(target_id: str) -> str:
        m = re.match(r"^(\d+)", target_id or "")
        return m.group(1) if m else str(target_id)

    targets = sorted({str(r["target_id"]) for r in rows})
    rnd = random.Random(seed)
    rnd.shuffle(targets)

    template_counts: Dict[str, int] = defaultdict(int)
    for t in targets:
        template_counts[template_of(t)] += 1
    template_valid_used: Dict[str, int] = {k: 0 for k in template_counts}

    desired_valid_n = int(len(targets) * valid_ratio)
    valid_targets: List[str] = []
    for t in targets:
        if len(valid_targets) >= desired_valid_n:
            break
        k = template_of(t)
        max_valid_for_template = max(template_counts[k] - 1, 0)
        if template_valid_used[k] < max_valid_for_template:
            valid_targets.append(t)
            template_valid_used[k] += 1

    valid_target_set = set(valid_targets)
    train = [r for r in rows if str(r["target_id"]) not in valid_target_set]
    valid = [r for r in rows if str(r["target_id"]) in valid_target_set]
    return train, valid


def _extract_template_num(name: str) -> str:
    m = re.match(r"^(\d+)", str(name or ""))
    return m.group(1) if m else ""


def _extract_hint_bodies(hint_str: str) -> List[str]:
    if not hint_str:
        return []
    matches = re.findall(r"/\*\+\s*(.*?)\s*\*/", hint_str, flags=re.DOTALL)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    text = hint_str.strip()
    return [text] if text else []


def _split_tokens(body: str) -> List[str]:
    tokens: List[str] = []
    cur: List[str] = []
    depth = 0
    for ch in body:
        if ch == "(":
            depth += 1
            cur.append(ch)
            continue
        if ch == ")":
            if depth > 0:
                depth -= 1
            cur.append(ch)
            continue
        if ch.isspace() and depth == 0:
            tok = "".join(cur).strip()
            if tok:
                tokens.append(tok)
            cur = []
            continue
        cur.append(ch)
    tok = "".join(cur).strip()
    if tok:
        tokens.append(tok)
    return tokens


def _head(token: str) -> str:
    token = token.strip()
    if "(" in token:
        token = token.split("(", 1)[0]
    return token.strip().upper()


def resolve_source_best_hint(row: Dict) -> str:
    source_plan_path = str(row.get("source_plan_json", "")).strip()
    if not source_plan_path:
        return "/*+ */"

    p = os.path.abspath(source_plan_path)
    source_dir = os.path.dirname(os.path.dirname(p))
    final_path = os.path.join(source_dir, "final_combined_hint.txt")
    suggest_path = os.path.join(source_dir, "suggest_hint.txt")

    for path in (final_path, suggest_path):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    return text
            except OSError:
                continue
    return "/*+ */"


def encode_hint_opt_vec(hint: str) -> np.ndarray:
    """4-bit opt vec:
    [has_set, has_leading, has_join_hint, has_scan_hint]
    """
    has_set = 0.0
    has_leading = 0.0
    has_join = 0.0
    has_scan = 0.0

    for body in _extract_hint_bodies(hint):
        toks = _split_tokens(body)
        for tok in toks:
            h = _head(tok)
            if h == "SET":
                has_set = 1.0
            elif h == "LEADING":
                has_leading = 1.0
            elif h in JOIN_HINT_HEADS:
                has_join = 1.0
            elif h in SCAN_HINT_HEADS:
                has_scan = 1.0

    return np.asarray([has_set, has_leading, has_join, has_scan], dtype=np.float32)


class PredicatePCAProjector:
    def __init__(self, out_dim: int = PRED_PCA_DIM):
        self.out_dim = int(out_dim)
        self.mean: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None

    def fit(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != PRED_EMBED_DIM:
            raise ValueError(f"expected [N,{PRED_EMBED_DIM}], got {embeddings.shape}")
        if embeddings.shape[0] == 0:
            self.mean = np.zeros(PRED_EMBED_DIM, dtype=np.float32)
            self.components = np.zeros((self.out_dim, PRED_EMBED_DIM), dtype=np.float32)
            return

        x = embeddings.astype(np.float32)
        self.mean = x.mean(axis=0)
        x_center = x - self.mean

        try:
            _, _, vt = np.linalg.svd(x_center, full_matrices=False)
            k_eff = min(self.out_dim, vt.shape[0], vt.shape[1])
            comps = np.zeros((self.out_dim, PRED_EMBED_DIM), dtype=np.float32)
            if k_eff > 0:
                comps[:k_eff, :] = vt[:k_eff, :]
            self.components = comps
        except np.linalg.LinAlgError:
            self.components = np.zeros((self.out_dim, PRED_EMBED_DIM), dtype=np.float32)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA projector not fitted")
        x = embeddings.astype(np.float32)
        if x.ndim == 1:
            x = x[None, :]
        x_center = x - self.mean
        out = x_center @ self.components.T
        return out.astype(np.float32)

    def save(self, path: str) -> None:
        if self.mean is None or self.components is None:
            raise RuntimeError("PCA projector not fitted")
        np.savez(path, mean=self.mean, components=self.components)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        obj = np.load(path)
        self.mean = obj["mean"].astype(np.float32)
        self.components = obj["components"].astype(np.float32)
        return True


@dataclass
class BaoHybridArtifacts:
    mean: List[float]
    std: List[float]
    op_vocab: List[str]


class BasePlanTreeEncoder:
    feature_dim: int

    def encode_path(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class CurrentPlanTreeEncoder(BasePlanTreeEncoder):
    """Current encoder: base tree feature + optional predicate PCA feature insertion."""

    def __init__(
        self,
        artifacts_dir: str,
        predicate_fit_dir: str,
        model_name: str,
        text_device: Optional[str],
        db_name: Optional[str],
        use_predicate_pca: bool,
    ):
        self.artifacts_dir = artifacts_dir
        norm_stats_path = os.path.join(artifacts_dir, "norm_stats.npy")
        if not os.path.isdir(artifacts_dir):
            raise FileNotFoundError(
                f"artifacts_dir not found: {artifacts_dir}. "
                f"Please run encoder first (mode=tree) to generate artifacts."
            )
        if not os.path.exists(norm_stats_path):
            raise FileNotFoundError(
                f"missing file: {norm_stats_path}. "
                f"Please regenerate artifacts with encoder.py --mode tree."
            )
        self.norm_stats = tuple(np.load(norm_stats_path))

        cfg_path = os.path.join(artifacts_dir, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        vocab = cfg.get("table_vocab", []) or []
        self.table_to_idx = {str(t): i for i, t in enumerate(vocab)}

        self.base_feature_dim = int(cfg.get("feature_dim", 6 + len(self.table_to_idx) + 6))
        self.insert_pos = len(NODE_TYPES) + len(self.table_to_idx)
        self.use_predicate_pca = bool(use_predicate_pca)
        self.feature_dim = self.base_feature_dim + (PRED_PCA_DIM if self.use_predicate_pca else 0)

        self.table_row_provider = TableRowsProvider(dbname=(db_name or None))
        self.text_encoder = TextEncoder(model_name=model_name, device=text_device)

        self.tree_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.pred_text_cache: Dict[str, np.ndarray] = {}

        self.pca = PredicatePCAProjector(out_dim=PRED_PCA_DIM)
        self.pca_cache_path = os.path.join(artifacts_dir, "predicate_pca16.npz")

        if self.use_predicate_pca:
            loaded = self.pca.load(self.pca_cache_path)
            if not loaded:
                print("[predicate-pca] fitting PCA16 from plan directory...")
                emb = self._collect_predicate_embeddings(predicate_fit_dir)
                self.pca.fit(emb)
                self.pca.save(self.pca_cache_path)
                print(f"[predicate-pca] fitted and saved: {self.pca_cache_path}")
            else:
                print(f"[predicate-pca] loaded: {self.pca_cache_path}")

    def _predicate_text_to_embed(self, text: str) -> np.ndarray:
        key = text.strip()
        cached = self.pred_text_cache.get(key)
        if cached is not None:
            return cached
        if not key:
            out = np.zeros(PRED_EMBED_DIM, dtype=np.float32)
        else:
            out = self.text_encoder.encode_text(key, prefix="predicate:")
            out = np.asarray(out, dtype=np.float32)
        self.pred_text_cache[key] = out
        return out

    def _collect_predicate_embeddings(self, plan_dir: str) -> np.ndarray:
        plan_files: List[str] = []
        for root, _, files in os.walk(plan_dir):
            if "plan.json" in files:
                plan_files.append(os.path.join(root, "plan.json"))
        plan_files = sorted(plan_files)

        all_emb: List[np.ndarray] = []
        for p in plan_files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    plan_json = json.load(f)
                pred_texts = self._collect_tree_predicate_texts(plan_json)
                for t in pred_texts:
                    all_emb.append(self._predicate_text_to_embed(t))
            except Exception:
                continue

        if not all_emb:
            return np.zeros((1, PRED_EMBED_DIM), dtype=np.float32)
        return np.stack(all_emb, axis=0).astype(np.float32)

    def _collect_tree_predicate_texts(self, plan_json: Dict) -> List[str]:
        pred_texts: List[str] = []
        children: List[List[int]] = []

        def add_node(text: str) -> int:
            idx = len(pred_texts)
            pred_texts.append(text)
            children.append([-1, -1])
            return idx

        def visit(node: Dict) -> List[int]:
            node_type = node.get("Node Type", "")
            plans = node.get("Plans", []) or []

            child_roots: List[int] = []
            for ch in plans:
                child_roots.extend(visit(ch))

            if node_type in JOIN_NODE_TYPES or node_type in SCAN_NODE_TYPES:
                preds = extract_predicates_from_node(node)
                if preds:
                    text = " AND ".join(sorted(set([str(x).strip() for x in preds if str(x).strip()])))
                else:
                    text = ""
                my_idx = add_node(text)
                left_id = child_roots[0] if len(child_roots) >= 1 else -1
                right_id = child_roots[1] if len(child_roots) >= 2 else -1
                children[my_idx] = [left_id, right_id]
                return [my_idx]

            return child_roots

        root_node = plan_json.get("Plan", plan_json)
        roots = visit(root_node)
        if not roots:
            return []

        root_idx = roots[0]
        if root_idx != 0:
            order = [root_idx] + [i for i in range(len(pred_texts)) if i != root_idx]
            pred_texts = [pred_texts[i] for i in order]
        return pred_texts

    def encode_path(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        key = os.path.abspath(path)
        cached = self.tree_cache.get(key)
        if cached is not None:
            return cached

        with open(key, "r", encoding="utf-8") as f:
            plan_json = json.load(f)

        tree = build_tree_from_plan_json(
            plan_json,
            encoder=None,
            norm_stats=self.norm_stats,
            table_to_idx=self.table_to_idx,
            table_row_provider=self.table_row_provider,
        )
        if tree is None:
            raise ValueError(f"Tree build failed for plan: {key}")

        base_feat = np.asarray(tree["features"], dtype=np.float32)
        children = np.asarray(tree["children"], dtype=np.int64)

        if not self.use_predicate_pca:
            feat_t = torch.tensor(base_feat, dtype=torch.float32)
            child_t = torch.tensor(children, dtype=torch.long)
            self.tree_cache[key] = (feat_t, child_t)
            return feat_t, child_t

        pred_texts = self._collect_tree_predicate_texts(plan_json)
        if len(pred_texts) != base_feat.shape[0]:
            pred_pca = np.zeros((base_feat.shape[0], PRED_PCA_DIM), dtype=np.float32)
        else:
            pred_emb = np.stack([self._predicate_text_to_embed(t) for t in pred_texts], axis=0)
            pred_pca = self.pca.transform(pred_emb)

        left = base_feat[:, : self.insert_pos]
        right = base_feat[:, self.insert_pos :]
        feat = np.concatenate([left, pred_pca, right], axis=1).astype(np.float32)

        feat_t = torch.tensor(feat, dtype=torch.float32)
        child_t = torch.tensor(children, dtype=torch.long)
        self.tree_cache[key] = (feat_t, child_t)
        return feat_t, child_t


class BaoHybridPlanTreeEncoder(BasePlanTreeEncoder):
    """Bao-style/hybrid tree encoder with train-split fitted numeric normalization."""

    def __init__(
        self,
        artifacts_path: str,
        use_predicate_pca: bool,
        model_name: str,
        text_device: Optional[str],
    ):
        self.artifacts_path = artifacts_path
        self.use_predicate_pca = bool(use_predicate_pca)
        self.tree_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.pred_text_cache: Dict[str, np.ndarray] = {}

        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"bao_hybrid encoder artifacts not found: {artifacts_path}")
        with open(artifacts_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        art = BaoHybridArtifacts(
            mean=[float(x) for x in raw.get("mean", [])],
            std=[float(x) for x in raw.get("std", [])],
            op_vocab=[str(x) for x in raw.get("op_vocab", BAO_OP_VOCAB)],
        )
        self.op_vocab = art.op_vocab
        self.op_to_idx = {k: i for i, k in enumerate(self.op_vocab)}

        self.numeric_dim = len(art.mean)
        self.mean = np.asarray(art.mean, dtype=np.float32)
        self.std = np.asarray([max(float(v), 1e-6) for v in art.std], dtype=np.float32)

        self.text_encoder = TextEncoder(model_name=model_name, device=text_device)
        self.pca = PredicatePCAProjector(out_dim=PRED_PCA_DIM)
        self.pca_cache_path = f"{artifacts_path}.predicate_pca16.npz"
        if self.use_predicate_pca:
            if not self.pca.load(self.pca_cache_path):
                raise FileNotFoundError(
                    f"predicate PCA requested but cache missing: {self.pca_cache_path}. "
                    "Please fit with use_predicate_pca=True during training."
                )
        self.feature_dim = len(self.op_vocab) + self.numeric_dim + (PRED_PCA_DIM if self.use_predicate_pca else 0)

    @staticmethod
    def _log1p_nonneg(v: float) -> float:
        return math.log1p(max(float(v), 0.0))

    @staticmethod
    def _flatten_plan_nodes(root: Dict) -> List[Dict]:
        out: List[Dict] = []

        def visit(n: Dict):
            out.append(n)
            for c in (n.get("Plans") or []):
                visit(c)

        visit(root)
        return out

    @staticmethod
    def _safe_float(d: Dict, key: str, default: float = 0.0) -> float:
        try:
            return float(d.get(key, default))
        except Exception:
            return float(default)

    def _canonical_op(self, node: Dict) -> str:
        op = str(node.get("Node Type", "Other"))
        if op in self.op_to_idx:
            return op
        if op in BAO_WRAPPER_TYPES:
            return op if op in self.op_to_idx else "Other"
        if "Bitmap" in op:
            if "Heap" in op:
                return "Bitmap Heap Scan"
            if "Index" in op:
                return "Bitmap Index Scan"
        return "Other"

    def _is_join(self, op: str) -> float:
        return 1.0 if "Join" in op or op == "Nested Loop" else 0.0

    def _is_scan(self, op: str) -> float:
        return 1.0 if "Scan" in op else 0.0

    def _is_bitmap(self, op: str) -> float:
        return 1.0 if "Bitmap" in op else 0.0

    def _is_index(self, op: str) -> float:
        return 1.0 if "Index" in op else 0.0

    def _subtree_table_count(self, node: Dict) -> float:
        rels = set()

        def visit(n: Dict):
            rn = str(n.get("Relation Name", "") or "").strip()
            if rn:
                rels.add(rn)
            for c in (n.get("Plans") or []):
                visit(c)

        visit(node)
        return float(len(rels))

    def _node_numeric(self, node: Dict, depth: int) -> np.ndarray:
        op = self._canonical_op(node)
        startup = self._safe_float(node, "Startup Cost", 0.0)
        total = self._safe_float(node, "Total Cost", startup)
        rows = self._safe_float(node, "Plan Rows", 0.0)
        width = self._safe_float(node, "Plan Width", 0.0)

        child_cost_sum = 0.0
        child_rows_sum = 0.0
        for c in (node.get("Plans") or []):
            child_cost_sum += self._safe_float(c, "Total Cost", 0.0)
            child_rows_sum += self._safe_float(c, "Plan Rows", 0.0)

        self_cost = max(total - child_cost_sum, 0.0)
        cost_ratio = total / max(child_cost_sum, 1.0)
        row_ratio = rows / max(child_rows_sum, 1.0)

        vals = np.asarray(
            [
                self._log1p_nonneg(total),
                self._log1p_nonneg(rows),
                self._log1p_nonneg(width),
                self._log1p_nonneg(startup),
                self._log1p_nonneg(self_cost),
                self._log1p_nonneg(float(depth)),
                self._log1p_nonneg(cost_ratio),
                self._log1p_nonneg(row_ratio),
                self._log1p_nonneg(self._subtree_table_count(node)),
                self._is_join(op),
                self._is_scan(op),
                self._is_bitmap(op),
                self._is_index(op),
            ],
            dtype=np.float32,
        )
        return vals

    def _predicate_text_for_node(self, node: Dict) -> str:
        preds = extract_predicates_from_node(node)
        if preds:
            return " AND ".join(sorted(set([str(x).strip() for x in preds if str(x).strip()])))
        return ""

    def _predicate_text_to_embed(self, text: str) -> np.ndarray:
        key = text.strip()
        cached = self.pred_text_cache.get(key)
        if cached is not None:
            return cached
        if not key:
            out = np.zeros(PRED_EMBED_DIM, dtype=np.float32)
        else:
            out = self.text_encoder.encode_text(key, prefix="predicate:")
            out = np.asarray(out, dtype=np.float32)
        self.pred_text_cache[key] = out
        return out

    def _build_binary_tree(self, plan_json: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        root = plan_json.get("Plan", plan_json)
        feats: List[np.ndarray] = []
        children: List[List[int]] = []
        pred_texts: List[str] = []

        def add_node(vec: np.ndarray, pred_text: str) -> int:
            idx = len(feats)
            feats.append(vec)
            children.append([-1, -1])
            pred_texts.append(pred_text)
            return idx

        def visit(node: Dict, depth: int) -> int:
            op = self._canonical_op(node)
            onehot = np.zeros(len(self.op_vocab), dtype=np.float32)
            onehot[self.op_to_idx.get(op, self.op_to_idx.get("Other", len(self.op_vocab) - 1))] = 1.0
            num = self._node_numeric(node, depth)
            vec = np.concatenate([onehot, num], axis=0)
            my_idx = add_node(vec, self._predicate_text_for_node(node))

            plan_children = node.get("Plans") or []
            if plan_children:
                child_ids = [visit(c, depth + 1) for c in plan_children]
                children[my_idx][0] = child_ids[0] if len(child_ids) > 0 else -1
                children[my_idx][1] = child_ids[1] if len(child_ids) > 1 else -1
            return my_idx

        root_idx = visit(root, 0)

        if root_idx != 0:
            order = [root_idx] + [i for i in range(len(feats)) if i != root_idx]
            remap = {old: new for new, old in enumerate(order)}
            feats = [feats[i] for i in order]
            pred_texts = [pred_texts[i] for i in order]
            new_children: List[List[int]] = []
            for old_i in order:
                l_old, r_old = children[old_i]
                l_new = remap[l_old] if l_old != -1 else -1
                r_new = remap[r_old] if r_old != -1 else -1
                new_children.append([l_new, r_new])
            children = new_children

        f = np.stack(feats, axis=0).astype(np.float32)
        c = np.asarray(children, dtype=np.int64)
        return f, c, pred_texts

    def encode_path(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        key = os.path.abspath(path)
        cached = self.tree_cache.get(key)
        if cached is not None:
            return cached

        with open(key, "r", encoding="utf-8") as f:
            plan_json = json.load(f)

        feat, child, pred_texts = self._build_binary_tree(plan_json)

        left = feat[:, : len(self.op_vocab)]
        num = feat[:, len(self.op_vocab) :]
        num = (num - self.mean[None, :]) / self.std[None, :]
        feat = np.concatenate([left, num], axis=1)

        if self.use_predicate_pca:
            if len(pred_texts) != feat.shape[0]:
                pred_pca = np.zeros((feat.shape[0], PRED_PCA_DIM), dtype=np.float32)
            else:
                pred_emb = np.stack([self._predicate_text_to_embed(t) for t in pred_texts], axis=0)
                pred_pca = self.pca.transform(pred_emb)
            feat = np.concatenate([feat, pred_pca], axis=1).astype(np.float32)

        feat_t = torch.tensor(feat, dtype=torch.float32)
        child_t = torch.tensor(child, dtype=torch.long)
        self.tree_cache[key] = (feat_t, child_t)
        return feat_t, child_t


def fit_bao_hybrid_artifacts_from_rows(
    rows: Sequence[Dict],
    output_path: str,
    use_predicate_pca: bool,
    model_name: str,
    text_device: Optional[str],
) -> str:
    """Fit bao_hybrid numeric normalization using train split source/target plan json only."""
    plan_paths: List[str] = []
    for r in rows:
        for k in ("source_plan_json", "target_plan_json"):
            p = str(r.get(k, "")).strip()
            if p:
                plan_paths.append(os.path.abspath(p))

    plan_paths = sorted(set(plan_paths))
    if not plan_paths:
        raise RuntimeError("No plan json paths found in train rows for bao_hybrid artifact fitting")

    tmp_encoder = BaoHybridPlanTreeEncoder.__new__(BaoHybridPlanTreeEncoder)
    tmp_encoder.op_vocab = list(BAO_OP_VOCAB)
    tmp_encoder.op_to_idx = {k: i for i, k in enumerate(tmp_encoder.op_vocab)}

    numeric_rows: List[np.ndarray] = []
    pred_texts_all: List[str] = []

    for p in plan_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                plan_json = json.load(f)
            feat, _, pred_texts = tmp_encoder._build_binary_tree(plan_json)  # type: ignore[attr-defined]
            num = feat[:, len(tmp_encoder.op_vocab) :]
            if num.size > 0:
                numeric_rows.append(num)
            pred_texts_all.extend(pred_texts)
        except Exception:
            continue

    if not numeric_rows:
        raise RuntimeError("Failed to collect any numeric rows while fitting bao_hybrid artifacts")

    all_num = np.concatenate(numeric_rows, axis=0).astype(np.float32)
    mean = all_num.mean(axis=0)
    std = all_num.std(axis=0)
    std = np.maximum(std, 1e-6)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "op_vocab": list(BAO_OP_VOCAB),
                "numeric_dim": int(all_num.shape[1]),
                "num_plan_files": len(plan_paths),
            },
            f,
            indent=2,
        )

    if use_predicate_pca:
        text_encoder = TextEncoder(model_name=model_name, device=text_device)
        projector = PredicatePCAProjector(out_dim=PRED_PCA_DIM)
        emb: List[np.ndarray] = []
        for t in pred_texts_all:
            tt = t.strip()
            if not tt:
                emb.append(np.zeros(PRED_EMBED_DIM, dtype=np.float32))
            else:
                e = text_encoder.encode_text(tt, prefix="predicate:")
                emb.append(np.asarray(e, dtype=np.float32))
        if not emb:
            emb = [np.zeros(PRED_EMBED_DIM, dtype=np.float32)]
        projector.fit(np.stack(emb, axis=0).astype(np.float32))
        projector.save(f"{output_path}.predicate_pca16.npz")

    return output_path


class PlanTreeEncoder(BasePlanTreeEncoder):
    """Factory wrapper that exposes unified interface for current/bao_hybrid encoders."""

    def __init__(
        self,
        artifacts_dir: str,
        predicate_fit_dir: str = "outputs/demo_pool",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_device: Optional[str] = None,
        db_name: Optional[str] = None,
        encoder_mode: str = "current",
        encoder_artifacts_dir: str = "",
        use_predicate_pca: bool = True,
    ):
        mode = str(encoder_mode).strip().lower()
        if mode not in {"current", "bao_hybrid"}:
            raise ValueError(f"Unsupported encoder_mode: {encoder_mode}")

        self.encoder_mode = mode
        self.use_predicate_pca = bool(use_predicate_pca)
        self.encoder_artifacts_dir = encoder_artifacts_dir

        if mode == "current":
            self.impl = CurrentPlanTreeEncoder(
                artifacts_dir=artifacts_dir,
                predicate_fit_dir=predicate_fit_dir,
                model_name=model_name,
                text_device=text_device,
                db_name=db_name,
                use_predicate_pca=self.use_predicate_pca,
            )
        else:
            art_path = encoder_artifacts_dir.strip() if encoder_artifacts_dir else ""
            if not art_path:
                art_path = os.path.join(artifacts_dir, "encoder_artifacts.json")
            self.impl = BaoHybridPlanTreeEncoder(
                artifacts_path=art_path,
                use_predicate_pca=self.use_predicate_pca,
                model_name=model_name,
                text_device=text_device,
            )

        self.feature_dim = int(self.impl.feature_dim)

    def encode_path(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.impl.encode_path(path)


def encode_row_opt_vec(row: Dict) -> torch.Tensor:
    hint = resolve_source_best_hint(row)
    return torch.tensor(encode_hint_opt_vec(hint), dtype=torch.float32)


def encode_row_reuse(row: Dict) -> torch.Tensor:
    source_id = str(row.get("source_id", ""))
    target_id = str(row.get("target_id", ""))
    same = 1.0 if (_extract_template_num(source_id) and _extract_template_num(source_id) == _extract_template_num(target_id)) else 0.0
    return torch.tensor([same], dtype=torch.float32)
