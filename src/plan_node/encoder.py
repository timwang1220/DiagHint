# encoder.py
# Rewritten encoder using sentence-transformers for text embeddings
# Reads plan.json files directly and produces encoded features.
#
# Feature layout:
#   onehot(node_type) 6
#   table_name_emb    384 (sentence-transformer)
#   column_embs       384 (sentence-transformer, averaged)
#   predicate_embs    384 (sentence-transformer)
#   num_feats         6  (normalized self_cost, plan_rows, total_cost, plan_width, depth; raw selectivity)
#
# (legacy) Total dim was 1163 before structured-table feature refactor.
import os
import sys
import json
import math
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any, Union, Set
from pathlib import Path

import psycopg2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plan_node.utils import qerror_and_bucket, bucket2id
from plan_node.embedding import TextEncoder
from plan_node.predicate import extract_predicates_from_node, parse_predicate_elements
from config import Config
import re


# ---------------------------
# Helper functions for similarity calculation
# ---------------------------

def extract_qerror_vector_from_plan(plan_json: Dict) -> List[float]:
    """Extract qerror vector from plan for similarity calculation."""
    qerrors = []

    def extract_recursive(node: Dict, depth: int = 0):
        if "Plan Rows" in node and "Actual Rows" in node:
            est = float(node.get("Plan Rows", 0))
            act = float(node.get("Actual Rows", 0))
            if est > 0 and act > 0:
                qerror = max(est / act, act / est)
                qerrors.append(math.log(qerror + 0.001))
        for child in node.get("Plans", []):
            extract_recursive(child, depth + 1)

    # Handle wrapped plan format
    if "Plan" in plan_json:
        extract_recursive(plan_json["Plan"])
    else:
        extract_recursive(plan_json)

    return qerrors


def extract_leading_sequence_from_plan(plan_json: Dict) -> List[str]:
    """Extract leading sequence (table join order) from plan."""
    sequence = []

    def extract_recursive(node: Dict):
        node_type = node.get("Node Type", "")
        is_join = node_type in ["Nested Loop", "Hash Join", "Merge Join"]
        is_scan = "Scan" in node_type

        if is_scan:
            alias = node.get("Alias") or node.get("Relation Name", "")
            if alias:
                sequence.append(alias)

        for child in node.get("Plans", []):
            extract_recursive(child)

    # Handle wrapped plan format
    if "Plan" in plan_json:
        extract_recursive(plan_json["Plan"])
    else:
        extract_recursive(plan_json)

    return sequence


def extract_join_type_from_plan(plan_json: Dict) -> List[List]:
    """Extract join type information from plan."""
    join_types = []

    def extract_recursive(node: Dict, depth: int = 0):
        node_type = node.get("Node Type", "")
        is_join = node_type in ["Nested Loop", "Hash Join", "Merge Join"]

        if is_join:
            plans = node.get("Plans", [])
            if len(plans) >= 2:
                # Extract left and right tables
                left_tables = _extract_tables_from_subtree(plans[0])
                right_tables = _extract_tables_from_subtree(plans[1])
                join_types.append([node_type, left_tables, right_tables])

        for child in node.get("Plans", []):
            extract_recursive(child, depth + 1)

    def _extract_tables_from_subtree(node: Dict) -> List[str]:
        """Extract table aliases from a subtree."""
        tables = []

        def _extract(n: Dict):
            ntype = n.get("Node Type", "")
            if "Scan" in ntype:
                alias = n.get("Alias") or n.get("Relation Name", "")
                if alias:
                    tables.append(alias)
            for child in n.get("Plans", []):
                _extract(child)

        _extract(node)
        return sorted(tables)

    # Handle wrapped plan format
    if "Plan" in plan_json:
        extract_recursive(plan_json["Plan"])
    else:
        extract_recursive(plan_json)

    return join_types


def extract_scan_type_from_plan(plan_json: Dict) -> List[List]:
    """Extract scan type information from plan."""
    scan_types = []

    def extract_recursive(node: Dict, depth: int = 0):
        node_type = node.get("Node Type", "")
        is_scan = "Scan" in node_type

        if is_scan:
            table_name = node.get("Relation Name", "")
            if table_name:
                scan_types.append([node_type, table_name])

        for child in node.get("Plans", []):
            extract_recursive(child, depth + 1)

    # Handle wrapped plan format
    if "Plan" in plan_json:
        extract_recursive(plan_json["Plan"])
    else:
        extract_recursive(plan_json)

    return scan_types


def hint_to_list(hint: str) -> List[List[str]]:
    """
    将 hint 字符串转换为列表形式。
    去掉前后 /*+ 和 */，按空格分割，括号外为操作名，括号内为参数。
    """
    content = hint.strip()

    if content.startswith("/*+"):
        content = content[3:]
    if content.endswith("*/"):
        content = content[:-2]

    pattern = r"(\w+)\s*\(([^)]+)\)"
    matches = re.findall(pattern, content)

    result = []
    for op_name, args_str in matches:
        args = args_str.strip().split()
        sorted_args = sorted(args)
        result.append([op_name] + sorted_args)

    return result


def is_join_op(op_name: str) -> bool:
    return op_name in ["HashJoin", "NestedLoop", "MergeJoin"]


def is_scan_op(op_name: str) -> bool:
    return op_name in ["SeqScan", "IndexScan", "BitmapHeapScan", "BitmapIndexScan", "IndexOnlyScan"]


def permutation_embedding(l1: List[str], l2: List[str], normalize: bool = True) -> np.ndarray:
    """计算两个序列之间的排列嵌入。"""
    if not l1 or not l2:
        return np.zeros(6, dtype=np.float32)

    # 取相同的元素
    common = list(set(l1) & set(l2))
    if not common:
        return np.zeros(6, dtype=np.float32)

    l1_common = [x for x in l1 if x in common]
    l2_common = [x for x in l2 if x in common]

    if len(l1_common) != len(l2_common):
        l1_common = sorted(l1_common)
        l2_common = sorted(l2_common)

    n = len(l1_common)
    if n == 0:
        return np.zeros(6, dtype=np.float32)

    pos1 = {k: i for i, k in enumerate(l1_common)}
    pos2 = {k: i for i, k in enumerate(l2_common)}

    displacements = []
    before_counts = []
    after_counts = []

    for k in l1_common:
        d = pos2[k] - pos1[k]
        displacements.append(d)

        before = 0
        after = 0
        for j in l1_common:
            if j == k:
                continue
            if pos1[j] < pos1[k] and pos2[j] > pos2[k]:
                after += 1
            if pos1[j] > pos1[k] and pos2[j] < pos2[k]:
                before += 1

        before_counts.append(before)
        after_counts.append(after)

    displacements = np.array(displacements, dtype=np.float32)
    before_counts = np.array(before_counts, dtype=np.float32)
    after_counts = np.array(after_counts, dtype=np.float32)

    features = []
    for arr in [displacements, np.abs(displacements), before_counts, after_counts]:
        if len(arr) > 0:
            features.extend([arr.mean(), arr.max()])
        else:
            features.extend([0.0, 0.0])

    emb = np.array(features, dtype=np.float32)

    if normalize:
        norm = np.linalg.norm(emb) + 1e-8
        emb = emb / norm

    return emb


def hint_minus(hint1: str, leading1: List[str], join1: List, scan1: List,
               hint2: str, leading2: List[str], join2: List, scan2: List) -> List:
    """计算 hint1 到 hint2 的差向量。"""
    hint_list1 = hint_to_list(hint1)
    hint_list2 = hint_to_list(hint2)

    minus_hint = []
    for op_name, *args in hint_list2:
        if [op_name] + args not in hint_list1:
            minus_hint.append([op_name] + args)

    for op_name, *args in hint_list1:
        if [op_name] + args not in hint_list2:
            if op_name == "Leading":
                continue
            if is_join_op(op_name):
                join_method = op_name
                left_join = sorted(args[:-1])
                right_join = [args[-1]]
                for join_node in join2:
                    if len(join_node) >= 3:
                        node_left = sorted(join_node[1]) if isinstance(join_node[1], list) else []
                        node_right = [join_node[2]] if len(join_node) > 2 else []
                        if node_left == left_join and node_right == right_join:
                            clean_name = join_node[0].replace(" ", "")
                            if clean_name != join_method:
                                minus_hint.append([clean_name] + args)
                            break
            elif is_scan_op(op_name):
                scan_method = op_name
                scan_table = args[0] if args else ""
                for scan_node in scan2:
                    if len(scan_node) >= 2:
                        if scan_node[1] == scan_table:
                            clean_name = scan_node[0].replace(" ", "")
                            if clean_name != scan_method:
                                minus_hint.append([clean_name] + args)
                            break

    # 构建差向量
    minus_vector = [0, [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for op_name, *args in minus_hint:
        if op_name == 'Set':
            minus_vector[0] = 1

        # Leading 排列嵌入
        minus_vector[1] = permutation_embedding(leading1, leading2).tolist()

        if is_join_op(op_name):
            if op_name == "HashJoin":
                minus_vector[2][0] += 1
            elif op_name == "NestedLoop":
                minus_vector[2][1] += 1
            elif op_name == "MergeJoin":
                minus_vector[2][2] += 1
        elif is_scan_op(op_name):
            if op_name == "SeqScan":
                minus_vector[3][0] += 1
            elif op_name == "IndexScan":
                minus_vector[3][1] += 1
            elif op_name == "IndexOnlyScan":
                minus_vector[3][2] += 1

    return minus_vector


def calculate_to_best_distance(v1: List, v2: List) -> float:
    """计算到 best hint 的距离。"""
    d1 = abs(v1[0] - v2[0])
    v1_part2 = np.array(v1[1], dtype=float)
    v2_part2 = np.array(v2[1], dtype=float)
    d2 = np.linalg.norm(v1_part2 - v2_part2, ord=1)

    v1_part3 = np.array(v1[2], dtype=float)
    v2_part3 = np.array(v2[2], dtype=float)
    d3 = np.linalg.norm(v1_part3 - v2_part3, ord=1)

    v1_part4 = np.array(v1[3], dtype=float)
    v2_part4 = np.array(v2[3], dtype=float)
    d4 = np.linalg.norm(v1_part4 - v2_part4, ord=1)

    total_distance = (d1 + d2 + d3 + d4) / 4.0
    return total_distance


def calculate_qerror_similarity(v1: List[float], v2: List[float]) -> float:
    """计算 qerror 向量的余弦相似度。"""
    if not v1 or not v2:
        return 0.0

    max_len = max(len(v1), len(v2))
    vec1_pad = np.pad(v1, (0, max_len - len(v1)))
    vec2_pad = np.pad(v2, (0, max_len - len(v2)))

    norm1 = np.linalg.norm(vec1_pad)
    norm2 = np.linalg.norm(vec2_pad)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    qerror_sim = np.dot(vec1_pad, vec2_pad) / (norm1 * norm2)
    return qerror_sim


def compute_similarity(meta1: Dict, meta2: Dict, to_best_vectors: Dict) -> float:
    """计算两个元数据之间的相似度。"""
    # 获取各自的 to_best_vector
    key1 = (meta1["parent_id"], meta1["parent_index"])
    key2 = (meta2["parent_id"], meta2["parent_index"])

    v1 = to_best_vectors.get(key1)
    v2 = to_best_vectors.get(key2)

    if v1 is None or v2 is None:
        return 0.0

    # 计算 to_best 相似度
    to_best_dist = calculate_to_best_distance(v1, v2)

    # 计算 qerror 相似度
    qerror_sim = calculate_qerror_similarity(meta1["qerror_vector"], meta2["qerror_vector"])

    # 这里需要在全部 pairs 计算完后进行归一化
    # 先返回原始值，归一化在外部进行
    return to_best_dist, qerror_sim


# ---------------------------
# Configuration / constants
# ---------------------------
NODE_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Nested Loop", "Hash Join", "Merge Join"]
JOIN_NODE_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
SCAN_NODE_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan"]


def normalize_table_token(alias: str, relation_name: str) -> str:
    tok = (alias or "").strip()
    if tok:
        return tok
    return (relation_name or "").strip()


class TableRowsProvider:
    """Cached DB-backed table row estimator using pg_class.reltuples."""

    def __init__(
        self,
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
    ):
        self._conn = None
        self._cache: Dict[str, float] = {}
        self._db_cfg = {
            "dbname": dbname or Config.DB_DATABASE,
            "user": user or Config.DB_USER,
            "password": password or Config.DB_PASSWORD,
            "host": host or Config.DB_HOST,
            "port": port or Config.DB_PORT,
        }

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _connect(self):
        if self._conn is not None:
            return self._conn
        self._conn = psycopg2.connect(
            dbname=self._db_cfg["dbname"],
            user=self._db_cfg["user"],
            password=self._db_cfg["password"],
            host=self._db_cfg["host"],
            port=self._db_cfg["port"],
        )
        self._conn.autocommit = True
        return self._conn

    @staticmethod
    def _split_schema_table(relation_name: str) -> Tuple[Optional[str], str]:
        rel = (relation_name or "").strip().strip('"')
        if "." in rel:
            schema, table = rel.split(".", 1)
            return schema.strip('"'), table.strip('"')
        return None, rel

    def get_table_rows(self, relation_name: str) -> float:
        rel = (relation_name or "").strip()
        if not rel:
            raise RuntimeError("empty relation name when resolving table rows")
        if rel in self._cache:
            return self._cache[rel]

        schema, table = self._split_schema_table(rel)
        rows_val = None
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                if schema:
                    cur.execute(
                        """
                        SELECT c.reltuples::double precision
                        FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE c.relname = %s AND n.nspname = %s
                        LIMIT 1
                        """,
                        (table, schema),
                    )
                else:
                    cur.execute(
                        """
                        SELECT c.reltuples::double precision
                        FROM pg_class c
                        WHERE c.oid = to_regclass(%s)
                        LIMIT 1
                        """,
                        (table,),
                    )
                r = cur.fetchone()
                if r is not None and r[0] is not None:
                    rows_val = float(r[0])
        except Exception as e:
            raise RuntimeError(f"table row lookup failed for '{rel}': {repr(e)}") from e
        if rows_val is None or rows_val <= 0:
            raise RuntimeError(
                f"table row lookup returned non-positive/empty reltuples for '{rel}' (value={rows_val}). "
                "Run ANALYZE on this table/database and verify search_path resolves the intended relation."
            )
        self._cache[rel] = rows_val
        return rows_val


def collect_plan_files(plan_dir: str) -> List[str]:
    plan_files: List[str] = []
    for root, _, files in os.walk(plan_dir):
        if "plan.json" in files:
            plan_files.append(os.path.join(root, "plan.json"))
    return sorted(plan_files)


def build_table_vocab_from_plan_files(plan_files: List[str]) -> Dict[str, int]:
    tables: Set[str] = set()
    for plan_path in plan_files:
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
            root = plan_data["Plan"] if "Plan" in plan_data else plan_data

            def walk(node: Dict):
                nt = node.get("Node Type", "")
                if nt in SCAN_NODE_TYPES:
                    tok = normalize_table_token(
                        alias=str(node.get("Alias", "")),
                        relation_name=str(node.get("Relation Name", "")),
                    )
                    if tok:
                        tables.add(tok)
                for ch in node.get("Plans", []):
                    walk(ch)

            walk(root)
        except Exception:
            continue
    ordered = sorted(tables)
    return {t: i for i, t in enumerate(ordered)}


def safe_float(x) -> float:
    """Safely convert to float, return 0.0 on failure."""
    try:
        return float(x)
    except Exception:
        return 0.0


def log_transform(x: float) -> float:
    """Apply log transformation with small offset."""
    return math.log(max(x, 0.001) + 0.001)


# ---------------------------
# Plan node collection
# ---------------------------
def parse_plan_node(
    node: Dict,
    depth: int = 0,
    nodes_list: Optional[List[Dict]] = None,
    check_actual_rows: bool = True,
) -> Tuple[List[Dict], bool]:
    """
    Recursively parse a plan node tree and extract Join/Scan nodes.
    """
    if nodes_list is None:
        nodes_list = []

    has_actual_rows = True
    if check_actual_rows:
        if "Actual Rows" not in node:
            has_actual_rows = False
        elif node["Actual Rows"] is None:
            has_actual_rows = False

    node_type = node.get("Node Type", "")

    # Extract relevant info for Join and Scan nodes
    if node_type in JOIN_NODE_TYPES or node_type in SCAN_NODE_TYPES:
        # Extract self_cost
        total_cost = safe_float(node.get("Total Cost", 0.0))
        startup_cost = safe_float(node.get("Startup Cost", 0.0))
        self_cost = total_cost - startup_cost

        # For join nodes, compute self cost more accurately
        if node_type in JOIN_NODE_TYPES:
            children = node.get("Plans", [])
            if len(children) >= 2:
                child_total_cost = sum(safe_float(c.get("Total Cost", 0.0)) for c in children)
                self_cost = max(0.0, total_cost - child_total_cost)

        node_info = {
            "node_type": node_type,
            "depth": depth,
            "self_cost": self_cost,
            "total_cost": total_cost,
            "plan_rows": safe_float(node.get("Plan Rows", 0.0)),
            "plan_width": safe_float(node.get("Plan Width", 0.0)),
            "actual_rows": safe_float(node.get("Actual Rows", 0.0)),
            "total_rows": safe_float(node.get("Plan Rows", 0.0)),
        }

        # Extract table name for scan nodes
        if node_type in SCAN_NODE_TYPES:
            node_info["table_name"] = node.get("Relation Name", "")
            node_info["alias"] = node.get("Alias", "")
            tok = normalize_table_token(node_info["alias"], node_info["table_name"])
            node_info["table_set"] = [tok] if tok else []
        else:
            node_info["table_name"] = ""
            node_info["alias"] = ""
            node_info["table_set"] = []

        # Extract predicates (raw text)
        predicates = extract_predicates_from_node(node)
        node_info["predicates"] = predicates

        # Extract column names from predicates for separate encoding
        columns = set()
        for pred in predicates:
            cols, _, _ = parse_predicate_elements(pred)
            columns.update(cols)
        node_info["columns"] = list(columns)

        nodes_list.append(node_info)

    # Also return list of (plan_file, node_indices) for tracking
    # This is used to split data by plan file instead of by node
    if not hasattr(parse_plan_node, '_plan_to_indices'):
        parse_plan_node._plan_to_indices = {}

    # Recursively process children
    children = node.get("Plans", [])
    child_has_actual = True
    for child in children:
        _, child_ok = parse_plan_node(child, depth + 1, nodes_list, check_actual_rows)
        if not child_ok:
            child_has_actual = False

    return nodes_list, has_actual_rows and child_has_actual


def collect_nodes_from_plan_files(
    plan_dir: str,
    check_actual_rows: bool = True,
    verbose: bool = True,
) -> List[Dict]:
    """
    Collect all Join/Scan nodes from plan.json files in a directory hierarchy.

    Returns:
        List of node dictionaries with 'plan_file' field indicating source plan file.
    """
    all_nodes = []
    skipped_files = 0
    total_files = 0
    files_without_actual_rows = 0

    # Find all plan.json files
    for root, dirs, files in os.walk(plan_dir):
        for filename in files:
            if filename == "plan.json":
                filepath = os.path.join(root, filename)
                total_files += 1

                try:
                    with open(filepath, "r") as f:
                        plan_data = json.load(f)

                    # Handle both wrapped and unwrapped plan formats
                    if "Plan" in plan_data:
                        plan_root = plan_data["Plan"]
                    else:
                        plan_root = plan_data

                    nodes_list, has_actual_rows = parse_plan_node(
                        plan_root,
                        depth=0,
                        check_actual_rows=check_actual_rows
                    )

                    if check_actual_rows and not has_actual_rows:
                        files_without_actual_rows += 1
                        skipped_files += len(nodes_list)
                        continue

                    # Add plan_file field to each node for tracking
                    for node in nodes_list:
                        node["plan_file"] = filepath

                    all_nodes.extend(nodes_list)

                except Exception as e:
                    if verbose:
                        print(f"Error processing {filepath}: {e}")

    if verbose:
        print(f"Collected {len(all_nodes)} nodes from {total_files} plan files")
        if check_actual_rows:
            print(f"  Files without 'Actual Rows': {files_without_actual_rows}")
            print(f"  Skipped {skipped_files} nodes")

    return all_nodes


# ---------------------------
# Feature encoding
# ---------------------------
def encode_node_features(
    node: Dict,
    encoder: Optional[TextEncoder] = None,
    norm_stats: Optional[Tuple] = None,
    table_to_idx: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """
    Encode a single node's features using sentence-transformer embeddings.

    Args:
        node: Node dictionary with fields: node_type, depth, self_cost, total_cost,
              plan_rows, total_rows, table_set, predicates, columns
        encoder: kept for backward compatibility (unused in new feature layout)
        norm_stats: Normalization statistics tuple

    Returns:
        Feature vector as numpy array
    """
    # 1. Node type one-hot (6)
    onehot_type = np.zeros(len(NODE_TYPES), dtype=np.float32)
    node_type = node.get("node_type", "")
    if node_type in NODE_TYPES:
        onehot_type[NODE_TYPES.index(node_type)] = 1.0

    # 2. Table multi-hot
    table_dim = len(table_to_idx) if table_to_idx is not None else 0
    table_vec = np.zeros(table_dim, dtype=np.float32)
    table_set = node.get("table_set", []) or []
    if table_dim > 0:
        for tok in table_set:
            idx = table_to_idx.get(str(tok))
            if idx is not None:
                table_vec[idx] = 1.0

    # 3. Numeric features (6) - normalized (except selectivity kept raw)
    self_cost = safe_float(node.get("self_cost", 0.0))
    plan_rows = safe_float(node.get("plan_rows", 0.0))
    total_cost = safe_float(node.get("total_cost", 0.0))
    total_rows = safe_float(node.get("total_rows", 0.0))
    depth = safe_float(node.get("depth", 0.0))
    plan_width = safe_float(node.get("plan_width", 0.0))
    selectivity = plan_rows / max(total_rows, 1e-9)

    if norm_stats is not None:
        # Unpack norm_stats
        (self_cost_min, self_cost_max,
         plan_rows_min, plan_rows_max,
         total_cost_min, total_cost_max,
         plan_width_min, plan_width_max,
         depth_min, depth_max) = norm_stats

        # Log transform then min-max normalize
        self_cost_log = log_transform(self_cost)
        plan_rows_log = log_transform(plan_rows)
        total_cost_log = log_transform(total_cost)
        plan_width_log = log_transform(plan_width)

        norm_self_cost = (self_cost_log - self_cost_min) / (self_cost_max - self_cost_min + 1e-9)
        norm_plan_rows = (plan_rows_log - plan_rows_min) / (plan_rows_max - plan_rows_min + 1e-9)
        norm_total_cost = (total_cost_log - total_cost_min) / (total_cost_max - total_cost_min + 1e-9)
        norm_plan_width = (plan_width_log - plan_width_min) / (plan_width_max - plan_width_min + 1e-9)
        norm_depth = (depth - depth_min) / (depth_max - depth_min + 1e-9)

        # Clip to [0, 1]
        norm_self_cost = np.clip(norm_self_cost, 0.0, 1.0)
        norm_plan_rows = np.clip(norm_plan_rows, 0.0, 1.0)
        norm_total_cost = np.clip(norm_total_cost, 0.0, 1.0)
        norm_plan_width = np.clip(norm_plan_width, 0.0, 1.0)
        norm_depth = np.clip(norm_depth, 0.0, 1.0)

        num_feats = np.array([
            norm_self_cost,
            norm_plan_rows,
            norm_total_cost,
            selectivity,
            norm_plan_width,
            norm_depth,
        ], dtype=np.float32)
    else:
        # No normalization - use log transform for large-scale fields
        num_feats = np.array([
            log_transform(self_cost),
            log_transform(plan_rows),
            log_transform(total_cost),
            selectivity,
            log_transform(plan_width),
            depth,
        ], dtype=np.float32)

    # Concatenate all features
    # layout: onehot_type(6) + table_multi_hot(T) + num_feats(6)
    features = np.concatenate([
        onehot_type,
        table_vec,
        num_feats,
    ], axis=0)

    return features


# ---------------------------
# Main dataset encoder
# ---------------------------
def encode_plan_dataset(
    plan_dir: str,
    out_dir: Optional[str] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Encode a dataset of plan.json files using sentence-transformer embeddings.

    Args:
        plan_dir: Directory containing plan.json files
        out_dir: Directory to save artifacts
        model_name: Name of the sentence-transformers model
        device: Device to use for encoding
        seed: Random seed

    Returns:
        Tuple of (X, y, qerrors, info)
    """
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    print("Step 1: Scanning plan files and building table vocabulary...")
    plan_files = collect_plan_files(plan_dir)
    table_to_idx = build_table_vocab_from_plan_files(plan_files)
    print(f"  plan files: {len(plan_files)}")
    print(f"  table vocab size: {len(table_to_idx)}")

    print("Step 2: Collecting nodes from plan files...")
    nodes = collect_nodes_from_plan_files(plan_dir, check_actual_rows=True)

    if len(nodes) == 0:
        print("No nodes found!")
        return None, None, None, {}

    print(f"Collected {len(nodes)} nodes")

    # Compute normalization statistics
    print("Step 3: Computing normalization statistics...")
    self_costs = []
    plan_rows_list = []
    total_costs = []
    plan_widths = []
    selectivities = []
    depths = []

    for node in nodes:
        self_costs.append(safe_float(node.get("self_cost", 0.0)))
        plan_rows_list.append(safe_float(node.get("plan_rows", 0.0)))
        total_costs.append(safe_float(node.get("total_cost", 0.0)))
        plan_widths.append(safe_float(node.get("plan_width", 0.0)))
        pr = safe_float(node.get("plan_rows", 0.0))
        tr = safe_float(node.get("total_rows", 0.0))
        selectivities.append(pr / max(tr, 1e-9))
        depths.append(safe_float(node.get("depth", 0.0)))

    # Log transform
    self_costs_log = np.array([log_transform(x) for x in self_costs])
    plan_rows_log = np.array([log_transform(x) for x in plan_rows_list])
    total_costs_log = np.array([log_transform(x) for x in total_costs])
    plan_widths_log = np.array([log_transform(x) for x in plan_widths])
    depths_arr = np.array(depths)

    norm_stats = (
        np.min(self_costs_log), np.max(self_costs_log),
        np.min(plan_rows_log), np.max(plan_rows_log),
        np.min(total_costs_log), np.max(total_costs_log),
        np.min(plan_widths_log), np.max(plan_widths_log),
        np.min(depths_arr), np.max(depths_arr),
    )

    print(f"  Self cost: [{norm_stats[0]:.4f}, {norm_stats[1]:.4f}]")
    print(f"  Plan rows: [{norm_stats[2]:.4f}, {norm_stats[3]:.4f}]")
    print(f"  Total cost: [{norm_stats[4]:.4f}, {norm_stats[5]:.4f}]")
    if selectivities:
        print(f"  Selectivity(raw): [{min(selectivities):.4f}, {max(selectivities):.4f}]")
    print(f"  Plan width: [{norm_stats[6]:.4f}, {norm_stats[7]:.4f}]")
    print(f"  Depth: [{norm_stats[8]:.4f}, {norm_stats[9]:.4f}]")

    # Encode nodes
    print("Step 4: Encoding nodes (this may take a while on first run)...")
    X_list = []
    y_list = []
    q_list = []
    plan_files_list = []  # Track source plan file for each node

    # Create mapping from plan file to index
    plan_file_to_idx = {}
    for node in tqdm(nodes, desc="Encoding"):
        feat = encode_node_features(
            node,
            encoder=None,
            norm_stats=norm_stats,
            table_to_idx=table_to_idx,
        )

        # Get label and qerror
        est = safe_float(node.get("plan_rows", 0.0))
        act = safe_float(node.get("actual_rows", 0.0))
        bucket_name, q = qerror_and_bucket(est, act)
        bucket_id = bucket2id[bucket_name]

        X_list.append(feat)
        y_list.append(bucket_id)
        q_list.append(q)

        # Track plan file
        plan_file = node.get("plan_file", "")
        if plan_file not in plan_file_to_idx:
            plan_file_to_idx[plan_file] = len(plan_file_to_idx)
        plan_files_list.append(plan_file_to_idx[plan_file])

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    qerrors = np.array(q_list, dtype=np.float32)
    plan_file_indices = np.array(plan_files_list, dtype=np.int32)

    feature_dim = X.shape[1]
    print(f"\nEncoded {X.shape[0]} nodes with {X.shape[1]} features")
    print(f"  Feature breakdown: 6 (type) + {len(table_to_idx)} (table multi-hot) + 6 (numeric)")
    print(f"  Number of unique plan files: {len(plan_file_to_idx)}")

    # Save artifacts
    if out_dir is not None:
        print("\nStep 5: Saving artifacts...")

        # Save normalization stats
        norm_stats_array = np.array(norm_stats)
        np.save(os.path.join(out_dir, "norm_stats.npy"), norm_stats_array)

        # Save encoded data
        np.save(os.path.join(out_dir, "X.npy"), X)
        np.save(os.path.join(out_dir, "y.npy"), y)
        np.save(os.path.join(out_dir, "qerrors.npy"), qerrors)

        # Save plan file indices for proper splitting
        np.save(os.path.join(out_dir, "plan_file_indices.npy"), plan_file_indices)

        # Save plan file names (mapping from index to file path)
        idx_to_file = {idx: file for file, idx in plan_file_to_idx.items()}
        with open(os.path.join(out_dir, "plan_file_names.json"), "w") as f:
            json.dump(idx_to_file, f, indent=2)

        # Save config
        config = {
            "model_name": "structured_table_features_v1",
            "feature_dim": feature_dim,
            "num_nodes": len(nodes),
            "norm_stats": norm_stats,
            "table_vocab_size": len(table_to_idx),
            "table_vocab": sorted(table_to_idx.keys(), key=lambda x: table_to_idx[x]),
        }
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved artifacts to {out_dir}")

    # Info dictionary
    info = {
        "model_name": "structured_table_features_v1",
            "feature_dim": feature_dim,
        "num_nodes": len(nodes),
        "norm_stats": norm_stats,
        "table_vocab_size": len(table_to_idx),
    }

    return X, y, qerrors, info


# ---------------------------
# Single node encoding (for inference)
# ---------------------------
def encode_single_node(
    node: Dict,
    artifacts_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Encode a single node using saved artifacts.

    Args:
        node: Node dictionary
        artifacts_dir: Directory containing saved artifacts
        model_name: Name of the sentence-transformers model

    Returns:
        Feature vector
    """
    # Load norm_stats
    norm_stats = tuple(np.load(os.path.join(artifacts_dir, "norm_stats.npy")))

    table_to_idx: Dict[str, int] = {}
    # Load config if exists
    config_path = os.path.join(artifacts_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        table_vocab = config.get("table_vocab", [])
        table_to_idx = {str(t): i for i, t in enumerate(table_vocab)}

    return encode_node_features(
        node,
        encoder=None,
        norm_stats=norm_stats,
        table_to_idx=table_to_idx,
    )


# ---------------------------
# Tree Building for Tree-LSTM
# ---------------------------

def build_tree_from_plan_json(
    plan_json: Dict,
    encoder: Optional[TextEncoder],
    norm_stats: Optional[Tuple] = None,
    table_to_idx: Optional[Dict[str, int]] = None,
    table_row_provider: Optional[TableRowsProvider] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build a tree structure from a single plan.json for tree-LSTM processing.

    Returns:
        {
            "features": List[List[float]],
            "children": List[List[int]],    # [left_id, right_id], -1 if missing
        }
        Returns None if no relevant nodes found.
    """
    features = []
    children = []

    def add_node(feat: List[float]) -> int:
        """Add a node and return its index."""
        idx = len(features)
        features.append(feat)
        children.append([-1, -1])
        return idx

    def visit(node: Dict, depth: int) -> Tuple[List[int], Set[str]]:
        """
        Recursively visit plan nodes.
        Returns list of relevant node indices from this subtree.
        """
        node_type = node.get("Node Type", "")
        plans = node.get("Plans", [])

        # First, recurse to children
        child_roots = []
        child_tables: Set[str] = set()
        for child in plans:
            roots, tables = visit(child, depth + 1)
            child_roots.extend(roots)
            child_tables.update(tables)

        # Check if this is a relevant node (scan or join)
        if node_type in JOIN_NODE_TYPES or node_type in SCAN_NODE_TYPES:
            # Build node info dict for encoding
            # Extract self_cost
            total_cost = safe_float(node.get("Total Cost", 0.0))
            startup_cost = safe_float(node.get("Startup Cost", 0.0))
            self_cost = total_cost - startup_cost

            # For join nodes, compute self cost more accurately
            if node_type in JOIN_NODE_TYPES and len(plans) >= 2:
                child_total_cost = sum(safe_float(c.get("Total Cost", 0.0)) for c in plans)
                self_cost = max(0.0, total_cost - child_total_cost)

            node_info = {
                "node_type": node_type,
                "depth": depth,
                "self_cost": self_cost,
                "total_cost": total_cost,
                "plan_rows": safe_float(node.get("Plan Rows", 0.0)),
                "plan_width": safe_float(node.get("Plan Width", 0.0)),
            }

            # Extract table name for scan nodes
            if node_type in SCAN_NODE_TYPES:
                node_info["table_name"] = node.get("Relation Name", "")
                node_info["alias"] = node.get("Alias", "")
                tok = normalize_table_token(node_info["alias"], node_info["table_name"])
                node_info["table_set"] = [tok] if tok else []
                if table_row_provider is None:
                    raise RuntimeError("table_row_provider is required for scan total_rows resolution")
                node_info["total_rows"] = table_row_provider.get_table_rows(node_info["table_name"])
            else:
                node_info["table_name"] = ""
                node_info["alias"] = ""
                node_info["table_set"] = sorted(child_tables)
                if len(plans) >= 2:
                    l_rows = safe_float(plans[0].get("Plan Rows", 0.0))
                    r_rows = safe_float(plans[1].get("Plan Rows", 0.0))
                    node_info["total_rows"] = max(1.0, l_rows * r_rows)
                else:
                    node_info["total_rows"] = max(1.0, safe_float(node.get("Plan Rows", 0.0)))

            # Extract predicates
            predicates = extract_predicates_from_node(node)
            node_info["predicates"] = predicates

            # Extract column names
            columns = set()
            for pred in predicates:
                cols, _, _ = parse_predicate_elements(pred)
                columns.update(cols)
            node_info["columns"] = list(columns)

            # Encode node features
            feat = encode_node_features(
                node_info,
                encoder=encoder,
                norm_stats=norm_stats,
                table_to_idx=table_to_idx,
            )
            feat = feat.tolist()

            # Add this node
            my_idx = add_node(feat)

            # Attach children
            left_id = child_roots[0] if len(child_roots) >= 1 else -1
            right_id = child_roots[1] if len(child_roots) >= 2 else -1
            children[my_idx] = [left_id, right_id]

            return [my_idx], set(node_info["table_set"])
        else:
            # Not a relevant node - bubble up child roots
            return child_roots, child_tables

    # Handle wrapped plan format
    if "Plan" in plan_json:
        plan_root = plan_json["Plan"]
    else:
        plan_root = plan_json

    roots, _ = visit(plan_root, 0)

    if not roots:
        return None

    # Ensure root is at index 0
    root_idx = roots[0]
    if root_idx != 0:
        order = [root_idx] + [i for i in range(len(features)) if i != root_idx]
        remap = {old: new for new, old in enumerate(order)}
        new_features = [features[old] for old in order]
        new_children = []
        for old in order:
            l, r = children[old]
            nl = remap[l] if l != -1 else -1
            nr = remap[r] if r != -1 else -1
            new_children.append([nl, nr])
        features, children = new_features, new_children

    return {"features": features, "children": children}


def encode_plan_dataset_as_trees(
    plan_dir: str,
    out_dir: Optional[str] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    seed: int = 42,
    db_name: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Encode plan.json files as tree structures for tree-LSTM.

    Args:
        plan_dir: Directory containing plan.json files (e.g., demonstration/pool-tree/)
        out_dir: Directory to save artifacts
        model_name: Name of the sentence-transformers model
        device: Device to use for encoding
        seed: Random seed

    Returns:
        Tuple of (trees, info)
        - trees: List of {"features": [...], "children": [...]} dicts
        - info: Metadata dictionary
    """
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    print("Step 1: Scanning for plan files...")
    plan_files = []
    parent_ids = []
    parent_indices = []

    # Walk through directory structure looking for plan.json files
    # Expected structure: parent_dir/0-8/plan.json
    for root, dirs, files in os.walk(plan_dir):
        if "plan.json" in files:
            plan_path = os.path.join(root, "plan.json")
            # Extract parent info from path
            # Expected: .../pool-tree/1a/0/plan.json
            parts = Path(root).parts
            if len(parts) >= 2:
                try:
                    parent_index = int(parts[-1])  # Last folder is 0-8
                    parent_id = parts[-2]  # Second to last is parent (1a, 2b, etc.)
                    parent_ids.append(parent_id)
                    parent_indices.append(parent_index)
                    plan_files.append((plan_path, parent_id, parent_index))
                except (ValueError, IndexError):
                    # If we can't parse parent info, skip
                    plan_files.append((plan_path, "", -1))

    print(f"  Found {len(plan_files)} plan files")
    table_vocab_paths = [p for p, _, _ in plan_files]
    table_to_idx = build_table_vocab_from_plan_files(table_vocab_paths)
    print(f"  Table vocab size: {len(table_to_idx)}")

    print("Step 2: Computing normalization statistics...")
    table_row_provider = TableRowsProvider(
        dbname=db_name,
    )
    # First pass: collect nodes for norm stats
    all_nodes_for_stats = []
    for plan_path, _, _ in plan_files[:100]:  # Sample for stats
        try:
            with open(plan_path, "r") as f:
                plan_data = json.load(f)
            if "Plan" in plan_data:
                plan_root = plan_data["Plan"]
            else:
                plan_root = plan_data

            # Collect nodes from this plan for stats
            def collect_for_stats(node: Dict, depth: int = 0):
                node_type = node.get("Node Type", "")
                if node_type in JOIN_NODE_TYPES or node_type in SCAN_NODE_TYPES:
                    total_cost = safe_float(node.get("Total Cost", 0.0))
                    startup_cost = safe_float(node.get("Startup Cost", 0.0))
                    self_cost = total_cost - startup_cost

                    if node_type in JOIN_NODE_TYPES:
                        plans = node.get("Plans", [])
                        if len(plans) >= 2:
                            child_total = sum(safe_float(c.get("Total Cost", 0.0)) for c in plans)
                            self_cost = max(0.0, total_cost - child_total)

                    all_nodes_for_stats.append({
                        "self_cost": self_cost,
                        "plan_rows": safe_float(node.get("Plan Rows", 0.0)),
                        "total_cost": total_cost,
                        "plan_width": safe_float(node.get("Plan Width", 0.0)),
                        "total_rows": (
                            table_row_provider.get_table_rows(node.get("Relation Name", ""))
                            if node_type in SCAN_NODE_TYPES
                            else max(
                                1.0,
                                safe_float(node.get("Plans", [{}])[0].get("Plan Rows", 0.0))
                                * safe_float(node.get("Plans", [{}, {}])[1].get("Plan Rows", 0.0))
                                if len(node.get("Plans", [])) >= 2
                                else safe_float(node.get("Plan Rows", 0.0))
                            )
                        ),
                        "selectivity": (
                            safe_float(node.get("Plan Rows", 0.0))
                            / max(
                                (
                                    table_row_provider.get_table_rows(node.get("Relation Name", ""))
                                    if node_type in SCAN_NODE_TYPES
                                    else max(
                                        1.0,
                                        safe_float(node.get("Plans", [{}])[0].get("Plan Rows", 0.0))
                                        * safe_float(node.get("Plans", [{}, {}])[1].get("Plan Rows", 0.0))
                                        if len(node.get("Plans", [])) >= 2
                                        else safe_float(node.get("Plan Rows", 0.0))
                                    )
                                ),
                                1e-9,
                            )
                        ),
                        "depth": float(depth),
                    })

                for child in node.get("Plans", []):
                    collect_for_stats(child, depth + 1)

            collect_for_stats(plan_root)
        except Exception as e:
            table_row_provider.close()
            raise RuntimeError(f"failed while computing normalization stats for {plan_path}: {e}") from e

    # Compute norm stats
    if not all_nodes_for_stats:
        norm_stats = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        print("  [warn] no nodes for stats, using default norm stats")
    else:
        self_costs = [log_transform(n["self_cost"]) for n in all_nodes_for_stats]
        plan_rows = [log_transform(n["plan_rows"]) for n in all_nodes_for_stats]
        total_costs = [log_transform(n["total_cost"]) for n in all_nodes_for_stats]
        selectivities = [safe_float(n["selectivity"]) for n in all_nodes_for_stats]
        plan_widths = [log_transform(n["plan_width"]) for n in all_nodes_for_stats]
        depths = [n["depth"] for n in all_nodes_for_stats]

        norm_stats = (
            np.min(self_costs), np.max(self_costs),
            np.min(plan_rows), np.max(plan_rows),
            np.min(total_costs), np.max(total_costs),
            np.min(plan_widths), np.max(plan_widths),
            np.min(depths), np.max(depths),
        )

    print(f"  Self cost: [{norm_stats[0]:.4f}, {norm_stats[1]:.4f}]")
    print(f"  Plan rows: [{norm_stats[2]:.4f}, {norm_stats[3]:.4f}]")
    print(f"  Total cost: [{norm_stats[4]:.4f}, {norm_stats[5]:.4f}]")
    if all_nodes_for_stats:
        print(f"  Selectivity(raw): [{min(selectivities):.4f}, {max(selectivities):.4f}]")
    print(f"  Plan width: [{norm_stats[6]:.4f}, {norm_stats[7]:.4f}]")
    print(f"  Depth: [{norm_stats[8]:.4f}, {norm_stats[9]:.4f}]")

    print("Step 3: Encoding plans as trees...")
    trees = []
    metadata = []

    for plan_path, parent_id, parent_index in tqdm(plan_files, desc="Encoding trees"):
        try:
            with open(plan_path, "r") as f:
                plan_data = json.load(f)

            tree = build_tree_from_plan_json(
                plan_data,
                encoder=None,
                norm_stats=norm_stats,
                table_to_idx=table_to_idx,
                table_row_provider=table_row_provider,
            )
            if tree is not None:
                trees.append(tree)

                # Extract hint from hint.txt in the same folder
                hint_dir = os.path.dirname(plan_path)
                hint_path = os.path.join(hint_dir, "hint.txt")
                hint = ""
                if os.path.exists(hint_path):
                    with open(hint_path, "r") as hf:
                        hint = hf.read().strip()

                # Find best_hint from parent folder (suggest_hint.txt)
                parent_dir = os.path.dirname(hint_dir)
                best_hint_path = os.path.join(parent_dir, "suggest_hint.txt")
                best_hint = ""
                if os.path.exists(best_hint_path):
                    with open(best_hint_path, "r") as hf:
                        best_hint = hf.read().strip()

                # Extract qerror_vector from plan
                qerror_vector = extract_qerror_vector_from_plan(plan_data)

                # Extract leading_sequence from plan
                leading_sequence = extract_leading_sequence_from_plan(plan_data)

                # Extract join_type and scan_type for hint comparison
                join_type = extract_join_type_from_plan(plan_data)
                scan_type = extract_scan_type_from_plan(plan_data)

                # Find best_hint_idx
                best_hint_idx = -1
                for idx in range(9):
                    test_hint_path = os.path.join(parent_dir, str(idx), "hint.txt")
                    if os.path.exists(test_hint_path):
                        with open(test_hint_path, "r") as hf:
                            test_hint = hf.read().strip()
                        if test_hint == best_hint:
                            best_hint_idx = idx
                            break

                metadata.append({
                    "plan_path": plan_path,
                    "parent_id": parent_id,
                    "parent_index": parent_index,
                    "hint": hint,
                    "best_hint": best_hint,
                    "best_hint_idx": best_hint_idx,
                    "qerror_vector": qerror_vector,
                    "leading_sequence": leading_sequence,
                    "join_type": join_type,
                    "scan_type": scan_type,
                })
        except Exception as e:
            table_row_provider.close()
            raise RuntimeError(f"failed while encoding tree for {plan_path}: {e}") from e

    print(f"\nEncoded {len(trees)} trees")
    print(f"  Number of unique parents: {len(set(m['parent_id'] for m in metadata))}")

    # Save artifacts
    if out_dir is not None:
        print("\nStep 4: Saving artifacts...")

        # Save trees
        with open(os.path.join(out_dir, "trees.json"), "w") as f:
            json.dump(trees, f)

        # Save metadata
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Save norm stats
        norm_stats_array = np.array(norm_stats)
        np.save(os.path.join(out_dir, "norm_stats.npy"), norm_stats_array)

        # Save config
        config = {
            "model_name": "structured_table_features_v1",
            "feature_dim": 6 + len(table_to_idx) + 6,
            "num_trees": len(trees),
            "norm_stats": norm_stats,
            "table_vocab_size": len(table_to_idx),
            "table_vocab": sorted(table_to_idx.keys(), key=lambda x: table_to_idx[x]),
        }
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved artifacts to {out_dir}")

    # Info dictionary
    info = {
        "model_name": "structured_table_features_v1",
        "feature_dim": 6 + len(table_to_idx) + 6,
        "num_trees": len(trees),
        "norm_stats": norm_stats,
        "table_vocab_size": len(table_to_idx),
        "metadata": metadata,
    }
    table_row_provider.close()
    return trees, info


def collect_plan_pairs_from_trees(
    trees: List[Dict[str, Any]],
    metadata: List[Dict[str, Any]],
    num_negative_samples: int = 6,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Collect plan pairs for similarity learning using similarity score.

    Uses the same similarity calculation as collect_pool_tree.py:
    - sim = 0.8 * to_best_similarity + 0.2 * qerror_similarity

    For each plan, creates:
    1. Positive pairs: with other plans from same parent (different hints)
    2. Negative pairs: with random plans from different parents

    Args:
        trees: List of tree dictionaries
        metadata: List of metadata dicts with hint, qerror_vector, etc.
        num_negative_samples: Number of negative samples per plan
        seed: Random seed

    Returns:
        List of {"tree1": {...}, "tree2": {...}, "sim": float} dicts
        sim: similarity score in [0, 1]
    """
    import random

    random.seed(seed)
    np.random.seed(seed)

    print("Step 1: Computing to_best vectors for each record...")
    # Group metadata by parent_id
    parent_to_metadata = {}
    for idx, meta in enumerate(metadata):
        parent_id = meta["parent_id"]
        if parent_id not in parent_to_metadata:
            parent_to_metadata[parent_id] = []
        parent_to_metadata[parent_id].append((idx, meta))

    # Compute to_best_vector for each record
    to_best_vectors = {}
    for parent_id, meta_list in parent_to_metadata.items():
        # Find best_hint for this parent
        best_hint = ""
        best_hint_idx = -1
        for idx, meta in meta_list:
            if meta.get("best_hint"):
                best_hint = meta["best_hint"]
                best_hint_idx = meta.get("best_hint_idx", -1)
                break

        # Find best_hint metadata
        best_meta = None
        for idx, meta in meta_list:
            if meta.get("parent_index") == best_hint_idx:
                best_meta = meta
                break

        if best_meta is None:
            continue

        # Compute to_best_vector for each record in this parent
        for idx, meta in meta_list:
            key = (meta["parent_id"], meta["parent_index"])
            to_best_vector, _ = hint_minus(
                meta["hint"],
                meta["leading_sequence"],
                meta["join_type"],
                meta["scan_type"],
                best_meta["hint"],
                best_meta["leading_sequence"],
                best_meta["join_type"],
                best_meta["scan_type"]
            )
            to_best_vectors[key] = to_best_vector

    print("Step 2: Creating pairs and computing raw similarities...")
    pairs = []

    # Group indices by parent for efficient lookup
    parent_to_indices = {}
    for idx, meta in enumerate(metadata):
        parent_id = meta["parent_id"]
        if parent_id not in parent_to_indices:
            parent_to_indices[parent_id] = []
        parent_to_indices[parent_id].append(idx)

    # Create pairs with raw similarity scores
    to_best_distances = []
    qerror_similarities = []

    for idx, (tree, meta) in enumerate(zip(trees, metadata)):
        parent_id = meta["parent_id"]
        parent_index = meta["parent_index"]

        # Positive pairs: same parent, different index
        if parent_id in parent_to_indices:
            same_parent_indices = [i for i in parent_to_indices[parent_id]
                                  if metadata[i]["parent_index"] > parent_index]
            for other_idx in same_parent_indices:
                other_meta = metadata[other_idx]

                # Compute raw similarities
                to_best_dist, qerror_sim = compute_similarity(meta, other_meta, to_best_vectors)

                pairs.append({
                    "tree1": tree,
                    "tree2": trees[other_idx],
                    "meta1": meta,
                    "meta2": other_meta,
                })
                to_best_distances.append(to_best_dist)
                qerror_similarities.append(qerror_sim)

        # Negative pairs: different parents
        different_parent_indices = []
        for other_parent_id, indices in parent_to_indices.items():
            if other_parent_id != parent_id:
                different_parent_indices.extend(indices)

        if different_parent_indices:
            # Sample random negatives
            sample_size = min(num_negative_samples, len(different_parent_indices))
            sampled_indices = random.sample(different_parent_indices, sample_size)
            for other_idx in sampled_indices:
                other_meta = metadata[other_idx]

                # Compute raw similarities
                to_best_dist, qerror_sim = compute_similarity(meta, other_meta, to_best_vectors)

                pairs.append({
                    "tree1": tree,
                    "tree2": trees[other_idx],
                    "meta1": meta,
                    "meta2": other_meta,
                })
                to_best_distances.append(to_best_dist)
                qerror_similarities.append(qerror_sim)

    print(f"Created {len(pairs)} pairs")

    # Convert to numpy arrays for normalization
    to_best_distances = np.array(to_best_distances)
    qerror_similarities = np.array(qerror_similarities)

    # Normalize to_best_distances to similarities
    d_min = np.min(to_best_distances)
    d_max = np.max(to_best_distances)

    if d_max == d_min:
        to_best_similarities = np.ones_like(to_best_distances)
    else:
        to_best_similarities = 1.0 - (to_best_distances - d_min) / (d_max - d_min)

    # Combine similarities: sim = 0.8 * to_best_sim + 0.2 * qerror_sim
    final_similarities = 0.8 * to_best_similarities + 0.2 * qerror_similarities

    # Update pairs with similarity scores and clean up
    for i, pair in enumerate(pairs):
        pair["sim"] = float(final_similarities[i])
        pair["tree1"] = pair["tree1"]
        pair["tree2"] = pair["tree2"]
        # Remove temporary metadata
        del pair["meta1"]
        del pair["meta2"]

    print(f"  Similarity stats: min={final_similarities.min():.4f}, max={final_similarities.max():.4f}, mean={final_similarities.mean():.4f}")

    return pairs


# ---------------------------
# If run as script
# ---------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_dir", type=str, required=True, help="Directory containing plan.json files")
    parser.add_argument("--out_dir", type=str, default="encoder_artifacts", help="Where to save artifacts")
    parser.add_argument("--model", type=str, default=os.environ.get("DIAGHINT_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"), help="Sentence transformer model")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu/cuda)")
    parser.add_argument("--mode", type=str, default="flat", choices=["flat", "tree", "pairs"],
                       help="Encoding mode: flat=flat embeddings, tree=tree structures, pairs=tree pairs")
    parser.add_argument("--num_negative", type=int, default=6, help="Number of negative samples per plan (for pairs mode)")
    parser.add_argument("--db_name", type=str, default="", help="Optional DB name override for table row lookup")
    args = parser.parse_args()
    db_name = args.db_name.strip() or None

    if args.mode == "flat":
        # Original flat encoding
        X, y, q, info = encode_plan_dataset(
            args.plan_dir,
            out_dir=args.out_dir,
            model_name=args.model,
            device=args.device,
        )

        if X is not None:
            print("\nDone!")
            print(f"X.shape = {X.shape}")
            print(f"y.shape = {y.shape}")
            print(f"qerrors.shape = {q.shape}")

    elif args.mode == "tree":
        # Tree encoding
        trees, info = encode_plan_dataset_as_trees(
            args.plan_dir,
            out_dir=args.out_dir,
            model_name=args.model,
            device=args.device,
            db_name=db_name,
        )

        print("\nDone!")
        print(f"Encoded {len(trees)} trees")

    elif args.mode == "pairs":
        # Tree encoding + pair collection
        trees, info = encode_plan_dataset_as_trees(
            args.plan_dir,
            out_dir=args.out_dir,
            model_name=args.model,
            device=args.device,
            db_name=db_name,
        )

        metadata = info["metadata"]
        pairs = collect_plan_pairs_from_trees(
            trees,
            metadata,
            num_negative_samples=args.num_negative,
        )

        # Save pairs
        with open(os.path.join(args.out_dir, "pairs.json"), "w") as f:
            json.dump(pairs, f)

        print("\nDone!")
        print(f"Encoded {len(trees)} trees")
        print(f"Created {len(pairs)} pairs")
