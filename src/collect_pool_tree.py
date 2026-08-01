import os
import json
from typing import List, Dict, Any, Optional
import numpy as np

# We will reuse plan encoding from plan_summarizer

from plan_summarizer import summarize_plan_tree, build_tree_from_plan, extract_plan_structures
import re
import random
def read_text_if_exists(path: str) -> Optional[str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return "/*+ */"


def load_plan_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def collect_pool_tree(root_dir: str = "outputs/demo_pool",
                      alias2id_path: str = "models/cardinality_bias/alias2id.json",
                      alias_embs_path: str = "models/cardinality_bias/alias_embs.npy",
                      norm_stats_path: Optional[str] = "models/cardinality_bias/norm_stats.npy") -> List[Dict[str, Any]]:
    """
    Traverse pool-tree. For each parent directory (e.g., 1a, 2a), record:
      - parent_id: folder name (e.g., 1a)
      - best_hint: content of suggest_hint.txt (if exists)
      - For child directories 0..8:
          - skip if plan.json missing
          - hint: content of hint.txt in child folder (if exists, else empty string)
          - deduplicate by hint across this parent
          - encode plan to feature tree via plan_summarizer.build_tree_from_plan
          - record index (0..8)
    Return an array of records without writing files.
    """

    records: List[Dict[str, Any]] = []

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    # Iterate parent folders (e.g., 1a, 2a, ...)
    for parent_name in sorted(os.listdir(root_dir)):
        parent_path = os.path.join(root_dir, parent_name)
        if not os.path.isdir(parent_path):
            continue

        # Read best hint from suggest_hint.txt in parent folder
        best_hint_path = os.path.join(parent_path, "suggest_hint.txt")
        best_hint = read_text_if_exists(best_hint_path) or ""

        # Track seen hints under this parent to deduplicate
        seen_hints = set()

        best_hint_idx = -1
        for idx in range(0, 9):
            hint_path = os.path.join(parent_path, str(idx), "hint.txt")
            hint_text = read_text_if_exists(hint_path) or ""
            if hint_text == best_hint:
                best_hint_idx = idx
                break
        else:
            # if not found, set to -1
            best_hint_idx = -1



        # Iterate child indices 0..8
        for idx in range(0, 9):
            child_path = os.path.join(parent_path, str(idx))
            if not os.path.isdir(child_path):
                continue

            # plan.json required; skip if missing
            plan_path = os.path.join(child_path, "plan.json")
            plan_obj = load_plan_json(plan_path)
            if plan_obj is None or not isinstance(plan_obj, dict):
                # skip this data, proceed to next index
                continue

            # hint.txt optional; if folder missing, hint is empty
            hint_path = os.path.join(child_path, "hint.txt")
            hint_text = read_text_if_exists(hint_path) or ""

            # Deduplicate by hint within this parent
            if hint_text in seen_hints:
                continue
            seen_hints.add(hint_text)

            # Build encoding for plan using plan_summarizer
            try:
                tree = build_tree_from_plan(
                    plan_obj,
                    alias2id_path=alias2id_path,
                    alias_embs_path=alias_embs_path,
                    norm_stats_path=norm_stats_path,
                )
                structs = extract_plan_structures(plan_obj)
            except Exception as e:
                # If encoding fails, still record minimal info with empty tree
                tree = {"features": [], "children": []}
                structs = {"used_tables": [], "leading_sequence": [], "join_type": [], "scan_type": []}

            # Append record
            records.append({
                "parent_id": parent_name,
                "parent_index": idx,
                "hint": hint_text,
                "best_hint": best_hint,
                "best_hint_idx": best_hint_idx,
                "feature_tree": tree,
                "used_tables": structs.get("used_tables", []),
                "leading_sequence": structs.get("leading_sequence", []),
                "join_type": structs.get("join_type", []),
                "scan_type": structs.get("scan_type", []),
                "qerror_vector": structs.get("qerror_vector", []),
                "selectivity_vector": structs.get("selectivity_vector", []),
            })

    return records


def find_record_by_id(records: List[Dict[str, Any]], parent_id: str, parent_index: int) -> Optional[Dict[str, Any]]:
    """
    Find a record by parent_id and parent_index.
    """
    for record in records:
        if record["parent_id"] == parent_id and record["parent_index"] == parent_index:
            return record
    return None




def hint_to_list(hint: str) -> List[List[str]]:
    """
    将 hint 字符串转换为列表形式。
    去掉前后 /*+ 和 */，按空格分割，括号外为操作名，括号内为参数。
    """
    # 1. 清理字符串：去除首尾空格
    content = hint.strip()
    
    # 2. 去除 SQL Hint 的注释标记 /*+ 和 */
    # 这里的判断比较宽松，只要是这两个符号开头结尾即可
    if content.startswith("/*+"):
        content = content[3:]
    if content.endswith("*/"):
        content = content[:-2]
    
    # 3. 使用正则表达式提取模式
    # r"(\w+)\s*\(([^)]+)\)"
    # (\w+)   : 捕获组1，匹配操作名 (如 Leading, HashJoin)
    # \s*     : 允许操作名和左括号之间有空格
    # \(      : 匹配左括号
    # ([^)]+) : 捕获组2，匹配括号内除右括号外的所有字符 (即参数部分)
    # \)      : 匹配右括号
    pattern = r"(\w+)\s*\(([^)]+)\)"
    
    matches = re.findall(pattern, content)
    
    result = []
    for op_name, args_str in matches:
        # args_str 是括号内的字符串，如 "a b c"
        # 使用 split() 默认按空白字符(空格、换行等)分割
        args = args_str.strip().split()
        # sort the args
        sorted_args = sorted(args)
        # 将操作名和参数列表合并
        result.append([op_name] + sorted_args)
        
    return result




def isjoin(op_name: str) -> bool:
    return op_name in ["HashJoin", "NestedLoop", "MergeJoin"]

def isscan(op_name: str) -> bool:
    return op_name in ["SeqScan", "IndexScan", "BitmapHeapScan", "BitmapIndexScan", "IndexOnlyScan"]

def permutation_embedding(
    l1: List[str],
    l2: List[str],
    normalize: bool = True
) -> np.ndarray:
    assert set(l1) == set(l2), "l1 和 l2 必须包含相同元素"

    n = len(l1)

    # 原始位置
    pos1 = {k: i for i, k in enumerate(l1)}
    pos2 = {k: i for i, k in enumerate(l2)}

    displacements = []
    before_counts = []
    after_counts = []

    for k in l1:
        d = pos2[k] - pos1[k]
        displacements.append(d)

        # 统计相对顺序变化
        before = 0
        after = 0
        for j in l1:
            if j == k:
                continue
            if pos1[j] < pos1[k] and pos2[j] > pos2[k]:
                after += 1   # 原来在前，现在在后
            if pos1[j] > pos1[k] and pos2[j] < pos2[k]:
                before += 1  # 原来在后，现在在前

        before_counts.append(before)
        after_counts.append(after)

    displacements = np.array(displacements, dtype=np.float32)
    before_counts = np.array(before_counts, dtype=np.float32)
    after_counts = np.array(after_counts, dtype=np.float32)

    # ---- pooling（长度无关的关键）----
    features = []

    for arr in [displacements, np.abs(displacements),
                before_counts, after_counts]:
        features.extend([
            arr.mean(),
            # arr.std(),
            arr.max()
        ])

    emb = np.array(features, dtype=np.float32)

    if normalize:
        norm = np.linalg.norm(emb) + 1e-8
        emb = emb / norm

    return emb

# must be the same used table (same template)
# how to transform the plan generated by hint1 to the plan generated by hint2
# only compute for the same query but different hint
def hint_minus(record1: Dict[str, Any], record2: Dict[str, Any]) -> List[str]:
    """
    Compute minus between two hints.
    """
    hint1 = hint_to_list(record1["hint"])
    hint2 = hint_to_list(record2["hint"])
    # hint2 - hint1 = (op_name, args) in hint2 but not in hint1 + (op_name, args)^-1 in hint1 but not in hint2
    minus_hint = []
    for op_name, *args in hint2:
        if [op_name] + args not in hint1:
            minus_hint.append([op_name] + args)
    for op_name, *args in hint1:
        if [op_name] + args not in hint2:
            if op_name == "Leading":
                continue
            if isjoin(op_name):
                join_method = op_name
                left_join = sorted(args[0:-1])
                right_join = [args[-1]]
                for join_node in record2["join_type"]:
                    if join_node[1] == left_join and join_node[2] == right_join:
                        if join_node[0].replace(" ", "") != join_method:
                            minus_hint.append([join_node[0].replace(" ", "")] + args)
                        break
                    else:
                        break
                
            if isscan(op_name):
                scan_method = op_name
                scan_table = args[0]
                for scan_node in record2["scan_type"]:
                    if scan_node[1] == scan_table:
                        if scan_node[0].replace(" ", "") != scan_method:
                            minus_hint.append([scan_node[0].replace(" ", "")] + args)
                        break
                    else:
                        break
    
    minus_vector = [0, [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for op_name, *args in minus_hint:
        if op_name == 'Set':
            minus_vector[0] = 1
        minus_vector[1] = permutation_embedding(record1["leading_sequence"], record2["leading_sequence"]).tolist()
        # print(minus_vector[1])
        if isjoin(op_name):
            if op_name == "HashJoin":
                minus_vector[2][0] += 1
            elif op_name == "NestedLoop":
                minus_vector[2][1] += 1
            elif op_name == "MergeJoin":
                minus_vector[2][2] += 1
        elif isscan(op_name):
            if op_name == "SeqScan":
                minus_vector[3][0] += 1
            elif op_name == "IndexScan":
                minus_vector[3][1] += 1
            elif op_name == "IndexOnlyScan":
                minus_vector[3][2] += 1
    # print(minus_vector)
    return minus_hint, minus_vector




def calculate_to_best_distance(v1, v2): 
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

def calculate_qerror_similarity(v1, v2):
    max_len = max(len(v1), len(v2))
    vec1_pad = np.pad(v1, (0, max_len - len(v1)))
    vec2_pad = np.pad(v2, (0, max_len - len(v2)))   
    qerror_sim = np.dot(vec1_pad, vec2_pad) / (np.linalg.norm(vec1_pad) * np.linalg.norm(vec2_pad))
    return qerror_sim



def compute_all_distances(pairs):
    to_best_distances = []
    qerror_similarities = []
    for pair in pairs:
        v1 = pair["tree1"]["to_best_vector"]
        v2 = pair["tree2"]["to_best_vector"]
        to_best_dist = calculate_to_best_distance(v1, v2)
        to_best_distances.append(to_best_dist)
        v1 = pair["tree1"]["qerror_vector"]
        v2 = pair["tree2"]["qerror_vector"]
        qerror_sim = calculate_qerror_similarity(v1, v2)
        qerror_similarities.append(qerror_sim)
    return np.array(to_best_distances), np.array(qerror_similarities)


def normalize_to_tobest_similarity(distances):
    d_min = np.min(distances)
    d_max = np.max(distances)
    
    if d_max == d_min:
        return np.ones_like(distances)
    
    similarities = 1.0 - (distances - d_min) / (d_max - d_min)
    return similarities


# def normalize_to_qerror_similarity(distances):
#     d_min = np.min(distances)
#     d_max = np.max(distances)
    
#     if d_max == d_min:
#         return np.ones_like(distances)
    
#     similarities = 1.0 - (distances - d_min) / (d_max - d_min)
#     return similarities


# def compute_similarity(record1: Dict[str, Any], record2: Dict[str, Any]) -> float:
#     """
#     Compute similarity between two records.
#     """
#     qerror_vector1 = record1["qerror_vector"]
#     qerror_vector2 = record2["qerror_vector"]
#     max_len = max(len(qerror_vector1), len(qerror_vector2))
#     vec1_pad = np.pad(qerror_vector1, (0, max_len - len(qerror_vector1)))
#     vec2_pad = np.pad(qerror_vector2, (0, max_len - len(qerror_vector2)))   
#     qerror_cos_sim = np.dot(vec1_pad, vec2_pad) / (np.linalg.norm(vec1_pad) * np.linalg.norm(vec2_pad))
    
    
#     to_best_vector1 = record1["to_best_vector"]
#     to_best_vector2 = record2["to_best_vector"]
    
    
#     return qerror_cos_sim



def make_data_pair(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Make data pairs from records.
    """
    data_pairs = []
    for record in records:
        to_best_hint, to_best_vector = hint_minus(record, find_record_by_id(records=records, parent_id=record["parent_id"], parent_index= record["best_hint_idx"]))
        record["to_best_hint"] = to_best_hint
        record["to_best_vector"] = to_best_vector
    
    for record in records:
        for record2 in [find_record_by_id(records=records, parent_id=record["parent_id"], parent_index=i)
                        for i in range(record["parent_index"]+1, 9)]:
            if record2 is None:
                continue
            # sim = compute_similarity(record, record2)
            data_pairs.append({
                "tree1": record,
                "tree2": record2,
                # "sim": sim,
            })
        # from records random sample 6 record2 and record2["parent_id"] != record["parent_id"]
        random_records = random.sample([r for r in records if r["parent_id"] != record["parent_id"]], 6)
        for record2 in random_records:
            # sim = compute_similarity(record, record2)
            data_pairs.append({
                "tree1": record,
                "tree2": record2,
                # "sim": sim,
            })
    to_best_distances, qerror_similarities = compute_all_distances(data_pairs)
    to_best_similarities = normalize_to_tobest_similarity(to_best_distances)
    # qerror_similarities = normalize_to_qerror_similarity(qerror_distances)
    for i, pair in enumerate(data_pairs):
        pair["sim"] = 0.8 * to_best_similarities[i] + 0.2 * qerror_similarities[i]
        pair["tree1"] = pair["tree1"]["feature_tree"]
        pair["tree2"] = pair["tree2"]["feature_tree"]
    return data_pairs

def load_records_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load records from a JSON file.
    """
    with open(file_path, "r") as f:
        records = json.load(f)
    return records





if __name__ == "__main__":
    # Execute and print summary size; do not write to disk per instruction
    arr = collect_pool_tree()
    print(f"Collected {len(arr)} records from pool-tree.")
    data_pairs = make_data_pair(arr)
    print(f"Collected {len(data_pairs)} data pairs from pool-tree.")
    # save data_pairs to disk
    with open("data_pairs.json", "w") as f:
        json.dump(data_pairs, f, indent=4)
    # save arr to disk
    with open("records.json", "w") as f:
        json.dump(arr, f, indent=4)
