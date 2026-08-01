import json
import logging
import os
from typing import Any, Dict, Optional, List, Tuple
import math
import numpy as np

# Configure logger
logger = logging.getLogger(__name__)

def safe_log_qerror(a: float, b: float) -> float:
    return math.log(( (max(0, a) + 0.001) / (max(0 , b) + 0.001)))

def safe_selectivity(a: float, b: float) -> float:
    # assert a >= 0 and b >= 0, f"Warning: negative value in selectivity calculation: a={a}, b={b}"
    if b <=0:
        return 0
    return a/b

def merge_struct_info(left_struct_info: Dict[str, Any], join_node_info: Dict[str, Any], right_struct_info: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key in left_struct_info:
        result[key] = left_struct_info[key] + join_node_info[key] + right_struct_info[key]
    return result

def summarize_plan_tree(plan: Dict[str, Any], depth=0) -> (str, List[Dict[str, Any]], set):
    """
    单递归遍历plan，返回(join_order str, scan/join节点node_list, 叶节点set)
    - join_order: 用s表达式展现所有join顺序和方法
    - node_list: 每个join节点额外记录input_left/input_right
    - 叶节点set用于生成父节点的input描述
    """
    node_type = plan.get('Node Type', '').lower()
    rel_name = plan.get('Relation Name')
    alias = plan.get('Alias')
    scan_label = alias or rel_name
    is_scan = 'scan' in node_type
    is_join = 'join' in node_type or 'nested loop' in node_type
    node_list = []
    struct_info = {
        'used_tables': [],
        'leading_sequence': [],
        'join_type': [],
        'scan_type': [],
        'qerror_vector': [],
        'selectivity_vector': [],
    }
    if is_scan and scan_label:
        op = plan.get('Node Type', '')
        # Debug: log when rel_name is None
        if rel_name is None:
            logger.warning(f"Scan node with None rel_name! scan_label={scan_label}, plan_node={plan.get('Node Type', '')}")
        node_summary = {
            'node_type': op,
            'label': scan_label,
            'depth': depth,
            'plan_node': plan,
            "self_cost": plan.get('Total Cost', 0.0)
        }
        node_list.append(node_summary)
        collect_single_node(node_summary)
        struct_info['used_tables'].append(rel_name)
        struct_info['leading_sequence'].append(scan_label)
        struct_info['scan_type'].append([op, scan_label, plan.get('Index Name', '')])
        struct_info['qerror_vector'].append(safe_log_qerror(plan.get('Actual Rows', 0.0), plan.get('Plan Rows', 0.0)))
        struct_info['selectivity_vector'].append(safe_selectivity(plan.get('Actual Rows', 0.0), get_total_rows(rel_name)))
        return scan_label, node_list, {scan_label}, {rel_name}, struct_info
    elif is_join:
        plans = plan.get('Plans', [])
        left_in, right_in = [], []
        join_expr = ""
        left_set, right_set = set(), set()
        left_nodes, right_nodes = [], []
        if len(plans) == 2:
            left_expr, left_nodes, left_set, left_rel_set, left_struct_info = summarize_plan_tree(plans[0], depth + 1)
            right_expr, right_nodes, right_set, right_rel_set, right_struct_info = summarize_plan_tree(plans[1], depth + 1)
            op = plan.get('Node Type', '').replace(' ', '')

            left_in, right_in = sorted(list(left_set)), sorted(list(right_set))
            left_in_rel, right_in_rel = sorted(list(left_rel_set)), sorted(list(right_rel_set))

            # Debug: check for None values in rel lists
            if None in left_in_rel:
                logger.error(f"None found in left_in_rel! left_rel_set={left_rel_set}, left_in_rel={left_in_rel}")
                logger.error(f"Left node type: {left_nodes[-1].get('node_type', 'unknown') if left_nodes else 'empty'}")
            if None in right_in_rel:
                logger.error(f"None found in right_in_rel! right_rel_set={right_rel_set}, right_in_rel={right_in_rel}")
                logger.error(f"Right node type: {right_nodes[-1].get('node_type', 'unknown') if right_nodes else 'empty'}")
            # -- 关键label: [a,b] JoinType [c]
            # keep linear: assure size left > right
            if len(left_in) < len(right_in):
                left_expr, right_expr = right_expr, left_expr
                left_nodes, right_nodes = right_nodes, left_nodes
                left_in, right_in = right_in, left_in
                left_in_rel, right_in_rel = right_in_rel, left_in_rel
                left_struct_info, right_struct_info = right_struct_info, left_struct_info
                swap = True
            else:
                swap = False
            join_expr = f"({left_expr} {op} {right_expr})"
            label = f"[{', '.join(left_in)}] {op} [{', '.join(right_in)}]"
            node_summary = {
                'node_type': plan.get('Node Type', ''),
                'label': label,
                'depth': depth,
                'plan_node': plan,
                'left_in': [', '.join(left_in_rel)],
                'right_in': [', '.join(right_in_rel)],
                'left_label_name': left_nodes[-1].get('label', ''),
                'right_label_name': right_nodes[-1].get('label', ''),
                'self_cost': plan.get('Total Cost', 0.0) - (left_nodes[-1].get('self_cost', 0.0) + right_nodes[-1].get('self_cost', 0.0)),
                'join_swap': swap,
            }
            node_list = left_nodes + right_nodes + [node_summary]
            total_set = left_set.union(right_set) # set for alias(output)
            rel_set = left_rel_set.union(right_rel_set) # set for relation_name
            collect_single_node(node_summary)

            struct_info['join_type'].append([op, left_in, right_in])
            struct_info['qerror_vector'].append(safe_log_qerror(plan.get('Actual Rows', 0.0), plan.get('Plan Rows', 0.0)))
            # struct_info['selectivity_vector'].append(plan.get('Actual Rows', 0.0) / (get_total_rows(left_in_rel) * get_total_rows(right_in_rel)))

            merge_info = merge_struct_info(left_struct_info, struct_info, right_struct_info)
            return join_expr, node_list, total_set, rel_set, merge_info

        else:
            # warning and skip
            logger.warning(f"Join node with {len(plans)} plans, expected 2.")

    # 非SCAN/JOIN节点：递归plans[0]
    plans = plan.get('Plans', [])
    if plans:
        return summarize_plan_tree(plans[0], depth + 1)
    # fallback
    result = (scan_label or plan.get('Node Type', '') or "", [], set(), set())

    return result

def get_dominant_join_types(node_list: List[Dict[str, Any]]) -> List[str]:
    join_types = {}
    for node in node_list:
        jt = node['plan_node'].get('Join Type')
        nt = node['node_type']
        if 'join' in nt.lower() and jt:
            join_types[jt] = join_types.get(jt, 0) + 1
        elif 'join' in nt.lower():
            join_types[nt] = join_types.get(nt, 0) + 1
    return sorted(join_types.keys())

def compute_plan_depth(node_list: List[Dict[str, Any]]) -> int:
    return max((n['depth'] for n in node_list), default=0)

def collect_top_cost_nodes(node_list: List[Dict[str, Any]], max_nodes=5) -> List[Dict[str, Any]]:
    # Try EXPLAIN ANALYZE (actual costs/time), fallback EXPLAIN
    nodes = []
    for node in node_list:
        plan = node['plan_node']
        node_type = node['node_type']
        label = node['label']
        # Pick cost metrics
        if 'Actual Total Time' in plan:
            cost = plan['Actual Total Time']
            kind = 'Actual Total Time'
        elif 'Total Cost' in plan:
            cost = plan['Total Cost']
            kind = 'Total Cost'
        else:
            continue
        nodes.append({
            'node': f"{node_type}{' on ' + label if label else ''}",
            'cost_type': kind,
            'cost': cost,
            'plan_rows': plan.get('Plan Rows'),
            'actual_rows': plan.get('Actual Rows'),
        })
    nodes = sorted(nodes, key=lambda x: x['cost'], reverse=True)
    return nodes[:max_nodes]

def _qerror_from_plan_node(plan_node: Dict[str, Any]) -> Optional[float]:
    est = plan_node.get('Plan Rows')
    act = plan_node.get('Actual Rows')
    if est is not None and act is not None and est > 0 and act > 0:
        return max(est / act, act / est)
    return None

def _format_count(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    try:
        n = float(v)
    except Exception:
        return str(v)
    if n >= 1_000_000:
        val = n / 1_000_000.0
        s = f"{val:.1f}m"
    elif n >= 1_000:
        val = n / 1_000.0
        s = f"{val:.1f}k"
    else:
        # 保留整数显示
        s = f"{int(n)}"
    # 去掉多余的 .0
    return s.replace('.0k', 'k').replace('.0m', 'm')

def _desc_est_err_old(est: Optional[float], act: Optional[float], q: Optional[float]) -> str:
    try:
        est_f = float(est)
        act_f = float(act)
    except Exception:
        return "N/A"
    # 零值规则
    if est_f == 0 and act_f == 0:
        return "Accurate"
    if est_f == 0 and act_f > 0:
        return "Total Underestimated"
    if act_f == 0 and est_f > 0:
        return "Total Overestimation"
    if abs(q - 1.0) < 1e-6:
        return "Accurate"
    # 一般情况
    if act_f > est_f:
        return f"Underestimated by {act_f/est_f:.1f}x"
    else:
        return f"Overestimated by {est_f/act_f:.1f}x"

def _desc_est_err(est: Optional[float], act: Optional[float], q: Optional[float]) -> str:
    try:
        est_f = float(est)
        act_f = float(act)
    except Exception:
        return "N/A"

    # 零值规则
    if est_f == 0 and act_f == 0:
        return "Accurate"
    if est_f == 0 and act_f > 0:
        return "Total Underestimated"
    if act_f == 0 and est_f > 0:
        return "Total Overestimation"

    # 计算qerror并记录日志
    qerror = max(est_f / act_f, act_f / est_f)
    if qerror is not None:
        logger.info(f"QError value: {qerror}")
        # 基于qerror值进行五类分类
        if qerror < 2:
            return "Approximately Accurate"
        elif qerror >= 2 and qerror < 10:
            if est_f > act_f:
                return "Slight Overestimation"
            else:
                return "Slight Underestimation"
        elif qerror >= 10:
            if est_f > act_f:
                return "Severe Overestimation"
            else:
                return "Severe Underestimation"

    # 如果没有qerror值，使用原始逻辑
    if act_f > est_f:
        return f"Underestimated by {act_f/est_f:.1f}x"
    else:
        return f"Overestimated by {est_f/act_f:.1f}x"

from plan_node.encoder import encode_node_features
from plan_node.predict import predict_from_vector

# Global variables for lazy loading
_global_norm_stats = None
_global_table_to_idx = None
_global_model_path = None
_global_predict_warning_logged = False


def _resolve_embedding_artifacts_dir() -> str:
    """Resolve a usable artifacts directory for new structured node encoding."""
    env_dir = os.environ.get("DIAGHINT_EMBEDDING_ARTIFACTS_DIR")
    candidates = []
    if env_dir:
        candidates.append(env_dir)

    candidates.extend(
        [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "cardinality_bias"),
        ]
    )

    for d in candidates:
        if not d:
            continue
        norm_stats_path = os.path.join(d, "norm_stats.npy")
        cfg_path = os.path.join(d, "config.json")
        if os.path.exists(norm_stats_path) and os.path.exists(cfg_path):
            return d

    searched = ", ".join(candidates)
    raise FileNotFoundError(
        f"norm_stats.npy not found in any artifacts dir. searched=[{searched}]"
    )


def _resolve_model_path() -> str:
    env_model = os.environ.get("DIAGHINT_EST_ERR_MODEL_PATH")
    candidates = [env_model] if env_model else []
    candidates.extend(
        [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "cardinality_bias", "best_model.pt"),
        ]
    )
    for p in candidates:
        if p and os.path.exists(p):
            return p
    searched = ", ".join([c for c in candidates if c])
    raise FileNotFoundError(f"model checkpoint not found. searched=[{searched}]")


def _get_encoding_artifacts():
    """Lazy load norm_stats + table vocab for feature encoding."""
    global _global_norm_stats, _global_table_to_idx, _global_model_path
    if _global_norm_stats is None or _global_table_to_idx is None:
        artifacts_dir = _resolve_embedding_artifacts_dir()
        norm_stats_path = os.path.join(artifacts_dir, "norm_stats.npy")
        cfg_path = os.path.join(artifacts_dir, "config.json")
        _global_norm_stats = tuple(np.load(norm_stats_path))
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        table_vocab = cfg.get("table_vocab", []) or []
        _global_table_to_idx = {str(t): i for i, t in enumerate(table_vocab)}
    if _global_model_path is None:
        _global_model_path = _resolve_model_path()
    return _global_norm_stats, _global_table_to_idx, _global_model_path


def _resolve_estimation_error_mode(explicit_predict_flag: Optional[bool] = None) -> bool:
    if explicit_predict_flag is not None:
        return bool(explicit_predict_flag)
    mode = os.environ.get("LLM_HINT_EST_ERR_MODE", "predict").strip().lower()
    if mode in {"na", "n/a", "none", "off", "false", "0"}:
        return False
    return True

def _desc_est_err_predict(node: Dict[str, Any]) -> str:
    """
    Predict estimation error bucket for a plan node using trained ML model.

    Returns one of:
    - "Severe Underestimation"
    - "Slight Underestimation"
    - "Approximately Accurate"
    - "Slight Overestimation"
    - "Severe Overestimation"
    """
    global _global_predict_warning_logged
    if node is None:
        return "N/A"

    try:
        # Lazy load new encoding artifacts (norm_stats + table vocab) and model path.
        norm_stats, table_to_idx, model_path = _get_encoding_artifacts()

        # Encode node with new structured feature layout.
        feature_vector = encode_node_features(
            node,
            encoder=None,
            norm_stats=norm_stats,
            table_to_idx=table_to_idx,
        )

        # Predict using auto input_dim inference from checkpoint if not provided.
        bucket_name, _ = predict_from_vector(
            feature_vector,
            model_path=model_path,
            input_dim=len(feature_vector),
            device='cpu'
        )
        return bucket_name
    except Exception as e:
        if not _global_predict_warning_logged:
            logger.warning(f"Error predicting estimation error: {e}")
            _global_predict_warning_logged = True
        return "N/A"

def collect_top_qerror_nodes(node_list: List[Dict[str, Any]], max_nodes=5) -> Dict[str, List[Dict[str, Any]]]:
    """
    For EXPLAIN ANALYZE:
    - Separate join and scan nodes
    - For join nodes, compute qerror_self and children (left/right) qerrors; sort by max of the three
    - For scan nodes, compute own qerror; sort by qerror
    Return dict: { 'join': [...], 'scan': [...] }
    """
    join_nodes: List[Dict[str, Any]] = []
    scan_nodes: List[Dict[str, Any]] = []
    for node in node_list:
        plan = node['plan_node']
        nt_lower = node['node_type'].lower()
        is_scan = 'scan' in nt_lower
        is_join = ('join' in nt_lower) or ('nested loop' in nt_lower)
        if is_join:
            q_self = _qerror_from_plan_node(plan)
            left_q, right_q = None, None
            plans = plan.get('Plans', []) or []
            self_est = plan.get('Plan Rows')
            self_act = plan.get('Actual Rows')
            left_est, left_act = None, None
            right_est, right_act = None, None
            if len(plans) >= 2:
                left_plan, right_plan = plans[0], plans[1]
                if node.get('join_swap', False):
                    left_plan, right_plan = right_plan, left_plan
                left_q = _qerror_from_plan_node(left_plan)
                left_est = left_plan.get('Plan Rows')
                left_act = left_plan.get('Actual Rows')

                right_q = _qerror_from_plan_node(right_plan)
                right_est = right_plan.get('Plan Rows')
                right_act = right_plan.get('Actual Rows')
            max_q = max([q for q in [q_self, left_q, right_q] if q is not None], default=None)
            join_nodes.append({
                'node': node['label'] or node['node_type'],
                'qerror_self': q_self,
                'qerror_left': left_q,
                'qerror_right': right_q,
                'qerror_max': max_q,
                'self_estimate': self_est,
                'self_actual': self_act,
                'left_estimate': left_est,
                'left_actual': left_act,
                'right_estimate': right_est,
                'right_actual': right_act,
            })
        elif is_scan:
            q = _qerror_from_plan_node(plan)
            if q is not None:
                scan_nodes.append({
                    'node': node['node_type'] + ' on ' + node['label'],
                    'qerror': q,
                    'estimate': plan.get('Plan Rows'),
                    'actual': plan.get('Actual Rows'),
                })
    join_nodes = sorted(join_nodes, key=lambda x: (x['qerror_max'] if x['qerror_max'] is not None else -1), reverse=True)
    scan_nodes = sorted(scan_nodes, key=lambda x: x['qerror'], reverse=True)
    return { 'join': join_nodes, 'scan': scan_nodes }


def plan_json_to_text(
    plan: Dict[str, Any],
    runtime_ms: Optional[float] = None,
    rows_out: Optional[int] = None,
    extra: Optional[Dict] = None,
    max_top_nodes: int = 20,
    predict_when_no_actual: Optional[bool] = None,
) -> str:
    join_order, node_list, _, _, _ = summarize_plan_tree(plan)
    plan_depth = compute_plan_depth(node_list)
    has_actual = any(('Actual Rows' in n['plan_node']) for n in node_list)
    parts = []
    if 'Total Cost' in plan:
        parts.append(f"Estimated Total Cost: {plan['Total Cost']}")
    if 'Actual Total Time' in plan:
        parts.append(f"Actual Total Runtime: {plan['Actual Total Time']:.2f} ms")
    summary = {
        "join_order": join_order,
        "total_cost": plan.get('Total Cost') or plan.get('Actual Total Time')
    }
    parts.append("(A) Plan summary\n" + json.dumps(summary, indent=2))
    if has_actual:
        top = collect_top_qerror_nodes(node_list, max_top_nodes)

        # 构建join节点的JSON格式
        join_nodes_json = []
        for n in top['join']:
            # 获取join类型和filter信息
            plan_node = None
            for node in node_list:
                if node['label'] == n['node'] or node['node_type'] == n['node']:
                    plan_node = node['plan_node']
                    break

            join_type = plan_node.get('Node Type', '') if plan_node else ''
            self_cost = plan_node.get('Total Cost', 0) if plan_node else 0

            # 构建join节点JSON
            join_node_json = {
                "description": n['node'],
                "join_type": join_type,
                "self_cost": self_cost,
                "estimated_rows": n.get('self_estimate', 0),
                "actual_rows": n.get('self_actual', 0),
                "estimation_error": _desc_est_err(n.get('self_estimate'), n.get('self_actual'), n.get('qerror_self')),
                "children": {
                    "left": {
                        "estimated_rows": n.get('left_estimate', 0),
                        "actual_rows": n.get('left_actual', 0),
                        # "estimation_error": _desc_est_err(n.get('left_estimate'), n.get('left_actual'), n.get('qerror_left'))
                    },
                    "right": {
                        "estimated_rows": n.get('right_estimate', 0),
                        "actual_rows": n.get('right_actual', 0),
                        # "estimation_error": _desc_est_err(n.get('right_estimate'), n.get('right_actual'), n.get('qerror_right'))
                    }
                }
            }
            filter_cond = plan_node.get('Join Filter', plan_node.get('Hash Cond', plan_node.get('Merge Cond', None)))
            if filter_cond:
                join_node_json['filter'] = filter_cond
            join_nodes_json.append(join_node_json)

        # 构建scan节点的JSON格式
        scan_nodes_json = []
        for n in top['scan']:
            # 获取scan节点的详细信息
            plan_node = None
            for node in node_list:
                if node['node_type'] + ' on ' + node['label'] == n['node']:
                    plan_node = node['plan_node']
                    break

            self_cost = plan_node.get('Total Cost', 0) if plan_node else 0

            # 构建scan节点JSON
            scan_node_json = {
                "description": n['node'],
                "self_cost": self_cost,
                "estimated_rows": n.get('estimate', 0),
                "actual_rows": n.get('actual', 0),
                "total_rows": get_total_rows(plan_node.get('Relation Name', '')),
                "estimation_error": _desc_est_err(n.get('estimate'), n.get('actual'), n.get('qerror'))
            }
            index_name = plan_node.get('Index Name', '')
            if index_name:
                scan_node_json['index_name'] = index_name
            filter_cond = plan_node.get('Filter', plan_node.get('Index Cond', None))
            if filter_cond:
                scan_node_json['filter'] = filter_cond
            scan_nodes_json.append(scan_node_json)

        # 构建最终JSON输出
        result_json = {
            "join_nodes": join_nodes_json,
            "scan_nodes": scan_nodes_json
        }

        parts.append("(B1) Top-qerror JOIN nodes\n\n" + json.dumps(result_json['join_nodes'], indent=2))
        parts.append("(B2) Top-qerror SCAN nodes\n\n" + json.dumps(result_json['scan_nodes'], indent=2))
    else:
        use_predict = _resolve_estimation_error_mode(predict_when_no_actual)
        # 分离join和scan节点，但不按qerror排序
        join_nodes: List[Dict[str, Any]] = []
        scan_nodes: List[Dict[str, Any]] = []
        for node in node_list:
            plan = node['plan_node']
            nt_lower = node['node_type'].lower()
            is_scan = 'scan' in nt_lower
            is_join = ('join' in nt_lower) or ('nested loop' in nt_lower)
            if is_join:
                join_nodes.append(node)
            elif is_scan:
                scan_nodes.append(node)

        join_nodes.sort(key=lambda n: n.get('self_cost', 0.0), reverse=True)
        scan_nodes.sort(key=lambda n: n.get('self_cost', 0.0), reverse=True)

        error_map: map[str, str] = {}
        # 生成scan节点JSON输出
        scan_json = []
        for n in scan_nodes:
            # 获取节点信息
            node_info = get_single_node_info(n)
            plan = n['plan_node']

            if use_predict:
                error_map[n['label']] = _desc_est_err_predict(node_info)
            else:
                error_map[n['label']] = "N/A (no Actual Rows)"
            # 构建scan节点JSON
            scan_node = {
                "description": f"{n['node_type']} on [{n['label']}]",
                "self_cost": plan.get('Total Cost', 0.0),
                "estimated_rows": plan.get('Plan Rows', 0),
                # "estimated_selectivity": safe_selectivity(plan.get('Plan Rows', 0), get_total_rows(plan.get('Relation Name'))),
                "estimation_error": error_map.get(n['label'], ""),
            }
            index_name = plan.get('Index Name', '')
            if index_name:
                scan_node['index_name'] = index_name
            filter_cond = plan.get('Filter', plan.get('Index Cond', None))
            if filter_cond:
                scan_node['filter'] = filter_cond
            scan_json.append(scan_node)

        join_json = []

        for n in join_nodes:
            # 获取误差信息，存储到map中
            node_info = get_single_node_info(n)
            if use_predict:
                error_map[n['label']] = _desc_est_err_predict(node_info)
            else:
                error_map[n['label']] = "N/A (no Actual Rows)"


        for n in join_nodes:
            node_info = get_single_node_info(n)
            plan = n['plan_node']

            # 获取左右子节点信息
            plans = plan.get('Plans', []) or []

            # 构建join节点JSON
            join_node = {
                "description": n['label'],
                "join_type": plan.get('Node Type', ''),
                "self_cost": plan.get('Total Cost', 0.0),
                "estimated_rows": plan.get('Plan Rows', 0),
                "estimation_error": error_map.get(n['label'], ""),
                "left": f"estimated_rows: {plans[0].get('Plan Rows', 0) if len(plans) >= 1 else 0}, estimation_error: {error_map.get(n['left_label_name'], '')}",
                "right": f"estimated_rows: {plans[1].get('Plan Rows', 0) if len(plans) >= 2 else 0}, estimation_error: {error_map.get(n['right_label_name'], '')}"
            }
            filter_cond = plan.get('Join Filter', plan.get('Hash Cond', plan.get('Merge Cond', None)))
            if filter_cond:
                join_node['filter'] = filter_cond
            join_json.append(join_node)

        parts.append("(B1) JOIN nodes analysis\n\n" + json.dumps(join_json, indent=2))
        parts.append("(B2) SCAN nodes analysis\n\n" + json.dumps(scan_json, indent=2))
    # perf = {
    #     'runtime_ms': runtime_ms,
    #     # 'rows_out': rows_out,
    #     'plan_depth': plan_depth,
    # }
    # if extra:
    #     perf.update(extra)
    # parts.append("(C) Performance metrics\n" + json.dumps(perf, indent=2))
    return "\n\n".join(parts)

import psycopg2, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # 将项目根目录添加到路径
from config import Config
def execute_query(sql: str) -> List[Tuple]:
    """
    执行SQL查询并返回结果

    Args:
        sql: 要执行的SQL查询字符串

    Returns:
        查询结果的列表，每个元素是一个元组
    """
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=Config.DB_DATABASE,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            host=Config.DB_HOST,
            port=Config.DB_PORT
        )
        conn.autocommit = True

        with conn.cursor() as cursor:
            cursor.execute(sql)
            return cursor.fetchall()
    except psycopg2.Error as e:
        print(f"数据库错误: {e}")
        return []
    finally:
        if conn:
            conn.close()
total_rows_cache = {}
def get_total_rows(table: str) -> int:
    """
    获取表的总行数

    Args:
        table: 表名
    Returns:
        表的总行数
    """
    # 检查缓存中是否已存在该表的总行数
    if table in total_rows_cache:
        return total_rows_cache[table]

    # 这里假设使用PostgreSQL的pg_class系统表来获取表的行数
    query = f"SELECT reltuples FROM pg_class WHERE relname = '{table}'"
    # 执行查询并返回结果
    result = execute_query(query)
    if result:
        # 缓存结果
        total_rows_cache[table] = int(result[0][0])
        return int(result[0][0])
    else:
        return 0

def get_single_node_info(summary: Dict[str, Any]) -> Dict[str, Any]:
    node = summary.get('plan_node', {})
    is_scan = node.get('Node Type') in ['Seq Scan', 'Index Scan', 'Index Only Scan']
    is_join = node.get('Node Type') in ['Nested Loop', 'Hash Join', 'Merge Join']
    if not is_scan and not is_join:
        return None
    node_info: Dict[str, Any] = {}
    if is_scan:
        node_info['node_type'] = node.get('Node Type', '')
        node_info['table_name'] = node.get('Relation Name', '')
        node_info['filter'] = node.get('Filter', node.get('Index Cond', ''))
        node_info['plan_rows'] = node.get('Plan Rows', 0)
        node_info['total_cost'] = node.get('Total Cost', 0.0)
        node_info['index_name'] = node.get('Index Name', '')
        node_info['total_rows'] = get_total_rows(node_info['table_name']) if node_info['table_name'] else 0

        node_info['self_cost'] = summary.get('self_cost', 0.0)
        node_info['plan_depth'] = summary.get('depth', 0)

        # actual info
        node_info['actual_rows'] = node.get('Actual Rows', 0)
        node_info['actual_time'] = node.get('Actual Total Time', 0.0)
    elif is_join:

        node_info['node_type'] = node.get('Node Type', '')
        node_info['plan_rows'] = node.get('Plan Rows', 0)
        node_info['total_cost'] = node.get('Total Cost', 0.0)
        node_info['filter'] = node.get('Join Filter', node.get('Hash Cond', node.get('Merge Cond', '')))

        node_info['self_cost'] = summary.get('self_cost', 0.0)
        node_info['left_in'] = summary.get('left_in', '')
        node_info['right_in'] = summary.get('right_in', '')
        node_info['plan_depth'] = summary.get('depth', 0)

        # actual info
        node_info['actual_rows'] = node.get('Actual Rows', None)
        node_info['actual_time'] = node.get('Actual Total Time', None)

    node_info['key'] = f"{node_info['node_type']}_{node_info.get('table_name', '')}_{node_info.get('left_in', '')}_{node_info.get('right_in', '')}"
    node_info['label'] = summary.get('label', '')
    return node_info


node_list_jsonl = []
query_id_counter = 1
def collect_single_node(summary: Dict[str, Any]) -> None:
    """
    收集单个计划节点信息并以JSONL格式追加保存到指定文件

    Args:
        node: 单个计划节点(dict)
        summary: 计划摘要(dict)
    """
    node_info: Dict[str, Any] = get_single_node_info(summary)

    if not node_info:
        return
    if not node_info.get('actual_rows') or not node_info.get('actual_time'):
        return

    if node_info['key'] not in [n['key'] for n in node_list_jsonl]:
        node_list_jsonl.append(node_info)





def save_node_list_jsonl(output_path: str = "outputs/plan_node/train.jsonl") -> None:
    global query_id_counter
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 以JSONL格式追加写入
    with open(output_path, 'a') as f:
        for node_info in node_list_jsonl:
            # 添加query_id字段
            node_info['query_id'] = query_id_counter
            f.write(json.dumps(node_info, ensure_ascii=False) + '\n')

    print(f"已保存 {len(node_list_jsonl)} 个节点的信息到 {output_path}, query_id={query_id_counter}")
    # 递增query_id
    query_id_counter += 1
    # 清空缓存
    node_list_jsonl.clear()


# =========================

# Relevant node types to keep in the tree
RELEVANT_SCAN_TYPES = [
    'Seq Scan', 'Index Scan', 'Index Only Scan'
]
RELEVANT_JOIN_TYPES = [
    'Nested Loop', 'Hash Join', 'Merge Join'
]

def _is_relevant(node_type: str) -> bool:
    return node_type in RELEVANT_SCAN_TYPES or node_type in RELEVANT_JOIN_TYPES

def _collect_leaf_aliases(plan: Dict[str, Any]) -> List[str]:
    """Collect leaf scan aliases/relation names under a plan subtree."""
    nt = plan.get('Node Type', '')
    plans = plan.get('Plans', []) or []
    if nt in RELEVANT_SCAN_TYPES:
        alias = plan.get('Alias') or plan.get('Relation Name') or ''
        return [alias] if alias else []
    leaves = []
    for ch in plans:
        leaves.extend(_collect_leaf_aliases(ch))
    return leaves

def _make_summary(plan: Dict[str, Any], depth: int) -> Dict[str, Any]:
    """Build a summary dict compatible with get_single_node_info for encoder."""
    nt = plan.get('Node Type', '')
    summary: Dict[str, Any] = {
        'node_type': nt,
        'depth': depth,
        'plan_node': plan,
    }
    if nt in RELEVANT_SCAN_TYPES:
        alias = plan.get('Alias') or plan.get('Relation Name') or ''
        summary.update({
            'label': alias,
            'self_cost': plan.get('Total Cost', 0.0),
        })
    elif nt in RELEVANT_JOIN_TYPES:
        plans = plan.get('Plans', []) or []
        left_aliases = _collect_leaf_aliases(plans[0]) if len(plans) >= 1 else []
        right_aliases = _collect_leaf_aliases(plans[1]) if len(plans) >= 2 else []
        # Compose a readable label
        label = f"[{', '.join(sorted(left_aliases))}] {nt} [{', '.join(sorted(right_aliases))}]"
        # Approximate self_cost as total minus immediate children total costs
        left_total = plans[0].get('Total Cost', 0.0) if len(plans) >= 1 else 0.0
        right_total = plans[1].get('Total Cost', 0.0) if len(plans) >= 2 else 0.0
        self_cost = float(plan.get('Total Cost', 0.0)) - float(left_total + right_total)
        summary.update({
            'label': label,
            'left_in': left_aliases,
            'right_in': right_aliases,
            'self_cost': self_cost,
        })
    return summary

def _encode_node(summary: Dict[str, Any], alias2id: Dict[str, int], alias_embs, emb_dim: int, norm_stats: Optional[List[float]]):
    node_json = get_single_node_info(summary)
    if not node_json:
        return None
    vec = encode_node_features(node_json, alias2id, alias_embs, emb_dim=emb_dim, norm_stats=norm_stats)
    vec = np.nan_to_num(vec, nan=0.0)
    return vec.tolist()

def build_tree_from_plan(
    plan: Dict[str, Any],
    alias2id_path: str = "models/cardinality_bias/alias2id.json",
    alias_embs_path: str = "models/cardinality_bias/alias_embs.npy",
    norm_stats_path: Optional[str] = "models/cardinality_bias/norm_stats.npy",
    emb_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build one tree sample:
    {
      "features": List[List[float]]  # each 4D numeric feature vector,
      "children": List[List[int]]    # adjacency list: [left_id, right_id]; -1 if missing
    }
    - keep only scan/join nodes
    - skip other nodes, directly connect their relevant children to the parent
    - ensure root is index 0
    """
    # Load encoder artifacts
    with open(alias2id_path, 'r') as f:
        alias2id = json.load(f)
    alias_embs = None
    if os.path.exists(alias_embs_path):
        import numpy as _np
        alias_embs = _np.load(alias_embs_path)
        actual_emb_dim = int(alias_embs.shape[1])
    else:
        actual_emb_dim = int(emb_dim or 64)
    if emb_dim is None:
        emb_dim = actual_emb_dim

    norm_stats = None
    if norm_stats_path and os.path.exists(norm_stats_path):
        import numpy as _np
        ns = _np.load(norm_stats_path)
        # Expected order: self_cost_min, self_cost_max, plan_rows_min, plan_rows_max,
        # total_cost_min, total_cost_max, total_rows_min, total_rows_max
        norm_stats = [float(x) for x in ns.tolist()]

    features: List[List[float]] = []
    children: List[List[int]] = []

    def add_node(summary: Dict[str, Any]) -> int:
        feat = _encode_node(summary, alias2id, alias_embs, emb_dim, norm_stats)
        if feat is None:
            feat = [0.0, 0.0, 0.0, 0.0]
        idx = len(features)
        features.append(feat)
        children.append([-1, -1])
        return idx

    def visit(cur: Dict[str, Any], depth: int) -> List[int]:
        nt = cur.get('Node Type', '')
        plans = cur.get('Plans', []) or []
        # Recurse first to collect relevant child roots
        child_roots: List[int] = []
        for ch in plans:
            child_roots.extend(visit(ch, depth + 1))

        if _is_relevant(nt):
            summary = _make_summary(cur, depth)
            my_idx = add_node(summary)
            # Attach up to two children
            left_id = child_roots[0] if len(child_roots) >= 1 else -1
            right_id = child_roots[1] if len(child_roots) >= 2 else -1
            children[my_idx] = [left_id, right_id]
            return [my_idx]
        else:
            # Not kept: bubble up relevant child roots
            return child_roots

    roots = visit(plan, 0)
    if not roots:
        return {"features": [], "children": []}
    root_idx = roots[0]

    # Ensure root is index 0 by reindexing if needed
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

# =========================
# Extraction helpers for used tables, leading sequence, join/scan types
# =========================



def extract_plan_structures(plan: Dict[str, Any]) -> Dict[str, Any]:
    _, _, _, _, struct_info = summarize_plan_tree(plan)
    return struct_info

def build_tree_list_from_plans(
    plans: List[Dict[str, Any]],
    alias2id_path: str = "models/cardinality_bias/alias2id.json",
    alias_embs_path: str = "models/cardinality_bias/alias_embs.npy",
    norm_stats_path: Optional[str] = "models/cardinality_bias/norm_stats.npy",
    emb_dim: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build a list of trees for many plans."""
    trees = []
    for plan in plans:
        trees.append(build_tree_from_plan(plan, alias2id_path, alias_embs_path, norm_stats_path, emb_dim))
    return trees

def save_tree_dataset(trees: List[Dict[str, Any]], output_path: str = "outputs/judger/dataset-trees.json") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(trees, f, ensure_ascii=False)
    print(f"已保存 {len(trees)} 棵树到 {output_path}")

def make_pair_dataset_from_trees(
    tree_pairs: List[Tuple[Dict[str, Any], Dict[str, Any], int]],
    output_path: str = "outputs/judger/dataset.json",
) -> None:
    """Save pair dataset compatible with TreePairDataset in judger/train.py.
    Each item: {"tree1": {features, children}, "tree2": {...}, "label": int}
    """
    data = []
    for t1, t2, label in tree_pairs:
        data.append({"tree1": t1, "tree2": t2, "label": int(label)})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"已保存 {len(data)} 对树到 {output_path}")
