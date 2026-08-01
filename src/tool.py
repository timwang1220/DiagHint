


import re


def generate_candidate_node_slot(plan_summary: str) -> str:
    """从 plan_summary 中解析出候选的节点 slot

    解析规则：
    - JOIN slot: 从 description 解析出 prefix 和 target，格式如 "[tables] JoinType [target]"
    - SCAN slot: 从 description 解析出 table 和 scan type，格式如 "ScanType on table"

    返回格式：
    - join_slot_X: prefix=[...], pair=(target), baseline=JoinType
    - scan_slot_X: table=xxx, baseline=ScanType
    """
    candidate_slots = []

    # 解析 JOIN nodes (B1 部分)
    # 使用正则表达式直接提取每个节点的 description 和 join_type
    join_section = re.search(r'\(B1\) Top-qerror JOIN nodes\s*\n\s*\[(.*?)\]\s*\n\s*\(B2\)', plan_summary, re.DOTALL)
    if not join_section:
        join_section = re.search(r'\(B1\) Top-qerror JOIN nodes\s*\n\s*\[(.*?)\]\s*$', plan_summary, re.DOTALL)

    if join_section:
        section_text = join_section.group(1)
        # 提取所有 description 字段
        descriptions = re.findall(r'"description":\s*"\[([^\]]+)\]\s+(\w+)\s+\[(\w+)\]"', section_text)
        # 提取所有 join_type 字段
        join_types = re.findall(r'"join_type":\s*"([^"]+)"', section_text)

        for idx, (prefix, join_method, target) in enumerate(descriptions, start=1):
            # join_method 可能是 "HashJoin" 或 "Hash Join"，需要标准化
            # 从 join_types 数组获取对应的 join_type
            if idx - 1 < len(join_types):
                join_type = join_types[idx - 1]
                join_baseline = join_type.replace(" ", "")  # "Hash Join" -> "HashJoin"
            else:
                join_baseline = join_method.replace(" ", "")

            slot = f"join_slot_{idx}: prefix=[{prefix}], pair=({target}), baseline={join_baseline}"
            candidate_slots.append(slot)

    # 解析 SCAN nodes (B2 部分)
    scan_section = re.search(r'\(B2\) Top-qerror SCAN nodes\s*\n\s*\[(.*?)\]$', plan_summary, re.DOTALL)
    if scan_section:
        section_text = scan_section.group(1)
        # 提取所有 description 字段
        # 格式: "description": "Seq Scan on cn"
        scan_descriptions = re.findall(r'"description":\s*"(\w+(?:\s+\w+)?)\s+on\s+(\w+)"', section_text)

        for idx, (scan_type, table) in enumerate(scan_descriptions, start=1):
            scan_baseline = scan_type.replace(" ", "")  # "Seq Scan" -> "SeqScan"
            slot = f"scan_slot_{idx}: table={table}, baseline={scan_baseline}"
            candidate_slots.append(slot)

    return "\n".join(candidate_slots)
