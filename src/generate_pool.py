import os
import re
import asyncio
import psycopg2 # 需要安装 psycopg2-binary
from pathlib import Path
from openai import OpenAI
from llm_provider import llm_provider
from postgresql import measure_sql_performance, hints_have_same_effect
from config import Config
from plan_summarizer import save_node_list_jsonl
import json


ROOT_DIR = Path(__file__).resolve().parents[1]

# ==== Prompt template paths (customize here) ====
SYSTEM_PROMPT_PATH = str(ROOT_DIR / "prompt" / "explore-system.prompt")
USER_PROMPT_ONE_SHOT_PATH = str(ROOT_DIR / "prompt" / "explore-user.prompt")
EXPLORE_GLOBAL_USER_PROMPT_PATH = str(ROOT_DIR / "prompt" / "explore-user-global.prompt")
EXPLORE_LEADING_USER_PROMPT_PATH = str(ROOT_DIR / "prompt" / "explore-user-leading.prompt")
EXPLORE_NODE_USER_PROMPT_PATH = str(ROOT_DIR / "prompt" / "explore-user-node.prompt")


# 数据库连接信息（通过 Config 读取 config/db.conf）
DB_CONFIG = {
    "dbname": Config.DB_DATABASE,
    "user": Config.DB_USER,
    "password": Config.DB_PASSWORD,
    "host": Config.DB_HOST,
    "port": Config.DB_PORT,
}

def _extract_tables_and_aliases(sql: str) -> dict[str, str]:
    """从SQL中解析出表名及其别名。支持多种表定义方式。"""
    aliases = {}
    
    # 清理SQL字符串，去除多余空格和换行
    sql_clean = ' '.join(sql.split())
    
    # 匹配 FROM 子句中的表定义（包括逗号分隔的）
    # 匹配格式: FROM table [AS] alias, table2 [AS] alias2
    from_pattern = re.compile(
        r'\bFROM\s+((?:[\w\.]+(?:\s+(?:AS\s+)?\w+)?\s*,\s*)*[\w\.]+(?:\s+(?:AS\s+)?\w+)?)',
        re.IGNORECASE
    )
    
    from_match = from_pattern.search(sql_clean)
    if from_match:
        from_clause = from_match.group(1)
        # 解析逗号分隔的表定义
        table_defs = re.split(r'\s*,\s*', from_clause)
        for table_def in table_defs:
            # 匹配 table [AS] alias
            table_alias_pattern = re.compile(
                r'([\w\.]+)(?:\s+(?:AS\s+)?(\w+))?',
                re.IGNORECASE
            )
            table_alias_match = table_alias_pattern.match(table_def.strip())
            if table_alias_match:
                table, alias = table_alias_match.groups()
                if alias:  # 如果有别名
                    table_name = table.split('.')[-1]
                    aliases[alias] = table_name
                # 如果没有别名，表名本身就是标识符
    
    # 匹配 JOIN 子句中的表定义
    join_pattern = re.compile(
        r'\b(?:INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|JOIN|CROSS\s+JOIN)\s+([\w\.]+)(?:\s+(?:AS\s+)?(\w+))?',
        re.IGNORECASE
    )
    
    join_matches = join_pattern.findall(sql_clean)
    for table, alias in join_matches:
        if alias:  # 如果有别名
            table_name = table.split('.')[-1]
            aliases[alias] = table_name
    
    return aliases

def get_query_statistics(sql: str) -> str:
    """为给定SQL中涉及的表和列获取统计信息和索引信息。"""
    aliases = _extract_tables_and_aliases(sql)
    if not aliases:
        return "Could not parse tables from SQL."

    stats_output = []
    
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                for alias, table_name in aliases.items():
                    stats_output.append(f"\n=== Table: {table_name} (alias: {alias}) ===")
                    
                    # 1. 获取表行数
                    cur.execute("""
                        SELECT reltuples::bigint
                        FROM pg_class
                        WHERE relname = %s;
                    """, (table_name,))
                    row_count = cur.fetchone()
                    
                    if row_count:
                        stats_output.append(f"  - Estimated Rows: {row_count[0]:,}")
                    else:
                        stats_output.append(f"  - Estimated Rows: Not found")

                    # 2. 获取 WHERE 条件中涉及的列
                    where_cols = set()
                    # 匹配 WHERE 子句中的列引用
                    where_pattern = re.compile(r'\bWHERE\b([\s\S]*?)(?:\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|\bHAVING\b|$)', re.IGNORECASE)
                    where_match = where_pattern.search(sql)
                    if where_match:
                        where_clause = where_match.group(1)
                        where_cols.update(re.findall(rf'\b{alias}\.(\w+)\b', where_clause, re.IGNORECASE))
                        where_cols.update(re.findall(r'\b(\w+)\s*[=<>!]', where_clause, re.IGNORECASE))

                    if where_cols:
                        stats_output.append(f"  - Index Information (for WHERE columns):")
                        # 获取该表的所有索引
                        cur.execute("""
                            SELECT
                                i.relname as index_name,
                                am.amname as index_type,
                                array_to_string(array_agg(a.attname ORDER BY x.attnum), ', ') as columns,
                                idx.indisunique as is_unique,
                                idx.indisprimary as is_primary
                            FROM
                                pg_index idx
                                JOIN pg_class i ON i.oid = idx.indexrelid
                                JOIN pg_class t ON t.oid = idx.indrelid
                                JOIN pg_namespace n ON n.oid = t.relnamespace
                                JOIN pg_am am ON i.relam = am.oid
                                JOIN pg_attribute a ON a.attrelid = t.oid
                                JOIN unnest(idx.indkey) WITH ORDINALITY AS x(attnum, ord) ON a.attnum = x.attnum
                            WHERE
                                t.relname = %s AND n.nspname = 'public'
                            GROUP BY
                                i.relname, am.amname, idx.indisunique, idx.indisprimary
                            ORDER BY
                                i.relname;
                        """, (table_name,))

                        indexes = cur.fetchall()
                        if indexes:
                            for index_name, index_type, columns, is_unique, is_primary in indexes:
                                # 检查索引是否包含 WHERE 条件中的列
                                index_cols = [col.strip() for col in columns.split(',')]
                                relevant_cols = [col for col in index_cols if col in where_cols]
                                if relevant_cols:
                                    index_type_str = f"{index_type.upper()}"
                                    if is_primary:
                                        index_type_str += " (PRIMARY KEY)"
                                    elif is_unique:
                                        index_type_str += " (UNIQUE)"
                                    stats_output.append(f"    * {index_name}: {index_type_str} on ({columns})")
                            # 如果没有相关索引
                            if not any(any(col in where_cols for col in idx[2].split(', ')) for idx in indexes):
                                stats_output.append(f"    * No indexes found for WHERE columns")
                        else:
                            stats_output.append(f"    * No indexes found for WHERE columns")

                    # 3. 找出SQL中与此表相关的列并获取统计信息
                    related_cols = set(re.findall(rf'\b{alias}\.(\w+)\b', sql, re.IGNORECASE))
                    
                    if related_cols:
                        stats_output.append(f"  - Column Statistics:")
                        for col_name in related_cols:
                            cur.execute("""
                                SELECT n_distinct, most_common_vals, most_common_freqs, histogram_bounds
                                FROM pg_stats
                                WHERE tablename = %s AND attname = %s;
                            """, (table_name, col_name))
                            col_stats = cur.fetchone()
                            if col_stats:
                                n_distinct, common_vals, common_freqs, histogram_bounds = col_stats
                                stats_output.append(f"    * Column: {col_name}")
                                
                                if n_distinct > 0:
                                    stats_output.append(f"      - Distinct Values: ~{int(n_distinct)}")
                                elif n_distinct < 0 and row_count:
                                    distinct_estimate = int(-n_distinct * row_count[0])
                                    stats_output.append(f"      - Distinct Values: ~{distinct_estimate:,} (Proportional)")

                                if common_vals:
                                    # 显示前5个最常见值
                                    vals = common_vals.strip('{}').split(',')
                                    freqs = common_freqs
                                    top_values = []
                                    for i, (val, freq) in enumerate(zip(vals[:5], freqs[:5])):
                                        top_values.append(f"'{val}' ({freq:.1%})")
                                    stats_output.append(f"      - Top Values: {', '.join(top_values)}")

                                if histogram_bounds:
                                    bounds = histogram_bounds.strip('{}').split(',')
                                    if bounds:
                                        stats_output.append(f"      - Value Range: {bounds[0]} to {bounds[-1]}")
                            else:
                                stats_output.append(f"    * Column: {col_name} - No statistics available")

    except Exception as e:
        return f"Error fetching statistics: {e}"
    return "\n".join(stats_output)


import sqlglot
from sqlglot import exp
from collections import defaultdict

def extract_join_adjacency(sql_text):
    try:
        parsed = sqlglot.parse_one(sql_text)
    except Exception as e:
        return f"Error parsing SQL: {str(e)}"

    adjacency = defaultdict(set)
    
    for node in parsed.find_all(exp.EQ): 
        
        left = node.left
        right = node.right

        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            table_a = left.table
            table_b = right.table

            if table_a and table_b and table_a != table_b:
                adjacency[table_a].add(table_b)
                adjacency[table_b].add(table_a)

    output_lines = []
    for table in sorted(adjacency.keys()):
        neighbors = sorted(list(adjacency[table]))
        neighbors_str = ", ".join(neighbors)
        output_lines.append(f"{table}: [{neighbors_str}]")

    if not output_lines:
        return "No explicit join conditions detected."

    return "\n".join(output_lines)

# def generate_initial_user_prompt(sql: str, execution_plan: str, execution_time: float, statistics: str) -> str:
#     prompt_path = Path(USER_PROMPT_FIRST_PATH)
#     with open(prompt_path, 'r') as f:
#         prompt = f.read()
#     prompt = prompt.replace("{{statistics}}", statistics)
#     prompt = prompt.replace("{{sql}}", sql)
#     prompt = prompt.replace("{{execution_plan_before_hint}}", execution_plan)
#     prompt = prompt.replace("{{execution_time_before_hint}}", f"{execution_time:.2f}")
#     return prompt


def generate_global_user_prompt(sql: str, baseline_execution_plan: str, baseline_execution_time: float) -> str:
    """生成 global hint 优化阶段的 user prompt

    占位符: {{sql}}, {{baseline_execution_plan}}, {{baseline_execution_time}}
    """
    prompt_path = Path(EXPLORE_GLOBAL_USER_PROMPT_PATH)
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    prompt = prompt.replace("{{sql}}", sql)
    prompt = prompt.replace("{{baseline_execution_plan}}", baseline_execution_plan)
    prompt = prompt.replace("{{baseline_execution_time}}", f"{baseline_execution_time:.2f}")
    return prompt


def generate_leading_user_prompt(
    sql: str,
    statistics: str,
    active_global_hints: str,
    current_execution_plan: str,
    current_execution_time: float,
    leading_history_records: list,
    join_graph_adjacency, forbidden_prefixes
) -> str:
    """生成 leading hint 优化阶段的 user prompt

    占位符: {{statistics}}, {{sql}}, {{active_global_hints}},
            {{current_execution_plan}}, {{current_execution_time}},
            {{leading_history_records_xml}}
    """
    prompt_path = Path(EXPLORE_LEADING_USER_PROMPT_PATH)
    with open(prompt_path, 'r') as f:
        prompt = f.read()

    # 生成 leading_history_records_xml
    history_xml = "<leading_history_records>"
    for idx, record in enumerate(leading_history_records, start=1):
        history_xml += f"""
        <round index="{idx}">
            <used_hint>
        {record["used_hint"]}
            </used_hint>
            <execution_plan>
        {record["execution_plan"]}
            </execution_plan>
            <execution_time>{record["execution_time"]:.2f}</execution_time>
        </round>"""
    history_xml += "\n</leading_history_records>"
    
    prompt = prompt.replace("{{statistics}}", statistics)
    prompt = prompt.replace("{{sql}}", sql)
    prompt = prompt.replace("{{active_global_hints}}", active_global_hints)
    prompt = prompt.replace("{{current_execution_plan}}", current_execution_plan)
    prompt = prompt.replace("{{current_execution_time}}", f"{current_execution_time:.2f}")
    prompt = prompt.replace("{{leading_history_records_xml}}", history_xml)
    prompt = prompt.replace("{{join_graph_adjacency}}", join_graph_adjacency)
    prompt = prompt.replace("{{forbidden_prefixes}}", forbidden_prefixes)
    return prompt


def generate_node_user_prompt(
    sql: str,
    statistics: str,
    current_execution_plan: str,
    current_execution_time: float,
    blacklist_hints: list
) -> str:
    """生成 node/local hint 优化阶段的 user prompt

    占位符: {{statistics}}, {{sql}}, {{active_global_hints}}, {{active_leading_hint}},
            {{current_execution_plan}}, {{current_execution_time}}, {{baseline_join_order}},
            {{local_history_records_xml}}, {{duplicate_hints_xml}}
    """
    prompt_path = Path(EXPLORE_NODE_USER_PROMPT_PATH)
    with open(prompt_path, 'r') as f:
        prompt = f.read()

    prompt = prompt.replace("{{statistics}}", statistics)
    prompt = prompt.replace("{{sql}}", sql)
    prompt = prompt.replace("{{current_execution_plan}}", current_execution_plan)
    prompt = prompt.replace("{{current_execution_time}}", f"{current_execution_time:.2f}")
    from tool import generate_candidate_node_slot
    prompt = prompt.replace("{{optimizable_slots}}", generate_candidate_node_slot(current_execution_plan))
    prompt = prompt.replace("{{blacklist_hints}}", "\n".join(blacklist_hints))
    return prompt


def combine_hints(hint1: str, hint2: str) -> str:
    """简单拼接所有hint成一个组合hint"""
    hints = []
    if hint1 and hint1.strip() != "/*+ */":
        # 移除 /*+ 和 */ 后拼接
        h1 = hint1.replace("/*+", "").replace("*/", "").strip()
        hints.append(h1)
    if hint2 and hint2.strip() != "/*+ */":
        h2 = hint2.replace("/*+", "").replace("*/", "").strip()
        hints.append(h2)    

    if not hints:
        return "/*+ */"
    return f"/*+ {' '.join(hints)} */"


def save_iteration_output(output_path: Path, iteration_idx: int, hint: str, raw_plan: dict, plan_summary: str, prompt: str) -> None:
    """保存单次迭代的输出到指定目录"""
    iter_dir = output_path / str(iteration_idx)
    iter_dir.mkdir(parents=True, exist_ok=True)
    with open(iter_dir / "plan.json", "w", encoding="utf-8") as f:
        json.dump(raw_plan, f, indent=2)
    with open(iter_dir / "hint.txt", "w", encoding="utf-8") as f:
        f.write(hint)
    with open(iter_dir / "plan-summary.txt", "w", encoding="utf-8") as f:
        f.write(plan_summary)
    with open(iter_dir / "prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)



async def generate_hint_with_retry_and_response(sql: str, system_prompt: str, user_prompt: str,
                                    base_hints: str, explored_hint: set, max_retries: int = 3) -> tuple[str, str]:
    
    for retry in range(max_retries):
        response = await llm_provider.generate([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        matches = re.findall(r'(/\*\+[\s\S]*?\*/)', response)
        response_hint = matches[-1].strip() if matches else ""

        if len(response_hint) < 10:
            return response_hint, response

        is_duplicate = False
        for explored_h in explored_hint:
            new_combined = combine_hints(base_hints, response_hint)
            old_combined = combine_hints(base_hints, explored_h)
            if hints_have_same_effect(sql, new_combined, old_combined):
                print(f"Hint {new_combined} produces same plan as already explored hint: {old_combined}")
                is_duplicate = True

        if not is_duplicate:
            return response_hint, response
        else:
            if retry < max_retries - 1:
                print(f"Duplicate hint found. Retrying...")
            else:
                print(f"Max retries reached. Returning duplicate hint without re-executing.")

    return response_hint, f"[DUPLICATE] {response}"



async def optimize_global_hints(sql: str, baseline_plan: str, baseline_time: float,
                                 system_prompt: str, output_path: Path = None) -> tuple[str, str, float]:
    """Stage 1: Global hint
    """
    print("=== Stage 1: Global Hint Optimization ===")
    global_infer_reasoning = ""
  
    user_prompt = generate_global_user_prompt(sql, baseline_plan, baseline_time)
    

    response = await llm_provider.generate([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    matches = re.findall(r'(/\*\+[\s\S]*?\*/)', response)
    response_hint = matches[-1].strip() if matches else "/*+ */"

    if len(response_hint) < 10:
        print(f"Empty hint from Global stage. Using empty hint.")
        best_hint = "/*+ */"
        best_plan = baseline_plan
        best_time = baseline_time
        raw_plan = {}
    else:
        print(f"Response Hint: {response_hint}")

        current_plan, current_time, raw_plan = measure_sql_performance(response_hint + " " + sql)
        print(f"Execution with hint: {current_time:.2f} ms")

        if current_time < baseline_time:
            print(f"Global hint improved performance! New best time: {current_time:.2f} ms (was {baseline_time:.2f} ms)")
            best_hint = response_hint
            best_plan = current_plan
            best_time = current_time
            global_infer_reasoning = response
        else:
            print(f"No improvement with global hint. Keeping baseline.")
            best_hint = "/*+ */"
            best_plan = baseline_plan
            best_time = baseline_time
            global_infer_reasoning = "Skip"

    if output_path:
        save_iteration_output(output_path, 1, response_hint, raw_plan, best_plan, user_prompt)

    print(f"Stage 1 complete. Best hint: {best_hint}, Best time: {best_time:.2f} ms")
    print("----------------------------------------------------")

    return best_hint, best_plan, best_time, global_infer_reasoning


async def optimize_leading_hints(sql: str, statistics: str, active_global_hints: str, join_graph_adjacency: str,
                                 current_plan: str, current_time: float,
                                 system_prompt: str, num_iterations: int,
                                 output_path: Path = None) -> tuple[str, str, float, str]:
    """Stage 2: Leading hint 优化 - 多轮迭代

    Returns:
        (best_hint, best_plan, best_time, infer_reason)
    """
    print(f"=== Stage 2: Leading Hint Optimization ({num_iterations} iterations) ===")

    # 初始化
    history_records = [{
        "used_hint": "/*+ */",
        "execution_plan": current_plan,
        "execution_time": current_time
    }]
    explored_hint = {"/*+ */"}
    explored_results = {"/*+ */": (current_plan, current_time, {})}

    best_hint = "/*+ */"
    best_plan = current_plan
    best_time = current_time
    leading_infer_reasoning = "Skip"
    forbidden_prefixes = ""

    for i in range(num_iterations):
        iter_idx = 2 + i  # 输出目录索引从2开始
        print(f"--- Leading Iteration {i + 1}/{num_iterations} ---")

        # 生成prompt
        user_prompt = generate_leading_user_prompt(
            sql, statistics, active_global_hints,
            best_plan, best_time, history_records, join_graph_adjacency, forbidden_prefixes)

        response_hint, response = await generate_hint_with_retry_and_response(
            sql, system_prompt, user_prompt, active_global_hints, explored_hint, max_retries=3)

        if len(response_hint) < 10:
            print(f"Empty hint from leading iteration {i + 1}. Stopping leading optimization.")
            break

        print(f"Response Hint: {response_hint}")
        forbidden_prefixes = forbidden_prefixes + "\n" + response_hint.split("/*+")[1].split("*/")[0].strip()

        is_from_cache = response.startswith("[DUPLICATE]")
        if is_from_cache:
            duplicate_h = None
            for explored_h in explored_hint:
                combine_new = combine_hints(active_global_hints, response_hint)
                combine_old = combine_hints(active_global_hints, explored_h)
                if hints_have_same_effect(sql, combine_new, combine_old):
                    print(f"Hint {combine_new} produces same plan as already explored hint: {combine_old}")
                    duplicate_h = explored_h
                    break
            if duplicate_h and duplicate_h in explored_results:
                print(f"Using cached result {duplicate_h} for duplicate hint: {response_hint}")
                new_plan, new_time, raw_plan = explored_results[duplicate_h]
            else:
                # 回退到执行
                combined_hint = combine_hints(active_global_hints, response_hint)
                new_plan, new_time, raw_plan = measure_sql_performance(combined_hint + " " + sql)
        else:

            combined_hint = combine_hints(active_global_hints, response_hint)
            new_plan, new_time, raw_plan = measure_sql_performance(combined_hint + " " + sql)
            # 缓存结果
            explored_results[response_hint] = (new_plan, new_time, raw_plan)

            print(f"Execution with hint {combined_hint}: {new_time:.2f} ms")

        # 评估
        if new_time < best_time:
            print(f"Improvement found! New best time: {new_time:.2f} ms (was {best_time:.2f} ms)")
            best_hint = response_hint
            best_plan = new_plan
            best_time = new_time
            leading_infer_reasoning = response  # 保存最佳hint的推理原因

        # 记录历史
        history_records.append({
            "used_hint": response_hint,
            "execution_plan": new_plan,
            "execution_time": new_time
        })
        explored_hint.add(response_hint)

        # 保存输出
        if output_path:
            save_iteration_output(output_path, iter_idx, response_hint, raw_plan, new_plan, user_prompt)

    print(f"Stage 2 complete. Best hint: {best_hint}, Best time: {best_time:.2f} ms")
    print("----------------------------------------------------")

    return best_hint, best_plan, best_time, leading_infer_reasoning


async def optimize_node_hints(sql: str, statistics: str, active_global_hints: str, active_leading_hint: str,
                              current_plan: str, current_time: float, baseline_join_order: str,
                              system_prompt: str, num_iterations: int,
                              output_path: Path = None) -> tuple[str, str, float, str]:
    """Stage 3: Node hint 优化 - 多轮迭代

    Returns:
        (best_hint, best_plan, best_time, infer_reason)
    """
    print(f"=== Stage 3: Node Hint Optimization ({num_iterations} iterations) ===")

    # 初始化 - 不需要 history_records，改用 blacklist
    explored_hint = {"/*+ */"}
    explored_results = {"/*+ */": (current_plan, current_time, {})}  # 缓存执行结果
    blacklist = ["/*+ */"]  # 记录无效的 hints

    best_hint = "/*+ */"
    best_plan = current_plan
    best_time = current_time
    node_infer_reasoning = "Skip"

    # 组合 base_hints: global + leading
    base_hints = combine_hints(active_global_hints, active_leading_hint)

    for i in range(num_iterations):
        iter_idx = 5 + i  # 输出目录索引从5开始
        print(f"--- Node Iteration {i + 1}/{num_iterations} ---")

        user_prompt = generate_node_user_prompt(
            sql, statistics, best_plan, best_time, blacklist)

        response_hint, response = await generate_hint_with_retry_and_response(
            sql, system_prompt, user_prompt, base_hints, explored_hint, max_retries=3)

        if len(response_hint) < 10:
            print(f"Empty hint from node iteration {i + 1}. Stopping node optimization.")
            break

        print(f"Response Hint: {response_hint}")

        is_from_cache = response.startswith("[DUPLICATE]")
        if is_from_cache:
            duplicate_h = None
            for explored_h in explored_hint:
                combined_new = combine_hints(base_hints, response_hint)
                combined_old = combine_hints(base_hints, explored_h)
                if hints_have_same_effect(sql, combined_new, combined_old):
                    duplicate_h = explored_h
                    print(f"Hint {combined_new} produces same plan as already explored hint: {combined_old}")
                    break
            if duplicate_h and duplicate_h in explored_results:
                print(f"Using cached result {duplicate_h} for duplicate hint: {response_hint}")
                new_plan, new_time, raw_plan = explored_results[duplicate_h]
            else:
                # 回退到执行
                combined_hint = combine_hints(base_hints, response_hint)
                new_plan, new_time, raw_plan = measure_sql_performance(combined_hint + " " + sql)
        else:
            combined_hint = combine_hints(base_hints, response_hint)
            new_plan, new_time, raw_plan = measure_sql_performance(combined_hint + " " + sql)
            explored_results[response_hint] = (new_plan, new_time, raw_plan)

            print(f"Execution with hint {combined_hint}: {new_time:.2f} ms")

        # 评估
        if new_time < best_time:
            print(f"Improvement found! New best time: {new_time:.2f} ms (was {best_time:.2f} ms)")
            best_hint = response_hint
            best_plan = new_plan
            best_time = new_time
            if node_infer_reasoning == "Skip":
                node_infer_reasoning = response
            else:
                node_infer_reasoning += f"\n{response}"
        else:
            # 没改善：记录到 blacklist
            print(f"No improvement. Adding to blacklist: {response_hint}")
            blacklist.append(response_hint)

        explored_hint.add(response_hint)

        # 保存输出
        if output_path:
            save_iteration_output(output_path, iter_idx, response_hint, raw_plan, new_plan, user_prompt)

    print(f"Stage 3 complete. Best hint: {best_hint}, Best time: {best_time:.2f} ms")
    print("----------------------------------------------------")

    return best_hint, best_plan, best_time, node_infer_reasoning


async def get_best_pg_hint_plan(sql: str, max_iterations: int = 7, output_path: Path = None) -> tuple:
    """
    - Stage 1 (Global): 1 iteration
    - Stage 2 (Leading): (max_iterations - 1) // 2 iterations (默认3)
    - Stage 3 (Node): 剩余 iterations (默认3)
    """
    print(f"Optimizing SQL: {sql[:100].strip()}...")

    # === 初始化 ===
    # 测量 baseline 性能
    baseline_plan, baseline_time, raw_baseline_plan = measure_sql_performance(sql)

    # 保存 baseline 到 output_path/0/
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "0").mkdir(parents=True, exist_ok=True)
        with open(output_path / "0" / "plan.json", "w", encoding="utf-8") as f:
            json.dump(raw_baseline_plan, f, indent=2)
        with open(output_path / "0" / "plan-summary.txt", "w", encoding="utf-8") as f:
            f.write(baseline_plan)

    print(f"Baseline Performance: {baseline_time:.2f} ms")

    # 获取数据库统计信息
    print("Fetching database statistics...")
    statistics = get_query_statistics(sql)
    print("----------------------------------------------------")

    # 读取 system prompt
    with open(SYSTEM_PROMPT_PATH, 'r') as f:
        system_prompt = f.read()

    # 计算各阶段迭代数
    global_iters = 1
    leading_iters = (max_iterations - 1) // 2
    node_iters = max_iterations - 1 - leading_iters

    print(f"Iteration distribution: Global={global_iters}, Leading={leading_iters}, Node={node_iters}")
    print("====================================================")

    # === Stage 1: Global Hint 优化 ===
    best_global_hint, global_plan, global_time, global_infer_reasoning = await optimize_global_hints(
        sql, baseline_plan, baseline_time, system_prompt, output_path
    )
    

    # === Stage 2: Leading Hint 优化 ===
    # 从 global 阶段最佳计划中提取 join_order
    global_join_order_match = re.search(r'"join_order":\s*"([^"]*(?:\\.[^"]*)*)"', global_plan)
    baseline_join_order = global_join_order_match.group(1).replace(r'\n', '\n').replace(r'\t', '\t').replace(r'\\"', '"').replace(r'\\', '\\') if global_join_order_match else ""
    join_graph_adjacency = extract_join_adjacency(sql)
    best_leading_hint, leading_plan, leading_time, leading_infer_reasoning = await optimize_leading_hints(
        sql, statistics, best_global_hint, join_graph_adjacency,
        global_plan, global_time,
        system_prompt, leading_iters, output_path
    )

    # === Stage 3: Node Hint 优化 ===
    # 从 leading 阶段最佳计划中提取 join_order 作为 baseline_join_order
    leading_join_order_match = re.search(r'"join_order":\s*"([^"]*(?:\\.[^"]*)*)"', leading_plan)
    baseline_join_order = leading_join_order_match.group(1).replace(r'\n', '\n').replace(r'\t', '\t').replace(r'\\"', '"').replace(r'\\', '\\') if leading_join_order_match else ""

    best_node_hint, node_plan, node_time, node_infer_reasoning = await optimize_node_hints(
        sql, statistics, best_global_hint, best_leading_hint,
        leading_plan, leading_time, baseline_join_order,
        system_prompt, node_iters, output_path
    )

    # === 组合最终 hint ===
    final_hint = combine_hints(combine_hints(best_global_hint, best_leading_hint), best_node_hint)
    final_plan = node_plan
    final_time = node_time

    # === 拼接三个阶段的推理原因 ===
    combined_infer_reason = f"""=== Stage 1: Global Hint ===
{global_infer_reasoning}

=== Stage 2: Leading Hint ===
{leading_infer_reasoning}

=== Stage 3: Node Hint ===
{node_infer_reasoning}
"""

    print("====================================================")
    print(f"Three-Stage Optimization Complete!")
    print(f"  Stage 1 (Global): {best_global_hint} -> {global_time:.2f} ms")
    print(f"  Stage 2 (Leading): {best_leading_hint} -> {leading_time:.2f} ms")
    print(f"  Stage 3 (Node): {best_node_hint} -> {node_time:.2f} ms")
    print(f"  Final Combined: {final_hint}")
    print(f"  Final Time: {final_time:.2f} ms (Baseline: {baseline_time:.2f} ms)")
    print(f"  Speedup: {(baseline_time / final_time if final_time > 0 else 0):.2f}x")
    print("====================================================")

    # 保存各阶段最佳 hint 和最终组合 hint
    if output_path:
        (output_path / "best_global_hint.txt").write_text(best_global_hint)
        (output_path / "best_leading_hint.txt").write_text(best_leading_hint)
        (output_path / "best_node_hint.txt").write_text(best_node_hint)
        (output_path / "final_combined_hint.txt").write_text(final_hint)
        (output_path / "infer_reason.txt").write_text(combined_infer_reason)

    save_node_list_jsonl()

    return baseline_plan, final_hint, final_plan, final_time, baseline_time, combined_infer_reason, []


async def generate_pool(train_query_dir: str, pool_dir: str, max_iterations: int = 7) -> None:

    if not os.path.exists(pool_dir):
        os.makedirs(pool_dir)

    sql_files = [f for f in os.listdir(train_query_dir) if f.endswith(".sql")]
    sql_pools = {}
    for file in sql_files:
        filepath = os.path.join(train_query_dir, file)
        print(f"\n===== Processing SQL file: {filepath} =====")
        with open(filepath, 'r') as f:
            sql = f.read()
            if not sql.strip():
                print(f"Warning: SQL file {file} is empty. Skipping.")
                continue
                        
            
            filename = Path(file).stem
            output_path = Path(pool_dir) / filename
            output_path.mkdir(exist_ok=True)

            baseline_plan, suggest_hint, execution_plan, best_time, baseline_time, infer_reason, regression_analysis = await get_best_pg_hint_plan(sql, max_iterations, output_path)


            sql_pools[filename] = {"best_time": best_time, "baseline_time": baseline_time}
            (output_path / "suggest_hint.txt").write_text(suggest_hint)
            (output_path / "execution_plan.txt").write_text(execution_plan)
            (output_path / "original_execution_plan.txt").write_text(baseline_plan)
            (output_path / "query.sql").write_text(sql)
            (output_path / "infer_reason.txt").write_text(infer_reason)
            import json
            (output_path / "regression_analysis.json").write_text(json.dumps(regression_analysis, ensure_ascii=False, indent=2))

    print("\n===== Summary of Best Times =====")
    print(sql_pools) ## write to log.csv
    with open(os.path.join(pool_dir, "log-new1.csv"), 'w') as f:
        f.write("filename,best_time_ms,baseline_time_ms,speedup\n")
        # sort by key
        for k in sorted(sql_pools.keys()):
            best_time = sql_pools[k]["best_time"]
            baseline_time = sql_pools[k]["baseline_time"]
            speedup = baseline_time / best_time if best_time > 0 else 0
            f.write(f"{k},{best_time:.2f},{baseline_time:.2f},{speedup:.2f}\n")
    print("=================================")

if __name__ == "__main__":
    import asyncio
    asyncio.run(generate_pool(
        train_query_dir=str(ROOT_DIR / "data" / "train-query"),
        pool_dir=str(ROOT_DIR / "outputs" / "demo_pool"),
        max_iterations=7,
    ))
