import time
import subprocess
from pathlib import Path
import sys
import os
from typing import Tuple, List, Dict, Optional
import statistics # 用于计算平均值

import psycopg2
from psycopg2 import sql as psycopg2_sql
# 导入特定异常类型，用于捕获超时错误
from psycopg2.errors import QueryCanceled
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from plan_summarizer import *

sys.path.append(str(Path(__file__).parent.parent))  # 将项目根目录添加到路径
from config import Config

# from plan2list import extract_nodes, execution_plan_to_json



def measure_sql_performance(sql: str, num_runs: int = 1, timeout_seconds: int = 600, optional_set: Optional[str] = None, raw_plan: Optional[bool] = False) -> Tuple[List[Dict], float, Dict]:
    """
    执行SQL并返回 join/scan 节点列表形式的执行计划，以及平均运行时间。
    """
    timings = []
    execution_plan_nodes: List[Dict] = []
    timeout_ms = timeout_seconds * 1000
    
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
            cursor.execute("LOAD 'pg_hint_plan';")

            for i in range(num_runs):
                try:
                    cursor.execute("DISCARD ALL;")
                    cursor.execute(f"SET statement_timeout = {timeout_ms};")
                    if optional_set:
                        cursor.execute(optional_set)
                    # 使用 JSON 格式输出
                    explain_query = psycopg2_sql.SQL("EXPLAIN (ANALYZE, FORMAT JSON) {}").format(
                        psycopg2_sql.SQL(sql)
                    )
                    
                    start_time = time.perf_counter()
                    cursor.execute(explain_query)
                    explain_result = cursor.fetchall()[0][0][0]
                    # 处理plan结构，当第一项是bao时的情况
                    if isinstance(explain_result, dict) and "Plan" in explain_result:
                        plan_json = explain_result["Plan"]
                    else:
                        plan_json = explain_result
                    end_time = time.perf_counter()
                    
                    duration_ms = (end_time - start_time) * 1000
                    timings.append(duration_ms)            

                except QueryCanceled: # 超时返回explain plan 和 超时阈值（ms）
                    print(f"第 {i + 1} 次运行超时 (超过 {timeout_seconds} 秒)，按 {timeout_ms} ms 计算。")
                    plan, raw_plan = get_sql_base_explain_plan(sql)
                    return plan, timeout_ms, raw_plan
                
                finally:
                    cursor.execute("SET statement_timeout = 0;")

        if not timings:
            return [], 0.0, plan_json
        # 取最后一次执行时间作为平均时间
        average_time_ms = timings[-1]
        return plan_json_to_text(plan_json, runtime_ms=average_time_ms), average_time_ms, plan_json

    except psycopg2.Error as e:
        if not isinstance(e, QueryCanceled):
             raise RuntimeError(f"数据库操作失败: {e}") from e
    finally:
        if conn:
            conn.close()


def get_sql_base_explain_plan(sql: str, summarize: bool = True) -> List[Dict]:
    """
    get the base explain plan of the sql in list form (explain but not explain analyze)
    """
    execution_plan_nodes: List[Dict] = []
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
                try:
                    explain_query = psycopg2_sql.SQL("EXPLAIN (FORMAT JSON) {}").format(
                        psycopg2_sql.SQL(sql)
                    )
                    cursor.execute("LOAD 'pg_hint_plan';")
                    cursor.execute(explain_query)
                    explain_result = cursor.fetchall()[0][0][0]
                    # 处理plan结构，当第一项是bao时的情况
                    if isinstance(explain_result, dict) and "Plan" in explain_result:
                        plan_json = explain_result["Plan"]
                    else:
                        plan_json = explain_result

                except psycopg2.Error as e:
                    raise RuntimeError(f"数据库操作失败: {e}") from e
        if summarize:
            return plan_json_to_text(plan_json), plan_json
        return "", plan_json
    except psycopg2.Error as e:
        raise RuntimeError(f"数据库连接失败: {e}") from e
    finally:
        if conn:
            conn.close()


def _get_hint_total_cost(
    sql: str,
    hint: str,
    cost_cache: Optional[Dict[Tuple[str, str], Optional[float]]] = None,
) -> Optional[float]:
    key = (sql, (hint or "").strip())
    if cost_cache is not None and key in cost_cache:
        return cost_cache[key]
    try:
        sql_with_hint = f"{hint} {sql}".strip()
        _, raw_plan = get_sql_base_explain_plan(sql_with_hint, summarize=False)
        cost = raw_plan.get("Total Cost", None) if isinstance(raw_plan, dict) else None
        cost_val = float(cost) if cost is not None else None
    except Exception as e:
        print(f"Error getting hint total cost: {e}")
        cost_val = None
    if cost_cache is not None:
        cost_cache[key] = cost_val
    return cost_val


def hints_have_same_effect(
    sql: str,
    hint1: str,
    hint2: str,
    cost_cache: Optional[Dict[Tuple[str, str], Optional[float]]] = None,
) -> bool:
    try:
        cost1 = _get_hint_total_cost(sql, hint1, cost_cache=cost_cache)
        cost2 = _get_hint_total_cost(sql, hint2, cost_cache=cost_cache)

        if cost1 is None or cost2 is None:
            return False

        return cost1 == cost2
    except Exception as e:
        print(f"Error comparing hints: {e}")
        return False


def only_execute(sql: str, timeout_seconds: int = 600) -> float:
    """
    只执行SQL（不做EXPLAIN ANALYZE），返回执行时间（毫秒）

    Args:
        sql: SQL语句（可能包含pg_hint_plan hints）
        timeout_seconds: 超时时间（秒）

    Returns:
        执行时间（毫秒）
    """
    timeout_ms = timeout_seconds * 1000

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
            # 加载 pg_hint_plan 扩展
            cursor.execute("LOAD 'pg_hint_plan';")

            # 设置超时
            cursor.execute(f"SET statement_timeout = {timeout_ms};")

            # 执行SQL
            start_time = time.perf_counter()
            cursor.execute(sql)
            # 获取所有结果（确保查询完全执行）
            cursor.fetchall()
            end_time = time.perf_counter()

            duration_ms = (end_time - start_time) * 1000
            return duration_ms

    except QueryCanceled:
        print(f"Query timeout (超过 {timeout_seconds} 秒)")
        return timeout_ms
    except psycopg2.Error as e:
        raise RuntimeError(f"数据库操作失败: {e}") from e
    finally:
        if conn:
            conn.close()

import re
def extract_execution_time_from_plan(plan: str) -> float:
    match = re.search(r'Actual Total Runtime:\s*([\d.]+)\s*ms', plan)
    if match:
        return float(match.group(1))
    else:
        print(f"Warning: Could not find execution time in plan")
        return None     
    return None

if __name__ == "__main__":
    demo_sql_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train-demo", "demo.sql")
    with open(demo_sql_path, "r", encoding="utf-8") as f:
        sql = f.read().strip()
    # hint = "/*+ Leading(mk k mc cn ci) HashJoin(mk k mc cn ci t n) */"
    # plan, raw_plan = get_sql_base_explain_plan(hint + " " + sql)
    # print(plan)
    plan, run_time, raw_plan = measure_sql_performance(sql)
    print(f"end2end: {run_time} ms")
    # print(plan)
    print(f"execution time from plan: {extract_execution_time_from_plan(plan)} ms")
    # hint1 = "/*+ */"
    # hint2 = "/*+ Leading(it1 it2 mi mi_idx) */"
    # print(hints_have_same_effect(sql, hint1, hint2))
