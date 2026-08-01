import os
import re
import asyncio
import time
import json
import argparse
import psycopg2
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import torch

from postgresql import get_sql_base_explain_plan, only_execute
from select_demonstration import choose_sql_by_filename, UtilityDemoSelector
from convert import json_to_pg_hint, pg_hint_to_json

# Assume llm_provider.py exists and provides an LLM generation function
from llm_provider import llm_provider


from config import Config
DB_CONFIG = {
    "dbname": Config.DB_DATABASE,
    "user": Config.DB_USER,
    "password": Config.DB_PASSWORD,
    "host": Config.DB_HOST,
    "port": Config.DB_PORT,
}

# --- Prompt Templates ---
# It's good practice to have these paths configurable or at the top
ROOT_DIR = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT_PATH = ROOT_DIR / "prompt" / "online-system.prompt"
USER_PROMPT_PATH = ROOT_DIR / "prompt" / "online-user.prompt"



from generate_pool import get_query_statistics


def _extract_template_num(name: str) -> str:
    stem = Path(name).stem
    m = re.match(r"(\d+)", stem)
    return m.group(1) if m else ""


def _read_text_file(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default


def _load_demo_payload_by_dir(demo_dir: Path) -> Tuple[str, str, str, str, str]:
    demo_sql = _read_text_file(demo_dir / "query.sql")
    demo_base_plan = _read_text_file(demo_dir / "original_execution_plan.txt")
    demo_recommended_hint = _read_text_file(demo_dir / "suggest_hint.txt")
    demo_hinted_plan = _read_text_file(demo_dir / "execution_plan.txt")
    infer_reason = _read_text_file(
        demo_dir / "infer_reason.txt",
        "Optimization based on execution plan analysis and performance patterns.",
    )
    return _adapt_demo_payload(
        (demo_sql, demo_base_plan, demo_recommended_hint, demo_hinted_plan, infer_reason)
    )


def _safe_hint_to_actions(hint_text: str) -> List[Dict[str, Any]]:
    text = (hint_text or "").strip()
    if not text or re.fullmatch(r"/\*\+\s*\*/", text, flags=re.DOTALL):
        return []
    try:
        return pg_hint_to_json(text)
    except Exception:
        return []


def _build_demo_recommended_output(recommended_hint: str, infer_reason: str) -> str:
    return json.dumps(
        {
            "reason": infer_reason,
            "actions": _safe_hint_to_actions(recommended_hint),
        },
        ensure_ascii=False,
        indent=2,
    )


def _adapt_demo_payload(payload: Tuple[str, str, str, str, str]) -> Tuple[str, str, str, str, str]:
    demo_sql, demo_base_plan, demo_recommended_hint, demo_hinted_plan, infer_reason = payload
    demo_recommended_output = _build_demo_recommended_output(demo_recommended_hint, infer_reason)
    return demo_sql, demo_base_plan, demo_recommended_output, demo_hinted_plan, infer_reason


def extract_json_object(llm_response: str) -> Dict[str, Any]:
    text = (llm_response or "").strip()
    if not text:
        return {"actions": [], "reason": ""}
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("LLM output must be a JSON object")
    actions = obj.get("actions", [])
    reason = obj.get("reason", "")
    if not isinstance(actions, list):
        raise ValueError("'actions' must be a list")
    if not isinstance(reason, str):
        raise ValueError("'reason' must be a string")
    return {"actions": actions, "reason": reason}


def extract_allowed_aliases(sql: str) -> List[str]:
    aliases: List[str] = []
    seen = set()

    def _add(alias: str) -> None:
        a = str(alias or "").strip()
        if not a or a in seen:
            return
        seen.add(a)
        aliases.append(a)

    sql_clean = " ".join(sql.split())
    from_match = re.search(
        r"\bFROM\b(.*?)(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)",
        sql_clean,
        flags=re.IGNORECASE,
    )
    if from_match:
        segment = from_match.group(1)
        for m in re.finditer(
            r"\b(?:FROM|JOIN)\s+([\w\.]+)(?:\s+(?:AS\s+)?(\w+))?",
            f"FROM {segment}",
            flags=re.IGNORECASE,
        ):
            table_name = (m.group(1) or "").split(".")[-1]
            alias = m.group(2) or table_name
            _add(alias)

    for m in re.finditer(r"\bJOIN\s+([\w\.]+)(?:\s+(?:AS\s+)?(\w+))?", sql_clean, flags=re.IGNORECASE):
        table_name = (m.group(1) or "").split(".")[-1]
        alias = m.group(2) or table_name
        _add(alias)

    return aliases


def generate_user_prompt(
    user_prompt_template: str,
    demo_payloads: List[Tuple[str, str, str, str, str]],
    target_sql: str,
    target_plan: str,
    target_stats: str,
    allowed_aliases: str,
) -> str:
    """Fills the user prompt template with all the required information."""
    prompt = user_prompt_template
    if demo_payloads:
        block_match = re.search(
            r"(=== Demonstration Example ===\s*)([\s\S]*?)(\s*=== Online Query ===)",
            prompt,
        )
        if block_match:
            block_template = block_match.group(2)
            rendered_blocks: List[str] = []
            for idx, (demo_sql, demo_base_plan, demo_recommended_output, demo_hinted_plan, _infer_reason) in enumerate(demo_payloads, start=1):
                block = block_template
                block = block.replace("{{demo_sql}}", demo_sql)
                block = block.replace("{{demo_base_plan}}", demo_base_plan)
                block = block.replace("{{demo_recommended_output}}", demo_recommended_output)
                block = block.replace("{{demo_hinted_plan}}", demo_hinted_plan)
                if len(demo_payloads) > 1:
                    block = block.replace("=== Demonstration Example ===", f"=== Demonstration Example {idx} ===")
                rendered_blocks.append(block.strip())
            prompt = (
                prompt[:block_match.start(2)]
                + "\n\n".join(rendered_blocks)
                + prompt[block_match.end(2):]
            )
    else:
        prompt = prompt.replace("{{demo_sql}}", "")
        prompt = prompt.replace("{{demo_base_plan}}", "")
        prompt = prompt.replace("{{demo_recommended_output}}", "")
        prompt = prompt.replace("{{demo_hinted_plan}}", "")

    prompt = prompt.replace("{{statistics}}", target_stats)
    prompt = prompt.replace("{{online_sql}}", target_sql)
    prompt = prompt.replace("{{online_base_plan}}", target_plan)
    prompt = prompt.replace("{{allowed_aliases}}", allowed_aliases)

    return prompt


@dataclass
class PreparedQuery:
    filename: str
    sql_query: str
    initial_plan: str
    input_plan: Dict[str, Any]
    original_time: float
    target_template: str
    demo_payloads: List[Tuple[str, str, str, str, str]] | None = None
    demo_sql: str = ""
    demo_base_plan: str = ""
    demo_recommended_output: str = ""
    demo_hinted_plan: str = ""
    infer_reason: str = ""
    selected_demo_id: str = ""
    selected_demo_template: str = ""
    selected_score: float | None = None
    copied_best_hint: str = ""
    should_call_llm: bool = False
    skip_optimization: bool = False
    no_demo_found: bool = False
    same_template_copy: bool = False
    response_hint: str = ""
    user_prompt: str = ""
    llm_response: str = ""
    token_usage: Dict[str, Any] | None = None
    reason_text: str = ""
    action_payload: Dict[str, Any] | None = None
    hint_time: float = 0.0
    current_plan: str = ""
    current_time: float = 0.0


def _save_query_outputs(
    output_path: Path,
    prepared: PreparedQuery,
    dump_debug_files: bool,
) -> None:
    output_dir = output_path / prepared.filename.replace(".sql", "")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "original_execution_plan.txt", "w", encoding="utf-8") as f:
        f.write(prepared.initial_plan)
    with open(output_dir / "suggested_hint.txt", "w", encoding="utf-8") as f:
        f.write(prepared.response_hint)
    with open(output_dir / "execution_plan.txt", "w", encoding="utf-8") as f:
        f.write(prepared.current_plan)
    with open(output_dir / "reason.txt", "w", encoding="utf-8") as f:
        f.write(prepared.reason_text)
    if prepared.token_usage is not None:
        with open(output_dir / "token_usage.json", "w", encoding="utf-8") as f:
            json.dump(prepared.token_usage, f, ensure_ascii=False, indent=2)
    if prepared.action_payload is not None:
        with open(output_dir / "action.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(prepared.action_payload, ensure_ascii=False) + "\n")
    with open(output_dir / "user_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prepared.user_prompt)
    if dump_debug_files:
        with open(output_dir / "llm_response.txt", "w", encoding="utf-8") as f:
            f.write(prepared.llm_response or "")
        selection_meta = {
            "target_file": prepared.filename,
            "target_template": prepared.target_template,
            "selected_demo_id": prepared.selected_demo_id,
            "selected_demo_template": prepared.selected_demo_template,
            "selected_score": prepared.selected_score,
            "same_template_copy": bool(prepared.same_template_copy),
            "skip_optimization": bool(prepared.skip_optimization),
            "used_demo_in_prompt": bool(
                bool(prepared.demo_payloads)
            ),
            "llm_called": bool(prepared.should_call_llm),
            "hint_generation_time_seconds": float(prepared.hint_time),
            "token_usage": prepared.token_usage or {},
        }
        with open(output_dir / "demo_selection.json", "w", encoding="utf-8") as f:
            json.dump(selection_meta, f, ensure_ascii=False, indent=2)


async def _run_llm_batch(
    prepared_queries: List[PreparedQuery],
    system_prompt: str,
    concurrency_limit: int,
) -> None:
    semaphore = asyncio.Semaphore(max(1, concurrency_limit))

    async def _one(prepared: PreparedQuery) -> None:
        async with semaphore:
            start = time.perf_counter()
            llm_result = await llm_provider.generate_with_metadata(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prepared.user_prompt},
                ]
            )
            prepared.hint_time = time.perf_counter() - start
            llm_response = str(llm_result.get("content", ""))
            prepared.llm_response = llm_response
            prepared.token_usage = dict(llm_result.get("usage", {}) or {})
            try:
                prepared.action_payload = extract_json_object(llm_response)
                prepared.reason_text = str(prepared.action_payload.get("reason", "") or "")
                prepared.response_hint = json_to_pg_hint(prepared.action_payload.get("actions", []))
            except Exception as e:
                prepared.reason_text = f"Invalid JSON LLM output: {e}"
                prepared.action_payload = {"actions": [], "reason": prepared.reason_text}
                prepared.response_hint = ""
                print(f"Failed to parse LLM JSON output for {prepared.filename}: {e}")

    await asyncio.gather(*[_one(p) for p in prepared_queries])


def _write_execution_csv(
    output_path: Path,
    file_time_dict: Dict[str, Tuple[float, float, float, int]],
    total_original_time: float,
    total_current_time: float,
    average_hint_time_sum: float,
    no_demo_found_count: int,
) -> None:
    record_count = len(file_time_dict)
    average_hint_time = (average_hint_time_sum / record_count) if record_count else 0.0
    average_original_time = (total_original_time / record_count) if record_count else 0.0
    average_current_time = (total_current_time / record_count) if record_count else 0.0
    no_demo_found_ratio = (no_demo_found_count / record_count) if record_count else 0.0
    with open(output_path / "execution_times.csv", "w", encoding="utf-8") as f:
        f.write("filename,original_execution_time_ms_not_measured,current_execution_time_ms,hint_generation_time_seconds,no_demo_found,no_demo_found_count,no_demo_found_ratio\n")
        for fname, (orig_time, curr_time, hint_time, no_demo_found) in file_time_dict.items():
            f.write(f"{fname},{orig_time:.2f},{curr_time:.2f},{hint_time:.2f},{no_demo_found},,\n")
        f.write(
            f"average,{average_original_time:.2f},{average_current_time:.2f},{average_hint_time:.2f},,"
            f"{no_demo_found_count},{no_demo_found_ratio:.6f}\n"
        )


def _get_online_explain_plan(sql: str) -> Tuple[str, Dict[str, Any]]:
    """Online planning must not execute the query; use plain EXPLAIN only."""
    plan_text, raw_plan = get_sql_base_explain_plan(sql, summarize=True)
    return str(plan_text), raw_plan


def _plan_and_execute_final_sql(sql: str) -> Tuple[str, float]:
    """Get the final plan with EXPLAIN, then execute the final SQL normally."""
    plan_text, _ = _get_online_explain_plan(sql)
    run_time_ms = only_execute(sql)
    return plan_text, run_time_ms


def _check_utility_model_artifacts_compat(
    ckpt_path: str,
    artifacts_dir: str,
) -> None:
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        ckpt_dim = int(state.get("config", {}).get("feature_dim", -1))
    except Exception as e:
        raise RuntimeError(f"failed to load utility checkpoint '{ckpt_path}': {e}") from e

    cfg_path = Path(artifacts_dir) / "config.json"
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        art_dim = int(cfg.get("feature_dim", -1))
    except Exception as e:
        raise RuntimeError(f"failed to load artifacts config '{cfg_path}': {e}") from e

    if ckpt_dim <= 0 or art_dim <= 0:
        raise RuntimeError(
            f"invalid feature dims: ckpt_dim={ckpt_dim}, artifacts_dim={art_dim}; "
            f"ckpt={ckpt_path}, artifacts={artifacts_dir}"
        )
    # New utility pipeline inserts predicate PCA(16) on top of base artifacts feature_dim.
    # So we accept either:
    # 1) direct match: ckpt_dim == art_dim
    # 2) pca-augmented match: ckpt_dim == art_dim + 16
    pca16_path = Path(artifacts_dir) / "predicate_pca16.npz"
    compat = (ckpt_dim == art_dim) or (pca16_path.exists() and ckpt_dim == art_dim + 16)
    if not compat:
        raise RuntimeError(
            f"utility feature_dim mismatch: ckpt={ckpt_dim}, artifacts={art_dim}. "
            f"Use matching pairs, e.g. train and run with the same --artifacts_dir."
        )


async def run_workload(
    query_dir: str,
    output_dir: str,
    demo_pool_dir: str,
    train_query_path: str,
    utility_ckpt_path: str,
    utility_artifacts_dir: str,
    dump_debug_files: bool = False,
    llm_concurrency: int = 1,
):
    """
    Processes each SQL file in query_dir to generate a pg_hint_plan hint.
    Online mode: run each query once first, then use that run's raw JSON plan
    as the input plan for demo selection.
    """
    query_path = Path(query_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    try:
        with open(SYSTEM_PROMPT_PATH, 'r') as f:
            system_prompt = f.read()
        with open(USER_PROMPT_PATH, 'r') as f:
            user_prompt_template = f.read()
    except FileNotFoundError as e:
        print(f"Error: Could not read prompt files. Make sure '{SYSTEM_PROMPT_PATH}' and '{USER_PROMPT_PATH}' exist. Details: {e}")
        return

    sql_files = sorted([f for f in os.listdir(query_path) if f.endswith(".sql")])
    print(f"Workload query_dir: {query_path}")
    print(f"Found {len(sql_files)} SQL files.")
    if not sql_files:
        print(
            "No SQL files found. Nothing to run. "
            "Please check query_dir and make sure it contains '*.sql' files."
        )
        return

    file_time_dict = {}# To store execution times for each file
    total_original_time = 0.0
    total_current_time = 0.0
    average_hint_time = 0.0
    no_demo_found_count = 0
    _check_utility_model_artifacts_compat(
        ckpt_path=utility_ckpt_path,
        artifacts_dir=utility_artifacts_dir,
    )
    utility_selector = UtilityDemoSelector(
        demo_pool_dir=demo_pool_dir,
        ckpt_path=utility_ckpt_path,
        artifacts_dir=utility_artifacts_dir,
        text_model_path=os.environ.get("DIAGHINT_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        device="cpu",
    )
    if llm_concurrency > 1:
        prepared_queries: List[PreparedQuery] = []
        llm_needed: List[PreparedQuery] = []
        for filename in sql_files:
            print(f"\n===== Processing SQL file: {filename} =====")
            filepath = query_path / filename
            with open(filepath, "r", encoding="utf-8") as f:
                sql_query = f.read()
            if not sql_query.strip():
                print(f"Warning: SQL file {filename} is empty. Skipping.")
                continue

            print("Running EXPLAIN only to get online raw JSON plan for demo selection...")
            warmup_plan_text, online_raw_plan = _get_online_explain_plan(sql_query)
            print("Online EXPLAIN finished; baseline query was not executed.")
            if not isinstance(online_raw_plan, dict):
                print(f"Skipping {filename} because online raw plan is invalid.")
                continue
            if "Error:" in warmup_plan_text:
                print(f"Skipping {filename} due to an error in plan generation.")
                continue

            prepared = PreparedQuery(
                filename=filename,
                sql_query=sql_query,
                initial_plan=warmup_plan_text,
                input_plan=online_raw_plan,
                original_time=0.0,
                target_template=_extract_template_num(filename),
            )

            same_template_payload = None
            try:
                same_template_payload = utility_selector.select_same_template_if_positive(
                    raw_plan=prepared.input_plan,
                    target_template=prepared.target_template,
                    target_name=filename.replace(".sql", ""),
                )
            except Exception as e:
                print(f"Same-template utility check failed for {filename}: {e}")

            if same_template_payload is not None:
                prepared.demo_payloads = [_adapt_demo_payload(same_template_payload)]
                (
                    prepared.demo_sql,
                    prepared.demo_base_plan,
                    prepared.demo_recommended_output,
                    prepared.demo_hinted_plan,
                    prepared.infer_reason,
                ) = prepared.demo_payloads[0]
            else:
                try:
                    top_items = utility_selector.select_top_in_context_nonempty(
                        raw_plan=prepared.input_plan,
                        target_template=prepared.target_template,
                        target_name=filename.replace(".sql", ""),
                        top_k=1,
                        allow_non_positive=False,
                    )
                    if top_items:
                        prepared.demo_payloads = [_adapt_demo_payload(top_items[0]["payload"])]
                        (
                            prepared.demo_sql,
                            prepared.demo_base_plan,
                            prepared.demo_recommended_output,
                            prepared.demo_hinted_plan,
                            prepared.infer_reason,
                        ) = prepared.demo_payloads[0]
                        prepared.should_call_llm = True
                    else:
                        prepared.skip_optimization = True
                        prepared.no_demo_found = True
                except Exception as e:
                    print(f"Utility selector failed, fallback to template pick for {filename}: {e}")
                    (
                        prepared.demo_sql,
                        prepared.demo_base_plan,
                        prepared.demo_recommended_output,
                        prepared.demo_hinted_plan,
                        prepared.infer_reason,
                    ) = _adapt_demo_payload(choose_sql_by_filename(filename))
                    prepared.demo_payloads = [(
                        prepared.demo_sql,
                        prepared.demo_base_plan,
                        prepared.demo_recommended_output,
                        prepared.demo_hinted_plan,
                        prepared.infer_reason,
                    )]
                    prepared.should_call_llm = True

            prepared.selected_demo_id = utility_selector.last_selected_demo_id or ""
            prepared.selected_demo_template = _extract_template_num(prepared.selected_demo_id)
            prepared.selected_score = utility_selector.last_selected_score
            prepared.copied_best_hint = (utility_selector.last_selected_best_hint or "").strip()
            prepared.same_template_copy = (
                bool(prepared.target_template)
                and bool(prepared.selected_demo_template)
                and prepared.target_template == prepared.selected_demo_template
            )

            if prepared.selected_demo_id and not prepared.copied_best_hint:
                prepared.response_hint = ""
                prepared.reason_text = f"Selected demo {prepared.selected_demo_id} has empty hint. Run PostgreSQL baseline without hint."
                prepared.action_payload = {"actions": [], "reason": prepared.reason_text}
                prepared.user_prompt = (
                    f"[SKIPPED_LLM] selected demo has empty hint. "
                    f"demo={prepared.selected_demo_id}, run_without_hint=true"
                )
            elif prepared.same_template_copy and prepared.copied_best_hint:
                prepared.response_hint = prepared.copied_best_hint
                prepared.reason_text = f"Same template detected. Reused best hint from demo {prepared.selected_demo_id}."
                prepared.action_payload = {
                    "actions": _safe_hint_to_actions(prepared.copied_best_hint),
                    "reason": prepared.reason_text,
                }
                prepared.user_prompt = (
                    f"[SKIPPED_LLM] same template detected. "
                    f"demo={prepared.selected_demo_id}, hint={prepared.response_hint}"
                )
            elif prepared.skip_optimization:
                prepared.response_hint = ""
                prepared.reason_text = "No positive non-empty in-context demo found. Run PostgreSQL baseline without hint."
                prepared.action_payload = {"actions": [], "reason": prepared.reason_text}
                prepared.user_prompt = (
                    "[SKIPPED_LLM] no positive in-context demo with non-empty hint; "
                    "run_without_hint=true"
                )
            elif prepared.should_call_llm:
                prepared.should_call_llm = True
                print("Fetching database statistics...")
                statistics = get_query_statistics(sql_query)
                allowed_aliases = json.dumps(extract_allowed_aliases(sql_query), ensure_ascii=False)
                prepared.user_prompt = generate_user_prompt(
                    user_prompt_template,
                    prepared.demo_payloads or [],
                    sql_query,
                    prepared.initial_plan,
                    statistics,
                    allowed_aliases,
                )
                llm_needed.append(prepared)
            else:
                prepared.response_hint = ""
                prepared.reason_text = "No online optimization decision path was taken."
                prepared.action_payload = {"actions": [], "reason": prepared.reason_text}
                prepared.user_prompt = "[SKIPPED_LLM] no decision path."

            prepared_queries.append(prepared)

        if llm_needed:
            print(f"\nSending {len(llm_needed)} LLM requests concurrently (limit={llm_concurrency})...")
            await _run_llm_batch(llm_needed, system_prompt=system_prompt, concurrency_limit=llm_concurrency)
            for prepared in llm_needed:
                print(f"LLM finished for {prepared.filename}: {prepared.hint_time:.2f} seconds")

        for prepared in prepared_queries:
            current_plan, current_time = _plan_and_execute_final_sql(f"{prepared.response_hint} {prepared.sql_query}".strip())
            prepared.current_plan = current_plan
            prepared.current_time = current_time
            print("Execution Time Original (No Hint): not measured online (EXPLAIN only)")
            print(f"Execution Time with Hint: {prepared.current_time:.2f} ms")
            _save_query_outputs(output_path, prepared, dump_debug_files=dump_debug_files)
            file_time_dict[prepared.filename] = (
                prepared.original_time,
                prepared.current_time,
                prepared.hint_time,
                int(prepared.no_demo_found),
            )
            total_original_time += prepared.original_time
            total_current_time += prepared.current_time
            average_hint_time += prepared.hint_time
            if prepared.no_demo_found:
                no_demo_found_count += 1

        _write_execution_csv(
            output_path=output_path,
            file_time_dict=file_time_dict,
            total_original_time=total_original_time,
            total_current_time=total_current_time,
            average_hint_time_sum=average_hint_time,
            no_demo_found_count=no_demo_found_count,
        )
        return

    for filename in sql_files:
        print(f"\n===== Processing SQL file: {filename} =====")
        filepath = query_path / filename

        with open(filepath, 'r') as f:
            sql_query = f.read()

        if not sql_query.strip():
            print(f"Warning: SQL file {filename} is empty. Skipping.")
            continue

        # 1) Online requirement: use EXPLAIN only, and use this raw JSON as input_plan.
        print("Running EXPLAIN only to get online raw JSON plan for demo selection...")
        warmup_plan_text, online_raw_plan = _get_online_explain_plan(sql_query)
        original_time = 0.0
        print("Online EXPLAIN finished; baseline query was not executed.")

        start_time = time.perf_counter()

        # 2) Use online run outputs directly (no extra explain/predict stage).
        # initial_plan = json.dumps(online_raw_plan)
        initial_plan = warmup_plan_text
        input_plan = online_raw_plan
        target_template = _extract_template_num(filename)
        demo_payloads: List[Tuple[str, str, str, str, str]] = []
        demo_sql = demo_base_plan = demo_recommended_output = demo_hinted_plan = infer_reason = ""
        llm_response = ""
        token_usage: Dict[str, Any] | None = None
        action_payload: Dict[str, Any] | None = None
        reason_text = ""
        same_template_payload = None
        per_query_user_prompt_template = user_prompt_template
        should_call_llm = False
        skip_optimization = False
        no_demo_found = False
        demo_payloads = []
        try:
            same_template_payload = utility_selector.select_same_template_if_positive(
                raw_plan=input_plan,
                target_template=target_template,
                target_name=filename.replace(".sql", ""),
            )
        except Exception as e:
            print(f"Same-template utility check failed for {filename}: {e}")

        if same_template_payload is not None:
            demo_payloads = [_adapt_demo_payload(same_template_payload)]
            demo_sql, demo_base_plan, demo_recommended_output, demo_hinted_plan, infer_reason = demo_payloads[0]
        else:
            try:
                top_items = utility_selector.select_top_in_context_nonempty(
                    raw_plan=input_plan,
                    target_template=target_template,
                    target_name=filename.replace(".sql", ""),
                    top_k=1,
                    allow_non_positive=False,
                )
                if top_items:
                    demo_payloads = [_adapt_demo_payload(top_items[0]["payload"])]
                    demo_sql, demo_base_plan, demo_recommended_output, demo_hinted_plan, infer_reason = demo_payloads[0]
                    should_call_llm = True
                else:
                    skip_optimization = True
                    no_demo_found = True
            except Exception as e:
                print(f"Utility selector failed, fallback to template pick for {filename}: {e}")
                demo_payloads = [_adapt_demo_payload(choose_sql_by_filename(filename))]
                demo_sql, demo_base_plan, demo_recommended_output, demo_hinted_plan, infer_reason = demo_payloads[0]
                should_call_llm = True
        if not isinstance(input_plan, dict):
            print(f"Skipping {filename} because online raw plan is invalid.")
            continue
        if "Error:" in initial_plan:
            print(f"Skipping {filename} due to an error in plan generation.")
            continue

        selected_demo_id = utility_selector.last_selected_demo_id or ""
        selected_demo_template = _extract_template_num(selected_demo_id)

        copied_best_hint = (utility_selector.last_selected_best_hint or "").strip()
        same_template = (
            bool(target_template)
            and bool(selected_demo_template)
            and target_template == selected_demo_template
        )

        if selected_demo_id and not copied_best_hint:
            print(
                f"Selected demo has empty hint: target={filename}, demo={selected_demo_id}. "
                "Skip LLM and run without hint."
            )
            response_hint = ""
            reason_text = f"Selected demo {selected_demo_id} has empty hint. Run PostgreSQL baseline without hint."
            action_payload = {"actions": [], "reason": reason_text}
            user_prompt = (
                f"[SKIPPED_LLM] selected demo has empty hint. "
                f"demo={selected_demo_id}, run_without_hint=true"
            )
            end_time = time.perf_counter()
            hint_time = end_time - start_time
            average_hint_time += hint_time
            print("Copied Hint: <empty>")
        elif same_template and copied_best_hint:
            print(
                f"Same-template hit: target={filename}, demo={selected_demo_id}. "
                "Skip LLM and copy demo best hint."
            )
            response_hint = copied_best_hint
            reason_text = f"Same template detected. Reused best hint from demo {selected_demo_id}."
            action_payload = {
                "actions": _safe_hint_to_actions(copied_best_hint),
                "reason": reason_text,
            }
            user_prompt = (
                f"[SKIPPED_LLM] same template detected. "
                f"demo={selected_demo_id}, hint={response_hint}"
            )
            end_time = time.perf_counter()
            hint_time = end_time - start_time
            average_hint_time += hint_time
            print(f"Copied Hint: {response_hint}")
        elif skip_optimization:
            print(
                f"No positive non-empty in-context demo for {filename}. "
                "Skip optimization and run PostgreSQL baseline (no hint)."
            )
            response_hint = ""
            reason_text = "No positive non-empty in-context demo found. Run PostgreSQL baseline without hint."
            action_payload = {"actions": [], "reason": reason_text}
            user_prompt = (
                "[SKIPPED_LLM] no positive in-context demo with non-empty hint; "
                "run_without_hint=true"
            )
            end_time = time.perf_counter()
            hint_time = end_time - start_time
            average_hint_time += hint_time
            no_demo_found_count += 1
            print("Response Hint: <empty>")
        elif should_call_llm:
            should_call_llm = True
            # 3. Get database statistics for the target query
            print("Fetching database statistics...")
            statistics = get_query_statistics(sql_query)
            allowed_aliases = json.dumps(extract_allowed_aliases(sql_query), ensure_ascii=False)

            # 4. Construct the full user prompt
            user_prompt = generate_user_prompt(
                per_query_user_prompt_template,
                demo_payloads,
                sql_query,
                initial_plan,
                statistics,
                allowed_aliases,
            )

            # 5. Call the LLM to get the hint
            print("Sending request to LLM to get hint...")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            with open("1.log", "w", encoding="utf-8") as log_file:
                log_file.write(f"User Prompt: {user_prompt}\n")
            llm_result = await llm_provider.generate_with_metadata(messages)
            llm_response = str(llm_result.get("content", ""))
            token_usage = dict(llm_result.get("usage", {}) or {})

            end_time = time.perf_counter()
            hint_time = end_time - start_time
            print(f"Time taken to generate hint: {hint_time:.2f} seconds")
            average_hint_time += hint_time

            print(f"LLM Suggested Hint: {llm_response}")
            try:
                action_payload = extract_json_object(llm_response)
                reason_text = str(action_payload.get("reason", "") or "")
                response_hint = json_to_pg_hint(action_payload.get("actions", []))
            except Exception as e:
                reason_text = f"Invalid JSON LLM output: {e}"
                action_payload = {"actions": [], "reason": reason_text}
                response_hint = ""
                print(f"Failed to parse LLM JSON output for {filename}: {e}")
            print(f"Response Hint: {response_hint}")
        else:
            # Defensive fallback: skip optimization.
            end_time = time.perf_counter()
            hint_time = end_time - start_time
            average_hint_time += hint_time
            response_hint = ""
            reason_text = "No online optimization decision path was taken."
            action_payload = {"actions": [], "reason": reason_text}
            user_prompt = "[SKIPPED_LLM] no decision path."

        current_plan, current_time = _plan_and_execute_final_sql(f"{response_hint} {sql_query}".strip())
        print("Execution Time Original (No Hint): not measured online (EXPLAIN only)")
        print(f"Execution Time with Hint: {current_time:.2f} ms")
        # 6. Save the results(base plan, hint, plan with hint) to output directory
        output_dir = output_path / filename.replace(".sql", "")
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "original_execution_plan.txt", 'w') as f:
            f.write(initial_plan)
        with open(output_dir / "suggested_hint.txt", 'w') as f:
            f.write(response_hint)
        with open(output_dir / "execution_plan.txt", 'w') as f:
            f.write(current_plan)
        with open(output_dir / "reason.txt", "w", encoding="utf-8") as f:
            f.write(reason_text)
        if token_usage is not None:
            with open(output_dir / "token_usage.json", "w", encoding="utf-8") as f:
                json.dump(token_usage, f, ensure_ascii=False, indent=2)
        if action_payload is not None:
            with open(output_dir / "action.jsonl", "w", encoding="utf-8") as f:
                f.write(json.dumps(action_payload, ensure_ascii=False) + "\n")
        # Save system prompt and user prompt for reference
        # with open(output_dir / "system_prompt.txt", 'w') as f:
        #     f.write(system_prompt)
        with open(output_dir / "user_prompt.txt", 'w') as f:
            f.write(user_prompt)
        if dump_debug_files:
            with open(output_dir / "llm_response.txt", "w", encoding="utf-8") as f:
                f.write(llm_response or "")
            selection_meta = {
                "target_file": filename,
                "target_template": target_template,
                "selected_demo_id": selected_demo_id,
                "selected_demo_template": selected_demo_template,
                "selected_score": (utility_selector.last_selected_score if utility_selector else None),
                "same_template_copy": bool(same_template and copied_best_hint),
                "skip_optimization": bool(skip_optimization),
                "used_demo_in_prompt": bool(
                    should_call_llm and
                    bool(demo_sql.strip() or demo_base_plan.strip() or demo_recommended_output.strip() or demo_hinted_plan.strip())
                ),
                "llm_called": bool(should_call_llm and not (same_template and copied_best_hint) and not skip_optimization),
                "hint_generation_time_seconds": float(hint_time),
                "token_usage": token_usage or {},
            }
            with open(output_dir / "demo_selection.json", "w", encoding="utf-8") as f:
                json.dump(selection_meta, f, ensure_ascii=False, indent=2)
        file_time_dict[filename] = original_time, current_time, hint_time, int(no_demo_found)
        total_original_time += original_time
        total_current_time += current_time

    _write_execution_csv(
        output_path=output_path,
        file_time_dict=file_time_dict,
        total_original_time=total_original_time,
        total_current_time=total_current_time,
        average_hint_time_sum=average_hint_time,
        no_demo_found_count=no_demo_found_count,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run online workload with utility demo selector.")

    parser.add_argument(
        "--query_dir",
        type=str,
        default=str(ROOT_DIR / "data" / "test-query"),
        help="Directory containing workload SQL files (*.sql).",
    )

    parser.add_argument(
        "--demo_pool_dir",
        type=str,
        default=str(ROOT_DIR / "outputs" / "demo_pool"),
        help="Demonstration pool directory.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT_DIR / "outputs" / "online_run"),
        help="Output directory for workload results.",
    )

    parser.add_argument(
        "--train_query_path",
        type=str,
        default=str(ROOT_DIR / "data" / "train-query"),
        help="Train query directory path (kept for compatibility).",
    )
    # default = True
    parser.add_argument(
        "--utility_ckpt_path",
        type=str,
        default=str(ROOT_DIR / "models" / "utility" / "best.pt"),
        help="Path to utility model checkpoint used by select_demonstration.",
    )
    parser.add_argument(
        "--utility_artifacts_dir",
        type=str,
        default=str(ROOT_DIR / "models" / "cardinality_bias"),
        help="Path to utility artifacts directory.",
    )
    parser.add_argument("--dump_debug_files", action="store_true", default=False)
    parser.add_argument("--llm_concurrency", type=int, default=1, help="Concurrent LLM requests limit. DB execution remains sequential.")
    args = parser.parse_args()

    asyncio.run(
        run_workload(
            query_dir=args.query_dir,
            output_dir=args.output_dir,
            demo_pool_dir=args.demo_pool_dir,
            train_query_path=args.train_query_path,
            utility_ckpt_path=args.utility_ckpt_path,
            utility_artifacts_dir=args.utility_artifacts_dir,
            dump_debug_files=args.dump_debug_files,
            llm_concurrency=args.llm_concurrency,
        )
    )
