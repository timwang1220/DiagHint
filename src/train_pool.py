#!/usr/bin/env python3
"""Build training pool with reusable same-template rounds + random-context rounds."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HINT_RE = re.compile(r"(/\*\+[\s\S]*?\*/)")
SQL_FILE_RE = re.compile(r"^(\d+)([a-zA-Z])\.sql$")
EMPTY_HINT_RE = re.compile(r"^/\*\+\s*\*/$", flags=re.DOTALL)
ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate training pool and utility trials from query folder.")
    parser.add_argument(
        "--train_query_dir",
        type=str,
        default=str(ROOT_DIR / "data" / "train-query"),
    )
    parser.add_argument(
        "--target_pool",
        type=str,
        default=str(ROOT_DIR / "outputs" / "demo_pool"),
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=str(ROOT_DIR / "prompt" / "online-system.prompt"),
    )
    parser.add_argument(
        "--user_prompt",
        type=str,
        default=str(ROOT_DIR / "prompt" / "online-user.prompt"),
    )
    parser.add_argument("--random_rounds", type=int, default=6)
    parser.add_argument("--max_total_rounds", type=int, default=8)
    parser.add_argument("--a_max_iterations", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_files", type=int, default=0, help="0 means all files.")
    parser.add_argument(
        "--resume_missing",
        action="store_true",
        help=(
            "Only process missing queries in target_pool and merge existing utility_trials.jsonl "
            "with newly generated trials."
        ),
    )
    return parser.parse_args()


def read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def is_effectively_empty_hint(hint: str) -> bool:
    """Treat /*+ */ (allowing whitespace/newlines) as empty hint."""
    text = (hint or "").strip()
    if not text:
        return True
    return bool(EMPTY_HINT_RE.fullmatch(text))


def resolve_demo_best_hint(demo_dir: Path) -> str:
    """Best-hint priority for demo: final_combined > suggest > empty."""
    final_hint = read_text(demo_dir / "final_combined_hint.txt", "").strip()
    if final_hint:
        return final_hint
    suggest_hint = read_text(demo_dir / "suggest_hint.txt", "/*+ */").strip()
    return suggest_hint or "/*+ */"


def extract_hint(llm_response: str) -> str:
    matches = HINT_RE.findall(llm_response or "")
    if not matches:
        return "/*+ */"
    return matches[-1].strip()


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


def parse_sql_filename(filename: str) -> Tuple[int, str]:
    m = SQL_FILE_RE.match(filename)
    if not m:
        raise ValueError(f"Invalid SQL filename: {filename}")
    return int(m.group(1)), m.group(2).lower()


def stable_query_seed(base_seed: int, query_id: str) -> int:
    digest = hashlib.md5(query_id.encode("utf-8")).hexdigest()
    salt = int(digest[:8], 16)
    return (base_seed ^ salt) & 0x7FFFFFFF


def sort_sql_files(files: List[str]) -> List[str]:
    def key_fn(name: str) -> Tuple[int, str]:
        t, s = parse_sql_filename(name)
        return t, s

    return sorted(files, key=key_fn)


def load_demo_record(pool_dir: Path, demo_id: str) -> Dict[str, str]:
    from convert import pg_hint_to_json  # pylint: disable=import-outside-toplevel

    demo_dir = pool_dir / demo_id
    recommended_hint = resolve_demo_best_hint(demo_dir)
    infer_reason = read_text(
        demo_dir / "infer_reason.txt",
        "Optimization based on execution plan analysis and performance patterns.",
    )
    try:
        recommended_output = json.dumps(
            {
                "reason": infer_reason,
                "actions": pg_hint_to_json(recommended_hint),
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception:
        recommended_output = json.dumps(
            {
                "reason": infer_reason,
                "actions": [],
            },
            ensure_ascii=False,
            indent=2,
        )
    return {
        "demo_id": demo_id,
        "sql": read_text(demo_dir / "query.sql"),
        "base_plan": read_text(demo_dir / "original_execution_plan.txt"),
        "recommended_hint": recommended_hint,
        "recommended_output": recommended_output,
        "hinted_plan": read_text(demo_dir / "execution_plan.txt"),
        "infer_reason": infer_reason,
        "source_plan_json": str((demo_dir / "0" / "plan.json").resolve()),
    }


def build_user_prompt(
    user_prompt_template: str,
    demo_sql: str,
    demo_base_plan: str,
    demo_recommended_output: str,
    demo_hinted_plan: str,
    target_sql: str,
    target_plan: str,
    target_stats: str,
    allowed_aliases: str,
) -> str:
    prompt = user_prompt_template
    prompt = prompt.replace("{{demo_sql}}", demo_sql)
    prompt = prompt.replace("{{demo_base_plan}}", demo_base_plan)
    prompt = prompt.replace("{{demo_recommended_output}}", demo_recommended_output)
    prompt = prompt.replace("{{demo_hinted_plan}}", demo_hinted_plan)
    prompt = prompt.replace("{{statistics}}", target_stats)
    prompt = prompt.replace("{{online_sql}}", target_sql)
    prompt = prompt.replace("{{online_base_plan}}", target_plan)
    prompt = prompt.replace("{{allowed_aliases}}", allowed_aliases)
    return prompt


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

    from_match = re.search(r"\bFROM\b(.*?)(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)", sql_clean, flags=re.IGNORECASE)
    if from_match:
        segment = from_match.group(1)
        for m in re.finditer(r"\b(?:FROM|JOIN)\s+([\w\.]+)(?:\s+(?:AS\s+)?(\w+))?", f"FROM {segment}", flags=re.IGNORECASE):
            table_name = (m.group(1) or "").split(".")[-1]
            alias = m.group(2) or table_name
            _add(alias)

    for m in re.finditer(r"\bJOIN\s+([\w\.]+)(?:\s+(?:AS\s+)?(\w+))?", sql_clean, flags=re.IGNORECASE):
        table_name = (m.group(1) or "").split(".")[-1]
        alias = m.group(2) or table_name
        _add(alias)

    return aliases


def save_round(
    query_out_dir: Path,
    round_idx: int,
    hint: str,
    raw_plan: Dict[str, Any],
    plan_summary: str,
    prompt_text: str,
    reason_text: str = "",
    action_payload: Optional[Dict[str, Any]] = None,
    token_usage: Optional[Dict[str, Any]] = None,
) -> Path:
    round_dir = query_out_dir / str(round_idx)
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "hint.txt").write_text(hint, encoding="utf-8")
    (round_dir / "plan-summary.txt").write_text(plan_summary, encoding="utf-8")
    (round_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    (round_dir / "reason.txt").write_text(reason_text, encoding="utf-8")
    if token_usage is not None:
        (round_dir / "token_usage.json").write_text(
            json.dumps(token_usage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if action_payload is not None:
        with open(round_dir / "action.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(action_payload, ensure_ascii=False) + "\n")
    with open(round_dir / "plan.json", "w", encoding="utf-8") as f:
        json.dump(raw_plan, f, ensure_ascii=False, indent=2)
    return round_dir / "plan.json"


def list_available_a_templates(source_pool: Path) -> List[int]:
    nums: List[int] = []
    for p in source_pool.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^(\d+)a$", p.name)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def choose_random_demo_ids(
    source_pool: Path,
    target_template_num: int,
    k: int,
    seed: int,
) -> List[str]:
    candidates: List[int] = []
    for n in list_available_a_templates(source_pool):
        if n == target_template_num:
            continue
        demo_dir = source_pool / f"{n}a"
        best_hint = resolve_demo_best_hint(demo_dir)
        if is_effectively_empty_hint(best_hint):
            continue
        candidates.append(n)
    if len(candidates) < k:
        raise ValueError(
            f"Not enough non-empty-hint demo templates to sample {k}, only {len(candidates)} available."
        )
    rnd = random.Random(seed)
    picked = rnd.sample(candidates, k)
    return [f"{n}a" for n in picked]


def query_in_pool(pool_dir: Path, query_id: str) -> bool:
    qdir = pool_dir / query_id
    return qdir.exists() and (qdir / "suggest_hint.txt").exists()


def query_completed(pool_dir: Path, query_id: str) -> bool:
    qdir = pool_dir / query_id
    required = [
        qdir / "query.sql",
        qdir / "original_execution_plan.txt",
        qdir / "suggest_hint.txt",
        qdir / "execution_plan.txt",
        qdir / "0" / "plan.json",
    ]
    return all(p.exists() for p in required)


def load_existing_utility_trials(target_pool: Path) -> List[Dict[str, Any]]:
    jsonl_path = target_pool / "utility_trials.jsonl"
    if not jsonl_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def resolve_reuse_sources(
    template_num: int,
    suffix: str,
    target_pool: Path,
) -> List[Dict[str, str]]:
    """Reuse all existing same-template variants from 'a' to suffix-1.

    Priority: current target_pool only (generated results in this pipeline).
    """
    if suffix <= "a":
        return []

    # Keep at most one same-template reuse source.
    # Prefer the smallest previous suffix (a, then b, ...), and require non-empty hint.
    for code in range(ord("a"), ord(suffix)):
        letter = chr(code)
        qid = f"{template_num}{letter}"
        if not query_in_pool(target_pool, qid):
            continue

        hint_text = resolve_demo_best_hint(target_pool / qid).strip()
        if is_effectively_empty_hint(hint_text):
            continue

        return [
            {
                "phase": f"reuse_{letter}",
                "source_id": qid,
                "source_pool_type": "target_pool",
                "query_id": qid,
            }
        ]
    return []


def resolve_empty_reuse_source(
    template_num: int,
    suffix: str,
    target_pool: Path,
) -> Optional[str]:
    """Find earliest previous same-template query whose best hint is empty."""
    if suffix <= "a":
        return None
    for code in range(ord("a"), ord(suffix)):
        letter = chr(code)
        qid = f"{template_num}{letter}"
        if not query_in_pool(target_pool, qid):
            continue
        hint_text = resolve_demo_best_hint(target_pool / qid).strip()
        if is_effectively_empty_hint(hint_text):
            return qid
    return None


def median(values: List[float]) -> float:
    if not values:
        return 1.0
    xs = sorted(values)
    n = len(xs)
    m = n // 2
    if n % 2 == 1:
        return xs[m]
    return 0.5 * (xs[m - 1] + xs[m])


def get_query_statistics_safe(sql: str) -> str:
    """Lazy-load get_query_statistics to avoid import-time dependency failures."""
    try:
        from generate_pool import get_query_statistics  # pylint: disable=import-outside-toplevel
        return get_query_statistics(sql)
    except Exception as exc:  # pragma: no cover - defensive fallback
        return f"Error fetching statistics: {exc}"


async def run_a_query_with_generate_pool(
    sql_file: Path,
    args: argparse.Namespace,
    target_pool: Path,
) -> Dict[str, float]:
    """Run generate_pool flow for *a.sql and write into target_pool."""
    from generate_pool import get_best_pg_hint_plan  # pylint: disable=import-outside-toplevel

    target_id = sql_file.stem
    out_dir = target_pool / target_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sql_text = sql_file.read_text(encoding="utf-8")

    print(f"\n===== Processing {sql_file.name} with generate_pool =====")
    baseline_plan, suggest_hint, execution_plan, best_time, baseline_time, infer_reason, regression = await get_best_pg_hint_plan(
        sql_text,
        max_iterations=args.a_max_iterations,
        output_path=out_dir,
    )

    (out_dir / "suggest_hint.txt").write_text(suggest_hint, encoding="utf-8")
    (out_dir / "execution_plan.txt").write_text(execution_plan, encoding="utf-8")
    (out_dir / "original_execution_plan.txt").write_text(baseline_plan, encoding="utf-8")
    (out_dir / "query.sql").write_text(sql_text, encoding="utf-8")
    (out_dir / "infer_reason.txt").write_text(infer_reason, encoding="utf-8")
    (out_dir / "regression_analysis.json").write_text(
        json.dumps(regression, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"best_time": best_time, "baseline_time": baseline_time}


async def run_single_query(
    sql_file: Path,
    args: argparse.Namespace,
    system_prompt: str,
    user_prompt_template: str,
    source_pool: Path,
    target_pool: Path,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    from convert import json_to_pg_hint, pg_hint_to_json  # pylint: disable=import-outside-toplevel
    from postgresql import measure_sql_performance  # pylint: disable=import-outside-toplevel
    from llm_provider import llm_provider  # pylint: disable=import-outside-toplevel

    filename = sql_file.name
    template_num, suffix = parse_sql_filename(filename)
    target_id = sql_file.stem
    query_out_dir = target_pool / target_id
    query_out_dir.mkdir(parents=True, exist_ok=True)

    sql_query = sql_file.read_text(encoding="utf-8")
    if not sql_query.strip():
        raise RuntimeError(f"Empty SQL file: {sql_file}")

    print(f"\n===== Processing {filename} =====")
    baseline_plan, baseline_time, baseline_raw = measure_sql_performance(sql_query, num_runs=args.num_runs)
    print(f"Baseline: {baseline_time:.2f} ms")

    (query_out_dir / "query.sql").write_text(sql_query, encoding="utf-8")
    (query_out_dir / "original_execution_plan.txt").write_text(baseline_plan, encoding="utf-8")
    (query_out_dir / "0").mkdir(parents=True, exist_ok=True)
    with open(query_out_dir / "0" / "plan.json", "w", encoding="utf-8") as f:
        json.dump(baseline_raw, f, ensure_ascii=False, indent=2)
    (query_out_dir / "0" / "plan-summary.txt").write_text(baseline_plan, encoding="utf-8")

    statistics = get_query_statistics_safe(sql_query)
    allowed_aliases = json.dumps(extract_allowed_aliases(sql_query), ensure_ascii=False)
    regression_analysis: List[Dict[str, Any]] = []
    utility_trials: List[Dict[str, Any]] = []

    best_time = baseline_time
    best_hint = "/*+ */"
    best_plan = baseline_plan
    best_round = 0
    best_reason = "Baseline is best."
    successful_rounds = 0
    round_failure_messages: List[str] = []

    round_specs: List[Dict[str, Any]] = []
    reuse_sources = resolve_reuse_sources(template_num, suffix, target_pool=target_pool)
    empty_reuse_source_id: Optional[str] = None
    if not reuse_sources:
        empty_reuse_source_id = resolve_empty_reuse_source(template_num, suffix, target_pool=target_pool)
    for i, src in enumerate(reuse_sources, start=1):
        round_specs.append(
            {
                "round_idx": i,
                "phase": src["phase"],
                "source_id": src["source_id"],
                "use_llm": False,
                "source_pool_type": src["source_pool_type"],
            }
        )

    random_budget = max(0, int(args.max_total_rounds) - len(reuse_sources))
    random_k = min(int(args.random_rounds), random_budget)
    random_seed = stable_query_seed(args.seed, target_id)
    random_demo_ids = choose_random_demo_ids(source_pool, template_num, random_k, random_seed)
    random_start = len(reuse_sources) + 1
    for i, demo_id in enumerate(random_demo_ids):
        round_specs.append(
            {
                "round_idx": random_start + i,
                "phase": "random_context",
                "source_id": demo_id,
                "demo_id": demo_id,
                "use_llm": True,
            }
        )

    print(
        f"Plan rounds for {target_id}: reuse={len(reuse_sources)}, "
        f"random={random_k}, total={len(round_specs)}"
    )
    if empty_reuse_source_id is not None:
        print(
            f"Empty-hint same-template source detected for {target_id}: {empty_reuse_source_id}. "
            "Will add a synthetic reuse trial with utility=0."
        )

    for spec in round_specs:
        round_idx = spec["round_idx"]
        phase = spec["phase"]
        source_id = spec["source_id"]
        use_llm = spec["use_llm"]

        if phase.startswith("reuse_"):
            source_pool_type = spec.get("source_pool_type", "target_pool")
            base_pool = target_pool if source_pool_type == "target_pool" else source_pool
            source_suggest = base_pool / source_id / "suggest_hint.txt"
            if not source_suggest.exists():
                msg = f"missing_dependency: {source_pool_type}/{source_id}/suggest_hint.txt not found, skip {phase}"
                print(f"[WARN] {target_id} round {round_idx}: {msg}")
                regression_analysis.append(
                    {
                        "round_idx": round_idx,
                        "phase": phase,
                        "source_id": source_id,
                        "skipped": True,
                        "missing_dependency": True,
                        "message": msg,
                    }
                )
                continue
            applied_hint = read_text(source_suggest, "/*+ */").strip() or "/*+ */"
            prompt_text = (
                f"[{phase.upper()}] target={target_id}, source={source_id}, "
                f"pool={source_pool_type}, hint={applied_hint}"
            )
            infer_reason = f"Reuse hint from {source_pool_type}/{source_id}."
            action_payload = {
                "actions": pg_hint_to_json(applied_hint) if not is_effectively_empty_hint(applied_hint) else [],
                "reason": infer_reason,
            }
            token_usage = None
            demo_template_id = str(template_num)
            source_plan_json = str((base_pool / source_id / "0" / "plan.json").resolve())
        else:
            demo = load_demo_record(source_pool, spec["demo_id"])
            prompt_text = build_user_prompt(
                user_prompt_template=user_prompt_template,
                demo_sql=demo["sql"],
                demo_base_plan=demo["base_plan"],
                demo_recommended_output=demo["recommended_output"],
                demo_hinted_plan=demo["hinted_plan"],
                target_sql=sql_query,
                target_plan=baseline_plan,
                target_stats=statistics,
                allowed_aliases=allowed_aliases,
            )
            llm_start = time.perf_counter()
            llm_result = await llm_provider.generate_with_metadata(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ]
            )
            llm_response = str(llm_result.get("content", ""))
            token_usage = dict(llm_result.get("usage", {}) or {})
            llm_sec = time.perf_counter() - llm_start
            try:
                action_payload = extract_json_object(llm_response)
            except Exception as exc:
                round_failure_reason = f"invalid_json_output round={round_idx} source={spec['demo_id']}: {exc}"
                round_failure_messages.append(round_failure_reason)
                regression_analysis.append(
                    {
                        "round_idx": round_idx,
                        "phase": phase,
                        "source_id": source_id,
                        "llm_response": llm_response,
                        "failed": True,
                        "message": round_failure_reason,
                    }
                )
                print(f"[WARN] {target_id}: {round_failure_reason}")
                continue
            applied_hint = json_to_pg_hint(action_payload.get("actions", [])) if action_payload.get("actions") else "/*+ */"
            infer_reason = action_payload.get("reason", "")
            demo_template_id = str(parse_sql_filename(f"{spec['demo_id']}.sql")[0])
            source_plan_json = demo["source_plan_json"]
        if not use_llm:
            llm_sec = 0.0
            token_usage = None

        hinted_sql = f"{applied_hint} {sql_query}".strip()
        plan_text, used_time, raw_plan = measure_sql_performance(hinted_sql, num_runs=args.num_runs)
        trial_plan_json_path = save_round(
            query_out_dir=query_out_dir,
            round_idx=round_idx,
            hint=applied_hint,
            raw_plan=raw_plan,
            plan_summary=plan_text,
            prompt_text=prompt_text,
            reason_text=infer_reason,
            action_payload=action_payload,
            token_usage=token_usage,
        )
        successful_rounds += 1
        improved = used_time < best_time
        if improved:
            best_time = used_time
            best_hint = applied_hint
            best_plan = plan_text
            best_round = round_idx
            best_reason = infer_reason

        print(
            f"Round {round_idx} [{phase}] source={source_id} "
            f"time={used_time:.2f} ms {'(improved)' if improved else ''}"
        )

        regression_analysis.append(
            {
                "round_idx": round_idx,
                "phase": phase,
                "source_id": source_id,
                "demo_template_id": demo_template_id,
                "applied_hint": applied_hint,
                "execution_time_ms": used_time,
                "hint_generation_time_seconds": llm_sec,
                "token_usage": token_usage or {},
                "is_best_so_far": improved,
                "improves_over_baseline": used_time < baseline_time,
            }
        )

        t_base = float(baseline_time)
        t_used = float(used_time)
        u = math.log(t_base / t_used)
        utility_trials.append(
            {
                "target_id": target_id,
                "round_idx": round_idx,
                "phase": phase,
                "source_id": source_id,
                "source_plan_json": source_plan_json,
                "target_plan_json": str((query_out_dir / "0" / "plan.json").resolve()),
                "trial_plan_json": str(trial_plan_json_path.resolve()),
                "t_base_ms": t_base,
                "t_used_ms": t_used,
                "u": u,
                "applied_hint": applied_hint,
                "demo_template_id": demo_template_id,
                "is_best_so_far": improved,
            }
        )

    if empty_reuse_source_id is not None:
        source_plan_json = str((target_pool / empty_reuse_source_id / "0" / "plan.json").resolve())
        utility_trials.append(
            {
                "target_id": target_id,
                "round_idx": 0,
                "phase": "reuse_empty",
                "source_id": empty_reuse_source_id,
                "source_plan_json": source_plan_json,
                "target_plan_json": str((query_out_dir / "0" / "plan.json").resolve()),
                "trial_plan_json": str((query_out_dir / "0" / "plan.json").resolve()),
                "t_base_ms": float(baseline_time),
                "t_used_ms": float(baseline_time),
                "u": 0.0,
                "applied_hint": "/*+ */",
                "demo_template_id": str(template_num),
                "is_best_so_far": False,
            }
        )
        regression_analysis.append(
            {
                "round_idx": 0,
                "phase": "reuse_empty",
                "source_id": empty_reuse_source_id,
                "applied_hint": "/*+ */",
                "execution_time_ms": float(baseline_time),
                "hint_generation_time_seconds": 0.0,
                "is_best_so_far": False,
                "synthetic_zero_utility": True,
            }
        )

    if successful_rounds == 0:
        query_failure_reason = (
            "; ".join(round_failure_messages)
            if round_failure_messages
            else "no_successful_rounds"
        )
        (query_out_dir / "infer_reason.txt").write_text(query_failure_reason, encoding="utf-8")
        (query_out_dir / "regression_analysis.json").write_text(
            json.dumps(regression_analysis, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {"best_time": baseline_time, "baseline_time": baseline_time}, []

    (query_out_dir / "suggest_hint.txt").write_text(best_hint, encoding="utf-8")
    (query_out_dir / "execution_plan.txt").write_text(best_plan, encoding="utf-8")
    (query_out_dir / "infer_reason.txt").write_text(best_reason, encoding="utf-8")
    (query_out_dir / "regression_analysis.json").write_text(
        json.dumps(regression_analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (query_out_dir / "best_global_hint.txt").write_text("", encoding="utf-8")
    (query_out_dir / "best_leading_hint.txt").write_text("", encoding="utf-8")
    (query_out_dir / "best_node_hint.txt").write_text("", encoding="utf-8")
    (query_out_dir / "final_combined_hint.txt").write_text(best_hint, encoding="utf-8")

    print(
        f"Best for {target_id}: round={best_round}, baseline={baseline_time:.2f} ms, "
        f"best={best_time:.2f} ms, speedup={(baseline_time / best_time if best_time > 0 else 0.0):.3f}x"
    )
    return {"best_time": best_time, "baseline_time": baseline_time}, utility_trials


def finalize_utility_trials(
    all_trials: List[Dict[str, Any]],
    target_pool: Path,
    alpha: float,
    seed: int,
    random_rounds: int,
) -> None:
    abs_u = [abs(float(r["u"])) for r in all_trials]
    tau = max(median(abs_u), 1e-8)
    for row in all_trials:
        y = math.tanh(float(row["u"]) / tau)
        row["y"] = y
        row["weight"] = 1.0 + alpha * abs(y)

    all_trials.sort(key=lambda r: (r["target_id"], int(r["round_idx"])))
    jsonl_path = target_pool / "utility_trials.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in all_trials:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "rows": len(all_trials),
        "tau": tau,
        "alpha": alpha,
        "seed": seed,
        "random_rounds": random_rounds,
    }
    (target_pool / "utility_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Utility trials written: {jsonl_path} (rows={len(all_trials)}, tau={tau:.8f})")


async def main_async(args: argparse.Namespace) -> None:
    target_pool = Path(args.target_pool)
    train_query_dir = Path(args.train_query_dir)
    target_pool.mkdir(parents=True, exist_ok=True)

    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    user_prompt_template = Path(args.user_prompt).read_text(encoding="utf-8")

    sql_files = [p for p in train_query_dir.iterdir() if p.is_file() and p.name.endswith(".sql")]
    sql_files = [p for p in sql_files if SQL_FILE_RE.match(p.name)]
    sql_files = [train_query_dir / s for s in sort_sql_files([p.name for p in sql_files])]
    if args.max_files > 0:
        sql_files = sql_files[: args.max_files]

    summary: Dict[str, Dict[str, float]] = {}
    all_trials: List[Dict[str, Any]] = []
    failed_queries: List[Dict[str, str]] = []
    existing_trials: List[Dict[str, Any]] = []
    if args.resume_missing:
        existing_trials = load_existing_utility_trials(target_pool)
        print(f"[resume_missing] loaded existing utility rows: {len(existing_trials)}")

    a_files: List[Path] = []
    rest_files: List[Path] = []
    for p in sql_files:
        _, suffix = parse_sql_filename(p.name)
        if suffix == "a":
            a_files.append(p)
        else:
            rest_files.append(p)

    if a_files:
        print(f"\n=== Stage A: generate_pool flow for {len(a_files)} *a.sql ===")
        for sql_file in a_files:
            if args.resume_missing and query_completed(target_pool, sql_file.stem):
                print(f"[resume_missing] skip existing Stage-A query: {sql_file.stem}")
                continue
            result = await run_a_query_with_generate_pool(sql_file=sql_file, args=args, target_pool=target_pool)
            summary[sql_file.stem] = result

    # After Stage A, demos for subsequent rounds should come from generated pool results.
    runtime_source_pool = target_pool

    if rest_files:
        print(f"\n=== Stage B+: reuse + random flow for {len(rest_files)} non-a queries ===")
    for sql_file in rest_files:
        if args.resume_missing and query_completed(target_pool, sql_file.stem):
            print(f"[resume_missing] skip existing Stage-B query: {sql_file.stem}")
            continue
        try:
            result, trials = await run_single_query(
                sql_file=sql_file,
                args=args,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                source_pool=runtime_source_pool,
                target_pool=target_pool,
            )
            summary[sql_file.stem] = result
            all_trials.extend(trials)
            if not trials:
                infer_reason_text = read_text(target_pool / sql_file.stem / "infer_reason.txt", "all_rounds_failed")
                failed_queries.append(
                    {
                        "query_id": sql_file.stem,
                        "reason": infer_reason_text,
                    }
                )
        except Exception as exc:
            failed_queries.append(
                {
                    "query_id": sql_file.stem,
                    "reason": str(exc),
                }
            )
            print(f"[FAIL] query={sql_file.stem} error={exc}")
            continue

    with open(target_pool / "log-new1.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "best_time_ms", "baseline_time_ms", "speedup"])
        for k in sorted(summary.keys(), key=lambda x: parse_sql_filename(f"{x}.sql")):
            best = summary[k]["best_time"]
            base = summary[k]["baseline_time"]
            speedup = base / best if best > 0 else 0.0
            writer.writerow([k, f"{best:.2f}", f"{base:.2f}", f"{speedup:.2f}"])

    if args.resume_missing and existing_trials:
        updated_target_ids = {str(r.get("target_id", "")) for r in all_trials}
        kept_existing = [r for r in existing_trials if str(r.get("target_id", "")) not in updated_target_ids]
        all_trials = kept_existing + all_trials
        print(
            f"[resume_missing] merge utility rows: kept_existing={len(kept_existing)} "
            f"new={len(all_trials) - len(kept_existing)} total={len(all_trials)}"
        )

    finalize_utility_trials(
        all_trials=all_trials,
        target_pool=target_pool,
        alpha=args.alpha,
        seed=args.seed,
        random_rounds=args.random_rounds,
    )
    if failed_queries:
        failed_path = target_pool / "failed_queries.json"
        failed_path.write_text(
            json.dumps(failed_queries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Failed queries written: {failed_path} (rows={len(failed_queries)})")
    print(f"Done. Output pool: {target_pool}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
