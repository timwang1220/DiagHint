#!/usr/bin/env python3
"""Generate pairwise training samples from plan artifacts.

This script is designed to be a standalone CLI entrypoint and keeps the output
format minimal:
    {"plan_a": "...", "plan_b": "...", "score": 0.73, "label": 0.73}

Core behavior:
1) Recursively scan --plan_dir for plan artifacts (.json/.jsonl).
2) Group plans by query.
3) Infer best plan per query.
4) Build pairs using fixed negative sampling (--num_negative).
5) Write JSONL to --out_dir/pairs.jsonl.

Hint subtraction semantics are intentionally strict:
    hint_x - hint_y = (x_only) U (ANTI::y_only)
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ANTI_PREFIX = "ANTI::"
DEFAULT_SEED = 42

# pg_hint_plan categories.
LEADING_HINTS = {
    "LEADING",
}

SET_HINTS = {
    "SET",
}

SCAN_HINTS = {
    "SEQSCAN",
    "NOSEQSCAN",
    "INDEXSCAN",
    "NOINDEXSCAN",
    "INDEXONLYSCAN",
    "NOINDEXONLYSCAN",
    "BITMAPSCAN",
    "NOBITMAPSCAN",
    "TIDSCAN",
    "NOTIDSCAN",
}

JOIN_HINTS = {
    "NESTLOOP",
    "NONESTLOOP",
    "HASHJOIN",
    "NOHASHJOIN",
    "MERGEJOIN",
    "NOMERGEJOIN",
}

CATEGORY_ORDER = ("leading", "set", "scan", "join")


@dataclass
class StructuredHintSet:
    leading: set[str]
    set_hint: set[str]
    scan: set[str]
    join: set[str]
    other: set[str]

    def as_dict(self) -> dict[str, set[str]]:
        return {
            "leading": self.leading,
            "set": self.set_hint,
            "scan": self.scan,
            "join": self.join,
            "other": self.other,
        }


@dataclass
class HintDiff:
    positive: dict[str, set[str]]
    negative: dict[str, set[str]]

    def all_tokens(self) -> set[str]:
        tokens: set[str] = set()
        for category in CATEGORY_ORDER:
            tokens.update(self.positive.get(category, set()))
            tokens.update(
                f"{ANTI_PREFIX}{token}" for token in self.negative.get(category, set())
            )
        tokens.update(self.positive.get("other", set()))
        tokens.update(
            f"{ANTI_PREFIX}{token}" for token in self.negative.get("other", set())
        )
        return tokens


@dataclass
class PlanRecord:
    query_id: str
    plan_id: str
    hint: str
    is_best: bool
    quality_value: float | None
    raw: dict[str, Any]

@dataclass
class PlanLinearFeatures:
    relation_order: list[str]
    join_methods: list[str]
    scan_methods: list[str]


def _normalize_join_method(node_type: str) -> str:
    mapping = {
        "Nested Loop": "NestLoop",
        "Hash Join": "HashJoin",
        "Merge Join": "MergeJoin",
    }
    return mapping.get(node_type, node_type.replace(" ", ""))


def _normalize_scan_method(node_type: str) -> str:
    mapping = {
        "Seq Scan": "SeqScan",
        "Index Scan": "IndexScan",
        "Index Only Scan": "IndexOnlyScan",
        "Bitmap Heap Scan": "BitmapScan",
        "Bitmap Index Scan": "BitmapScan",
        "Tid Scan": "TidScan",
        "CTE Scan": "CTEScan",
        "Subquery Scan": "SubqueryScan",
    }
    return mapping.get(node_type, node_type.replace(" ", ""))


def _is_join_node(node_type: str) -> bool:
    return node_type in {"Nested Loop", "Hash Join", "Merge Join"}


def _relation_name(node: dict[str, Any], fallback: str) -> str | None:
    alias = node.get("Alias")
    rel = node.get("Relation Name")
    if isinstance(alias, str) and alias.strip():
        return alias.strip()
    if isinstance(rel, str) and rel.strip():
        return rel.strip()
    return fallback


def _linearize_plan_node(
    node: dict[str, Any],
    anon_counter: list[int],
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    node_type = str(node.get("Node Type", ""))
    children = node.get("Plans", [])
    if not isinstance(children, list):
        children = []

    # Base table scan node.
    if "Scan" in node_type and (node.get("Alias") or node.get("Relation Name")):
        anon_counter[0] += 1
        rel = _relation_name(node, f"anon_{anon_counter[0]}")
        if rel is None:
            return [], [], []
        return [rel], [], [(rel, _normalize_scan_method(node_type))]

    # Join node: fold left-to-right over child plans and append join method at each merge.
    if _is_join_node(node_type) and children:
        base_order, base_joins, base_scans = _linearize_plan_node(children[0], anon_counter)
        for child in children[1:]:
            ch_order, ch_joins, ch_scans = _linearize_plan_node(child, anon_counter)
            base_order.extend(ch_order)
            base_joins.extend(ch_joins)
            base_joins.append(_normalize_join_method(node_type))
            base_scans.extend(ch_scans)
        return base_order, base_joins, base_scans

    # Non-join internal node: concatenate child outputs.
    rel_order: list[str] = []
    join_methods: list[str] = []
    scan_pairs: list[tuple[str, str]] = []
    for child in children:
        c_order, c_joins, c_scans = _linearize_plan_node(child, anon_counter)
        rel_order.extend(c_order)
        join_methods.extend(c_joins)
        scan_pairs.extend(c_scans)

    return rel_order, join_methods, scan_pairs


def extract_plan_linear_features(plan: dict[str, Any]) -> PlanLinearFeatures:
    """Extract linear relation/join/scan sequences from a PostgreSQL plan tree.

    Returns:
      - relation_order: length n
      - join_methods: length n-1 for a standard binary join tree
      - scan_methods: length n, aligned with relation_order
    """
    rel_order, join_methods, scan_pairs = _linearize_plan_node(plan, [0])
    scan_by_rel = {r: s for r, s in scan_pairs}
    scan_methods = [scan_by_rel.get(r, "UnknownScan") for r in rel_order]
    return PlanLinearFeatures(
        relation_order=rel_order,
        join_methods=join_methods,
        scan_methods=scan_methods,
    )


def extract_plan_linear_features_from_file(plan_path: Path) -> PlanLinearFeatures:
    obj = _load_json(plan_path)
    if isinstance(obj, dict):
        return extract_plan_linear_features(obj)
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return extract_plan_linear_features(obj[0])
    return PlanLinearFeatures(relation_order=[], join_methods=[], scan_methods=[])


def validate_plan_feature_extraction(plan_dir: Path, max_examples: int = 3) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    paths = sorted(plan_dir.rglob("plan.json"))[:max_examples]
    for p in paths:
        feat = extract_plan_linear_features_from_file(p)
        n = len(feat.relation_order)
        out.append(
            {
                "path": str(p),
                "n": n,
                "relation_order": feat.relation_order,
                "join_methods": feat.join_methods,
                "scan_methods": feat.scan_methods,
                "length_ok": (len(feat.join_methods) == max(0, n - 1) and len(feat.scan_methods) == n),
            }
        )
    return out


# TODO: Replace ANTI/no-hint cancellation with a true plan-aware cancellation
# based on structural deltas from extracted plan linear features.

def subtract_hints_plan_aware_todo(hint_x: str, hint_y: str, plan_x: dict[str, Any], plan_y: dict[str, Any]) -> HintDiff:
    return subtract_hints(hint_x, hint_y)


def _extract_hint_body(hint_str: str) -> str:
    if not hint_str:
        return ""
    match = re.search(r"/\*\+\s*(.*?)\s*\*/", hint_str, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return hint_str.strip()


def _split_hint_tokens(hint_body: str) -> list[str]:
    # Split by spaces while preserving parenthesized payloads.
    tokens: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in hint_body:
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


def _hint_head(token: str) -> str:
    t = token.strip()
    if "(" in t:
        return t.split("(", 1)[0].upper()
    return t.upper()


def _categorize_token(token: str) -> str:
    head = _hint_head(token)
    if head in LEADING_HINTS:
        return "leading"
    if head in SET_HINTS:
        return "set"
    if head in SCAN_HINTS:
        return "scan"
    if head in JOIN_HINTS:
        return "join"
    return "other"


def parse_hint_tokens(hint_str: str) -> StructuredHintSet:
    body = _extract_hint_body(hint_str)
    tokens = _split_hint_tokens(body)

    bucket: dict[str, set[str]] = {k: set() for k in (*CATEGORY_ORDER, "other")}
    for token in tokens:
        bucket[_categorize_token(token)].add(token)

    return StructuredHintSet(
        leading=bucket["leading"],
        set_hint=bucket["set"],
        scan=bucket["scan"],
        join=bucket["join"],
        other=bucket["other"],
    )


def _stable_serialize(
    positive: dict[str, set[str]], negative: dict[str, set[str]], anti_prefix: str = ANTI_PREFIX
) -> str:
    ordered: list[str] = []
    for category in (*CATEGORY_ORDER, "other"):
        pos = sorted(positive.get(category, set()))
        neg = sorted(negative.get(category, set()))
        ordered.extend(pos)
        ordered.extend(f"{anti_prefix}{token}" for token in neg)
    return " ".join(ordered).strip()


def subtract_hints(hint_x: str, hint_y: str) -> HintDiff:
    x = parse_hint_tokens(hint_x).as_dict()
    y = parse_hint_tokens(hint_y).as_dict()

    positive: dict[str, set[str]] = {}
    negative: dict[str, set[str]] = {}

    for category in (*CATEGORY_ORDER, "other"):
        x_only = x.get(category, set()) - y.get(category, set())
        y_only = y.get(category, set()) - x.get(category, set())
        positive[category] = set(x_only)
        negative[category] = set(y_only)

    return HintDiff(positive=positive, negative=negative)


def _hint_distance(hint: str, best_hint: str) -> float:
    diff = subtract_hints(hint, best_hint)
    return float(len(diff.all_tokens()))


def _fallback_pair_score(hint_a: str, hint_b: str, best_hint: str) -> float:
    d_a = _hint_distance(hint_a, best_hint)
    d_b = _hint_distance(hint_b, best_hint)
    if d_a == 0.0 and d_b == 0.0:
        return 1.0
    return max(0.0, 1.0 - abs(d_a - d_b) / (d_a + d_b + 1e-12))


def _load_legacy_pair_scorer() -> Callable[[str, str, str], float] | None:
    try:
        import encoder  # type: ignore
    except Exception:
        return None

    candidate_names = (
        "compute_pair_score",
        "pair_score",
        "compute_similarity",
        "similarity_score",
    )
    for name in candidate_names:
        fn = getattr(encoder, name, None)
        if callable(fn):
            def wrapped(a: str, b: str, best: str, _fn: Callable[..., Any] = fn) -> float:
                sig = inspect.signature(_fn)
                kwargs_variants = [
                    {"hint_a": a, "hint_b": b, "hint_best": best},
                    {"plan_a": a, "plan_b": b, "best_plan": best},
                    {"a": a, "b": b, "best": best},
                ]

                for kwargs in kwargs_variants:
                    usable = {k: v for k, v in kwargs.items() if k in sig.parameters}
                    if not usable:
                        continue
                    try:
                        res = _fn(**usable)
                        return float(res)
                    except Exception:
                        continue

                try:
                    return float(_fn(a, b, best))
                except Exception:
                    return _fallback_pair_score(a, b, best)

            return wrapped
    return None


def compute_pair_score(
    plan_a: PlanRecord,
    plan_b: PlanRecord,
    best_plan: PlanRecord,
    legacy_pair_scorer: Callable[[str, str, str], float] | None = None,
) -> float:
    if legacy_pair_scorer is not None:
        try:
            return float(legacy_pair_scorer(plan_a.hint, plan_b.hint, best_plan.hint))
        except Exception:
            pass
    return _fallback_pair_score(plan_a.hint, plan_b.hint, best_plan.hint)


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "best"}
    return False


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _pick_first(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _extract_hint_from_record(item: dict[str, Any]) -> str:
    direct = _pick_first(
        item,
        [
            "hint",
            "hints",
            "plan_hint",
            "optimizer_hint",
            "hint_str",
            "sql_hint",
        ],
    )
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    sql_text = _pick_first(item, ["sql", "query", "statement"])
    if isinstance(sql_text, str) and sql_text.strip():
        match = re.search(r"/\*\+.*?\*/", sql_text, flags=re.DOTALL)
        if match:
            return match.group(0)
    return ""


def _quality_from_record(item: dict[str, Any]) -> float | None:
    # Lower is better.
    for key in ["rank", "latency", "runtime", "cost", "execution_time", "total_cost"]:
        v = _to_float(item.get(key))
        if v is not None:
            return v
    return None


def _normalize_plan_item(
    item: dict[str, Any], default_query_id: str, fallback_plan_id: str
) -> PlanRecord | None:
    hint = _extract_hint_from_record(item)
    if not hint:
        return None

    query_id = str(_pick_first(item, ["query_id", "qid", "query_name"]) or default_query_id)
    plan_id = str(_pick_first(item, ["plan_id", "id", "plan_name"]) or fallback_plan_id)
    is_best = _to_bool(_pick_first(item, ["is_best", "best", "is_best_plan"]))
    quality = _quality_from_record(item)
    return PlanRecord(
        query_id=query_id,
        plan_id=plan_id,
        hint=hint,
        is_best=is_best,
        quality_value=quality,
        raw=item,
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except Exception:
                continue
    return rows


def _iter_plan_rows(plan_dir: Path) -> list[PlanRecord]:
    records: list[PlanRecord] = []
    files = sorted(
        p
        for p in plan_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
    )

    for path in files:
        default_query_id = path.stem
        rows: list[dict[str, Any]] = []
        if path.suffix.lower() == ".jsonl":
            rows = _load_jsonl(path)
        else:
            obj = _load_json(path)
            if isinstance(obj, list):
                rows = [x for x in obj if isinstance(x, dict)]
            elif isinstance(obj, dict):
                if "plans" in obj and isinstance(obj["plans"], list):
                    rows = [x for x in obj["plans"] if isinstance(x, dict)]
                    default_query_id = str(
                        _pick_first(obj, ["query_id", "qid", "query_name"]) or default_query_id
                    )
                else:
                    rows = [obj]

        for idx, item in enumerate(rows):
            rec = _normalize_plan_item(
                item=item,
                default_query_id=default_query_id,
                fallback_plan_id=f"{path.stem}_{idx}",
            )
            if rec is not None:
                records.append(rec)
    return records


def _group_by_query(records: list[PlanRecord]) -> dict[str, list[PlanRecord]]:
    grouped: dict[str, list[PlanRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.query_id, []).append(rec)
    return grouped


def _pick_best_plan(plans: list[PlanRecord]) -> PlanRecord:
    best_marked = [p for p in plans if p.is_best]
    if best_marked:
        if len(best_marked) == 1:
            return best_marked[0]
        return sorted(
            best_marked,
            key=lambda p: (
                p.quality_value if p.quality_value is not None else float("inf"),
                p.plan_id,
            ),
        )[0]

    with_quality = [p for p in plans if p.quality_value is not None]
    if with_quality:
        return sorted(with_quality, key=lambda p: (p.quality_value, p.plan_id))[0]

    return sorted(plans, key=lambda p: p.plan_id)[0]


def build_pairs_for_query(
    plans: list[PlanRecord],
    num_negative: int,
    rng: random.Random,
    legacy_pair_scorer: Callable[[str, str, str], float] | None = None,
) -> list[dict[str, Any]]:
    if len(plans) < 2:
        return []

    best = _pick_best_plan(plans)
    others = [p for p in plans if p.plan_id != best.plan_id]
    if not others:
        return []

    out: list[dict[str, Any]] = []
    for anchor in others:
        candidates = [p for p in plans if p.plan_id != anchor.plan_id]
        scored: list[tuple[PlanRecord, float]] = []
        for cand in candidates:
            score = compute_pair_score(
                anchor, cand, best, legacy_pair_scorer=legacy_pair_scorer
            )
            scored.append((cand, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        positive = scored[0]

        neg_pool = scored[1:] if len(scored) > 1 else []
        if neg_pool:
            if len(neg_pool) <= num_negative:
                negatives = neg_pool
            else:
                negatives = rng.sample(neg_pool, k=num_negative)
        else:
            negatives = []

        pos_score = float(positive[1])
        out.append(
            {
                "plan_a": anchor.hint,
                "plan_b": positive[0].hint,
                "score": pos_score,
                "label": pos_score,
            }
        )

        for neg_plan, neg_score in negatives:
            score_f = float(neg_score)
            out.append(
                {
                    "plan_a": anchor.hint,
                    "plan_b": neg_plan.hint,
                    "score": score_f,
                    "label": score_f,
                }
            )

    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset(
    plan_dir: Path,
    out_dir: Path,
    num_negative: int,
    seed: int,
) -> Path:
    rng = random.Random(seed)
    records = _iter_plan_rows(plan_dir)
    grouped = _group_by_query(records)
    legacy_pair_scorer = _load_legacy_pair_scorer()

    rows: list[dict[str, Any]] = []
    for qid in sorted(grouped.keys()):
        plans = grouped[qid]
        rows.extend(
            build_pairs_for_query(
                plans=plans,
                num_negative=num_negative,
                rng=rng,
                legacy_pair_scorer=legacy_pair_scorer,
            )
        )

    out_path = out_dir / "pairs.jsonl"
    _write_jsonl(out_path, rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pairwise training data with strict hint subtraction."
    )
    parser.add_argument("--plan_dir", required=True, type=Path, help="Input plan directory.")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--num_negative",
        type=int,
        default=6,
        help="Number of negatives per anchor.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Reserved for compatibility with existing CLI calls.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--validate_plan_features",
        action="store_true",
        help="Only validate plan linear feature extraction on sample plan.json files.",
    )
    parser.add_argument(
        "--validate_num",
        type=int,
        default=3,
        help="How many plan.json files to validate when --validate_plan_features is set.",
    )
    args = parser.parse_args()

    if args.num_negative < 0:
        raise ValueError("--num_negative must be >= 0")
    if not args.plan_dir.exists():
        raise FileNotFoundError(f"plan_dir not found: {args.plan_dir}")

    if args.validate_plan_features:
        rows = validate_plan_feature_extraction(args.plan_dir, max_examples=args.validate_num)
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        return

    out_path = build_dataset(
        plan_dir=args.plan_dir,
        out_dir=args.out_dir,
        num_negative=args.num_negative,
        seed=args.seed,
    )
    print(str(out_path))


if __name__ == "__main__":
    hint1 = "/*+ Leading(t1 t2 t3) Set(enable_nestloop off) HashJoin(t1 t2) */"
    hint2 = "/*+ Leading(t1 t2) Set(enable_mergejoin off) IndexScan(t1 idx1) */"
    diff = subtract_hints(hint1, hint2)
    print("Positive:", diff.positive)
    print("Negative:", diff.negative)
    distance = _hint_distance(hint1, hint2)
    print("Hint distance:", distance)
