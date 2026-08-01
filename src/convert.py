#!/usr/bin/env python3
"""Convert between structured hint JSON and pg_hint_plan strings."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


HINT_BLOCK_RE = re.compile(r"/\*\+\s*(.*?)\s*\*/", flags=re.DOTALL)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*")


def json_to_pg_hint(items: List[Dict[str, Any]]) -> str:
    """Convert structured hint items to one pg_hint_plan block."""
    parts: List[str] = []
    for item in items:
        item_type = str(item.get("type", "")).strip().lower()
        if item_type == "global":
            guc = str(item["guc"]).strip()
            value = str(item["value"]).strip()
            parts.append(f"Set({guc} {value})")
        elif item_type == "leading":
            relations = [str(x).strip() for x in item.get("relations", []) if str(x).strip()]
            parts.append(f"Leading({' '.join(relations)})")
        elif item_type == "join":
            method = str(item["method"]).strip()
            left = [str(x).strip() for x in item.get("left", []) if str(x).strip()]
            right = [str(x).strip() for x in item.get("right", []) if str(x).strip()]
            relations = left + right
            if not left or not right:
                raise ValueError("Join hint items must contain non-empty 'left' and 'right'")
            parts.append(f"{method}({' '.join(relations)})")
        elif item_type == "scan":
            method = str(item["method"]).strip()
            relation = str(item["relation"]).strip()
            parts.append(f"{method}({relation})")
        else:
            raise ValueError(f"Unsupported hint item type: {item_type}")
    if not parts:
        return "/*+ */"
    return f"/*+ {' '.join(parts)} */"


def pg_hint_to_json(hint: str) -> List[Dict[str, Any]]:
    """Parse one pg_hint_plan block into structured hint items."""
    match = HINT_BLOCK_RE.search(hint.strip())
    if not match:
        raise ValueError("Input does not contain a valid /*+ ... */ hint block")

    body = match.group(1).strip()
    tokens = TOKEN_RE.findall(body)
    out: List[Dict[str, Any]] = []

    for token in tokens:
        name, raw_args = token.split("(", 1)
        args = raw_args[:-1].strip()
        if name == "Set":
            parts = args.split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid Set() token: {token}")
            out.append(
                {
                    "type": "global",
                    "guc": parts[0],
                    "value": parts[1],
                }
            )
        elif name == "Leading":
            relations = [x for x in args.split() if x]
            out.append(
                {
                    "type": "leading",
                    "relations": relations,
                }
            )
        elif name in {"SeqScan", "IndexScan", "IndexOnlyScan", "BitmapScan", "TidScan"}:
            out.append(
                {
                    "type": "scan",
                    "method": name,
                    "relation": args,
                }
            )
        else:
            relations = [x for x in args.split() if x]
            if len(relations) < 2:
                raise ValueError(f"Join hint must contain at least 2 relations: {token}")
            out.append(
                {
                    "type": "join",
                    "method": name,
                    "left": relations[:-1],
                    "right": [relations[-1]],
                }
            )
    return out


def main() -> None:
    example = [
        {
            "type": "global",
            "guc": "enable_nestloop",
            "value": "off",
        },
        {
            "type": "leading",
            "relations": ["a", "b", "c"],
        },
        {
            "type": "join",
            "method": "HashJoin",
            "left": ["a", "b"],
            "right": ["c"],
        },
        {
            "type": "scan",
            "method": "SeqScan",
            "relation": "c",
        },
    ]

    hint = json_to_pg_hint(example)
    parsed = pg_hint_to_json(hint)

    print("Input JSON:")
    print(json.dumps(example, indent=2, ensure_ascii=False))
    print("\nPG Hint:")
    print(hint)
    print("\nParsed Back:")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
