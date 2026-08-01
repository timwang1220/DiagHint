# predicate.py
# Predicate extraction utilities for PostgreSQL execution plans
# Extracts columns, operators, and values from plan node predicates
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set


# Operators we want to extract
OPERATORS = [
    "=", "<>", "!=", "<", ">", "<=", ">=",
    "LIKE", "ILIKE", "~~", "~~*",  # ~~ is PostgreSQL's LIKE operator
    "!~~", "!~~*",                  # NOT LIKE
    "IN", "NOT IN",
    "IS NULL", "IS NOT NULL",
    "BETWEEN",
    "@>", "<@",  # JSON/array operators
    "&&",        # Array overlap
    "||",        # Array concatenation
    "&", "|", "#",  # Geometric/path operators
    "~", "~*", "!~", "!~*",  # Regex matching (same as LIKE)
    "SIMILAR TO",
]


def extract_predicates_from_node(node: Dict) -> List[str]:
    """
    Extract all predicate strings from a plan node.

    Looks for these keys in order:
    - Filter
    - Index Cond
    - Hash Cond
    - Join Filter
    - Merge Cond
    - Qual
    - Recheck Cond

    Args:
        node: Plan node dictionary

    Returns:
        List of predicate strings
    """
    predicates = []

    # Keys that contain predicate information
    predicate_keys = [
        "Filter",
        "Index Cond",
        "Hash Cond",
        "Join Filter",
        "Merge Cond",
        "Qual",
        "Recheck Cond",
    ]

    for key in predicate_keys:
        if key in node and node[key]:
            predicates.append(str(node[key]))

    return predicates


def parse_predicate_elements(predicate_text: str) -> Tuple[Set[str], Set[str], List[str]]:
    """
    Parse columns, operators, and values from a predicate string.

    Examples:
        "(mc.movie_id = t.id)" -> columns={'mc.movie_id', 't.id'}, ops={'='}, values=[]
        "(info)::text = 'top 250 rank'::text" -> columns={'info'}, ops={'='}, values=['top 250 rank']
        "salary > 50000" -> columns={'salary'}, ops={'>'}, values=['50000']

    Args:
        predicate_text: Predicate string

    Returns:
        Tuple of (columns set, operators set, values list)
    """
    if not predicate_text:
        return set(), set(), []

    columns = set()
    operators = set()
    values = []

    # Clean up the predicate text
    text = predicate_text.strip()

    # Remove outer parentheses if present
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()

    # Extract operators first
    for op in OPERATORS:
        if op in text.upper() or op in text:
            # Check if this operator is actually present (not part of a larger word)
            pattern = r'(?<![a-zA-Z0-9_])' + re.escape(op) + r'(?![a-zA-Z0-9_])'
            if re.search(pattern, text, re.IGNORECASE):
                operators.add(op)

    # Extract column references (table.column or column)
    # Pattern: optional table name/alias + dot + column name
    # Column names typically start with a letter and contain alphanumeric chars and underscores
    column_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\b'

    # SQL keywords and data type names to filter out
    sql_keywords = {
        'AND', 'OR', 'NOT', 'NULL', 'TRUE', 'FALSE',
        'IS', 'IN', 'LIKE', 'BETWEEN', 'EXISTS',
        'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
        'COALESCE', 'NULLIF', 'CAST', 'EXTRACT',
        'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
        'text', 'integer', 'numeric', 'date', 'timestamp', 'bpchar',
        'AS', 'BY', 'ORDER', 'GROUP', 'HAVING', 'WHERE', 'FROM',
        'JOIN', 'INNER', 'OUTER', 'LEFT', 'RIGHT', 'FULL', 'CROSS',
        'ON', 'USING', 'SELECT', 'DISTINCT', 'ALL',
    }

    # Find all potential column references
    potential_columns = re.findall(column_pattern, text)
    for col in potential_columns:
        col_upper = col.upper()
        # Filter out SQL keywords and data types
        if col_upper in sql_keywords:
            continue
        # Filter out standalone numbers
        if re.match(r'^\d+$', col):
            continue
        # Filter out common function names followed by (
        idx = text.find(col)
        if idx >= 0 and idx + len(col) < len(text):
            next_char = text[idx + len(col)]
            if next_char == '(':
                continue
        # Only add if it looks like a proper column reference
        # - Either has a dot (table.column)
        # - Or is all alphanumeric with underscores
        if '.' in col:
            # Split and check both parts
            parts = col.split('.')
            if len(parts) == 2:
                table_ref, col_name = parts
                # Both should be valid identifiers
                if table_ref and col_name:
                    # Check that neither part is a SQL keyword
                    if table_ref.upper() not in sql_keywords and col_name.upper() not in sql_keywords:
                        columns.add(col)
        else:
            # Single column name - add if it's not a keyword
            if col_upper not in sql_keywords and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', col):
                columns.add(col)

    # Extract values
    # String literals in single quotes
    string_pattern = r"'([^']*)'"
    string_values = re.findall(string_pattern, text)
    values.extend(string_values)

    # Numeric values
    # Match integers and floats, but not version numbers or identifiers
    numeric_pattern = r'(?<![a-zA-Z0-9_])(\d+\.?\d*)(?![a-zA-Z0-9_])'
    numeric_values = re.findall(numeric_pattern, text)
    for num in numeric_values:
        # Only add if it's not part of something like "3.14159::numeric"
        # and if it's a reasonable number (not huge like timestamps)
        try:
            fnum = float(num)
            if 0 < fnum < 1e10:  # Filter out unreasonable values
                values.append(num)
        except ValueError:
            pass

    return columns, operators, values


def build_vocabularies(nodes: List[Dict], min_column_count: int = 1, min_operator_count: int = 1):
    """
    Build vocabularies for columns and operators from a list of nodes.

    The nodes are expected to have a 'predicates' key with a list of predicate strings.
    If not, it will try to extract predicates using extract_predicates_from_node.

    Args:
        nodes: List of plan node dictionaries
        min_column_count: Minimum frequency for a column to be included
        min_operator_count: Minimum frequency for an operator to be included

    Returns:
        Tuple of (column2id dict, operator2id dict)
    """
    column_counts = defaultdict(int)
    operator_counts = defaultdict(int)

    for node in nodes:
        # Use the 'predicates' key if it exists, otherwise extract from node
        if "predicates" in node:
            predicates = node.get("predicates", [])
        else:
            predicates = extract_predicates_from_node(node)

        for pred in predicates:
            cols, ops, _ = parse_predicate_elements(pred)
            for col in cols:
                column_counts[col] += 1
            for op in ops:
                operator_counts[op] += 1

    # Filter by minimum count and create mappings
    column2id = {}
    idx = 0
    for col, count in sorted(column_counts.items(), key=lambda x: x[1], reverse=True):
        if count >= min_column_count:
            column2id[col] = idx
            idx += 1

    operator2id = {}
    idx = 0
    for op, count in sorted(operator_counts.items(), key=lambda x: x[1], reverse=True):
        if count >= min_operator_count:
            operator2id[op] = idx
            idx += 1

    return column2id, operator2id


def hash_value(value: str, mod: int) -> int:
    """
    Stable hash function for mapping values to embedding indices.

    Args:
        value: String value to hash
        mod: Modulus for the hash

    Returns:
        Hash value in range [0, mod)
    """
    if not value:
        return 0
    return abs(hash(value)) % mod


if __name__ == "__main__":
    # Test the predicate extraction
    test_predicates = [
        "(mc.movie_id = t.id)",
        "(info)::text = 'top 250 rank'::text",
        "salary > 50000 AND salary < 100000",
        "name LIKE '%test%'",
        "(mc.company_type_id = ct.id)",
        "((note)::text !~~ '%(as Metro-Goldwyn-Mayer Pictures)%'::text)",
    ]

    print("Testing predicate parsing:")
    for pred in test_predicates:
        cols, ops, vals = parse_predicate_elements(pred)
        print(f"\nPredicate: {pred}")
        print(f"  Columns: {cols}")
        print(f"  Operators: {ops}")
        print(f"  Values: {vals}")
