"""Estimate-only features shared by export and inference; no DB/LLM imports."""
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime

import sqlglot
from sqlglot import exp

JOIN_TYPES = {'Nested Loop', 'Hash Join', 'Merge Join'}
PREDICATES = ('Filter', 'Index Cond', 'Recheck Cond', 'Hash Cond', 'Merge Cond', 'Join Filter')
ESTIMATE_KEYS = {'Node Type', 'Parent Relationship', 'Parallel Aware', 'Async Capable',
    'Startup Cost', 'Total Cost', 'Plan Rows', 'Plan Width', 'Join Type', 'Inner Unique',
    'Relation Name', 'Alias', 'Index Name', 'Scan Direction', 'Workers Planned',
    'Single Copy', 'Strategy', 'Partial Mode', 'Sort Key', 'Cache Key', 'Cache Mode', *PREDICATES}
CLASS_NAMES = ['severe_underestimation', 'slight_underestimation', 'approximately_accurate',
               'slight_overestimation', 'severe_overestimation']


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def estimated_only(node):
    return {**{k: node[k] for k in ESTIMATE_KEYS if k in node},
            'Plans': [estimated_only(c) for c in node.get('Plans', [])]}


def walk(plan, path='root'):
    yield path, plan
    for i, child in enumerate(plan.get('Plans', [])):
        yield from walk(child, path+'.'+str(i))


def is_target(node):
    return node.get('Node Type') in JOIN_TYPES or 'Scan' in node.get('Node Type', '')


def label(node):
    """Runtime fields appear ONLY in this supervision function, never extraction."""
    values = [node.get(k) for k in ['Plan Rows', 'Actual Rows', 'Actual Loops']]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        return None, 'invalid_or_missing_cardinality'
    estimated, actual, loops = values
    if loops == 0:
        return None, 'unexecuted'
    ratio = (estimated+1)/(actual+1)
    bucket = 0 if ratio <= .1 else 1 if ratio <= .5 else 2 if ratio <= 2 else 3 if ratio <= 10 else 4
    return {'class_id': bucket, 'class_name': CLASS_NAMES[bucket], 'rho': ratio,
            'log_rho': math.log(ratio), 'actual_rows': actual, 'actual_loops': loops,
            'zero_actual': actual == 0, 'zero_rounding_risk': actual == 0 and loops > 1}, None


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(node, key):
    value = node.get(key)
    return float(value) if finite(value) else 0.


def log(value):
    return math.copysign(math.log1p(min(abs(value), 1e100)), value)


def normalize_predicate(text):
    try:
        return sqlglot.parse_one(text, read='postgres').sql(dialect='postgres'), False
    except Exception:
        return ' '.join(text.split()), True


def sql_context(sql):
    try:
        tree = sqlglot.parse_one(sql, read='postgres')
        tables = {t.alias_or_name: t.name for t in tree.find_all(exp.Table)}
        return tree, tables, False
    except Exception:
        return None, {}, True


def predicate_features(texts, metadata, alias_tables):
    """Structured signals and lexical tokens; retain literals, never evaluate SQL."""
    out = {'predicate_count': float(len(texts)), 'parse_failure_count': 0.}
    columns, referenced_aliases, numeric, dates = [], set(), [], []
    normalized = []
    for text in texts:
        clean, failed = normalize_predicate(text)
        normalized.append(clean)
        if failed:
            out['parse_failure_count'] += 1
            continue
        tree = sqlglot.parse_one(text, read='postgres')
        for n in tree.walk():
            name = type(n).__name__
            if name in {'And', 'Or', 'EQ', 'NEQ', 'GT', 'GTE', 'LT', 'LTE', 'Like', 'ILike', 'In', 'Is', 'Not', 'Cast', 'Between'}:
                out['predicate_op:'+name] = out.get('predicate_op:'+name, 0.)+1
            if isinstance(n, exp.Column):
                alias = n.table
                if not alias:
                    owners = [a for a,m in metadata.items() if n.name in m.get('columns', [])]
                    alias = owners[0] if len(owners) == 1 else ''
                    out['unresolved_columns'] = out.get('unresolved_columns', 0.)+(not alias)
                relation = alias_tables.get(alias, metadata.get(alias, {}).get('relation', 'unknown'))
                columns.append((alias, n.name))
                referenced_aliases.add(alias)
                out['column:'+relation+'.'+n.name] = out.get('column:'+relation+'.'+n.name, 0.)+1
            if isinstance(n, exp.In):
                out['in_items'] = out.get('in_items', 0.)+len(n.expressions)
            if isinstance(n, exp.Literal):
                if n.is_string:
                    text_value = str(n.this)
                    out['string_length_log_sum'] = out.get('string_length_log_sum', 0.)+log(len(text_value))
                    out['percent_wildcards'] = out.get('percent_wildcards', 0.)+text_value.count('%')
                    out['underscore_wildcards'] = out.get('underscore_wildcards', 0.)+text_value.count('_')
                    out['leading_wildcards'] = out.get('leading_wildcards', 0.)+text_value.startswith(('%', '_'))
                    out['trailing_wildcards'] = out.get('trailing_wildcards', 0.)+text_value.endswith(('%', '_'))
                    try:
                        dates.append((datetime.fromisoformat(text_value.rstrip('Z'))-datetime(1970,1,1)).total_seconds()/86400)
                    except (ValueError, TypeError): pass
                else:
                    try:
                        value = float(n.this)
                        if isinstance(n.parent, exp.Neg): value = -value
                        if math.isfinite(value): numeric.append(value)
                    except (ValueError, TypeError): pass
    for name, values in [('numeric', numeric), ('date_days', dates)]:
        out[name+'_count'] = float(len(values))
        for agg in ['min', 'max', 'mean', 'range']:
            value = (min(values) if agg == 'min' else max(values) if agg == 'max' else
                     sum(v/len(values) for v in values) if agg == 'mean' else max(values)-min(values)) if values else 0.
            out[name+'_'+agg+'_log'] = log(value)
    out['predicate_column_count'] = float(len(columns))
    out['predicate_relation_count'] = float(len(referenced_aliases- {''}))
    return out, columns, normalized


def extract_features(plan, sql, metadata):
    """Only whitelisted estimated plan fields are visible below this boundary."""
    plan = estimated_only(plan)
    sql_tree, alias_tables, sql_failed = sql_context(sql)
    sql_terms = []
    def split_and(expression):
        if isinstance(expression, exp.And):
            return split_and(expression.left)+split_and(expression.right)
        return [expression]
    if sql_tree is not None:
        where = sql_tree.args.get('where')
        if where: sql_terms.extend(split_and(where.this))
        for join in sql_tree.find_all(exp.Join):
            if join.args.get('on'): sql_terms.extend(split_and(join.args['on']))
    def column_owner(column):
        if column.table: return column.table
        owners = [a for a,m in metadata.items() if column.name in m.get('columns', [])]
        return owners[0] if len(owners)==1 else ''
    sql_term_records = [(term.sql(dialect='postgres'), {column_owner(c) for c in term.find_all(exp.Column)}) for term in sql_terms]
    edges = set()
    for term in sql_terms:
        if isinstance(term, exp.EQ) and isinstance(term.left, exp.Column) and isinstance(term.right, exp.Column):
            a,b = column_owner(term.left),column_owner(term.right)
            if a and b and a!=b: edges.add(tuple(sorted([a,b])))
    index_aliases = {}
    for alias, m in metadata.items():
        for name in m.get('indexes', []): index_aliases.setdefault(name, []).append(alias)
    root_cost = number(plan, 'Total Cost')
    def identity(node, inherited=''):
        alias = node.get('Alias')
        if alias: return str(alias)
        relation = node.get('Relation Name')
        owners = [a for a,t in alias_tables.items() if t == relation] if relation else []
        if len(owners) == 1: return owners[0]
        owners = index_aliases.get(node.get('Index Name'), [])
        if inherited and inherited in owners: return inherited
        return owners[0] if len(owners) == 1 else ''
    def summarize(node, inherited=''):
        alias = identity(node, inherited)
        children = [summarize(c, alias or inherited) for c in node.get('Plans', [])]
        # Bitmap index children do not add a second base-table occurrence.
        aliases = Counter({alias: 1}) if alias and 'Scan' in node.get('Node Type', '') else sum((x['aliases'] for x in children), Counter())
        local = [str(node[k]) for k in PREDICATES if node.get(k)]
        return {'aliases': aliases, 'nodes': 1+sum(c['nodes'] for c in children),
                'height': 1+max([c['height'] for c in children], default=-1),
                'ops': Counter({node.get('Node Type', 'unknown'): 1})+sum((c['ops'] for c in children), Counter()),
                'predicates': local+[p for c in children for p in c['predicates']], 'children': children}
    summary = summarize(plan)
    result = []
    def visit(node, summary, path='root', ancestors=(), inherited=''):
        children = node.get('Plans', [])
        kind = node.get('Node Type', 'unknown')
        alias = identity(node, inherited)
        if is_target(node):
            paper = {'operator': kind, 'depth': float(len(ancestors))}
            for key in ['Startup Cost', 'Total Cost', 'Plan Rows', 'Plan Width']:
                paper[key+'_log'] = log(number(node, key))
                paper[key+'_missing'] = float(not finite(node.get(key)))
            child_cost = sum(number(c, 'Total Cost') for c in children)
            residual = number(node, 'Total Cost')-child_cost
            paper.update(self_cost_residual_log=log(residual), self_cost_negative=float(residual < 0),
                         estimated_selectivity_proxy_missing=1.)
            relations = Counter()
            for a,count in summary['aliases'].items():
                relation = alias_tables.get(a, metadata.get(a, {}).get('relation', 'unknown'))
                relations[relation] += count
                paper['table:'+relation] = 1.
            structure = {'parent_operator': ancestors[-1].get('Node Type', 'ROOT') if ancestors else 'ROOT',
                'parent_relationship': node.get('Parent Relationship', 'unknown'),
                'join_type': node.get('Join Type', 'none'), 'inner_unique': float(bool(node.get('Inner Unique'))),
                'parallel_aware': float(bool(node.get('Parallel Aware'))),
                'under_gather': float(any(a.get('Node Type') in ['Gather', 'Gather Merge'] for a in ancestors)),
                'under_memoize': float(any(a.get('Node Type') == 'Memoize' for a in ancestors)),
                'workers_planned': max([number(a, 'Workers Planned') for a in (*ancestors, node)], default=0.),
                'height': float(summary['height']), 'subtree_nodes': float(summary['nodes']),
                'children': float(len(children)), 'covered_aliases': float(len(summary['aliases'])),
                'covered_relations': float(len(relations)), 'base_relation_occurrences': float(sum(relations.values())),
                'startup_cost_fraction': number(node, 'Startup Cost')/max(number(node, 'Total Cost'), 1.),
                'root_cost_fraction': number(node, 'Total Cost')/max(root_cost, 1.),
                'estimated_cost_per_row_log': log(number(node, 'Total Cost')/max(number(node, 'Plan Rows'), 1.))}
            for op,count in summary['ops'].items(): structure['subtree_operator:'+op] = float(count)
            for relation,count in relations.items(): structure['relation_occurrences:'+relation] = float(count)
            for side,child,sub in zip(['left', 'right'], children, summary['children']):
                structure[side+'_operator'] = child.get('Node Type', 'unknown')
                for key in ['Plan Rows', 'Total Cost', 'Plan Width']:
                    structure[side+'_'+key+'_log'] = log(number(child, key))
                for a in sub['aliases']:
                    structure[side+'_table:'+alias_tables.get(a, metadata.get(a, {}).get('relation', 'unknown'))] = 1.
            if len(children) == 2 and kind in JOIN_TYPES:
                left, right = [number(c, 'Plan Rows') for c in children]
                proxy = number(node, 'Plan Rows')/max(left*right, 1.)
                paper.update(estimated_selectivity_proxy_log=log(proxy), estimated_selectivity_proxy_missing=0.)
                structure.update(left_right_row_ratio_log=log(left/max(right, 1.)),
                                 output_left_row_ratio_log=log(number(node, 'Plan Rows')/max(left, 1.)),
                                 output_right_row_ratio_log=log(number(node, 'Plan Rows')/max(right, 1.)))
            local = [str(node[k]) for k in PREDICATES if node.get(k)]
            local_features, _, normalized_local = predicate_features(local, metadata, alias_tables)
            covered = set(summary['aliases'])
            applicable_sql = [text for text,owners in sql_term_records if owners and '' not in owners and owners<=covered]
            sql_features, columns, normalized_all = predicate_features(
                list(dict.fromkeys(summary['predicates']+applicable_sql)), metadata, alias_tables)
            degree = [sum(a in edge for edge in edges) for a in covered]
            sql_features.update(sql_applicable_predicates=float(len(applicable_sql)),
                query_join_graph_edges=float(len(edges)),
                subtree_join_graph_edges=float(sum(set(edge)<=covered for edge in edges)),
                subtree_boundary_edges=float(sum(len(set(edge)&covered)==1 for edge in edges)),
                join_graph_degree_min=float(min(degree,default=0)),
                join_graph_degree_max=float(max(degree,default=0)),
                join_graph_degree_mean=sum(degree)/max(len(degree),1))
            for key,value in local_features.items(): sql_features['local_'+key] = value
            m = metadata.get(alias, {})
            definitions = m.get('index_definitions', [])
            selected = next((d['definition'] for d in definitions if d['name'] == node.get('Index Name')), '')
            indexed_columns = []
            index_parse_failed = False
            if selected:
                try:
                    create = sqlglot.parse_one(selected, read='postgres')
                    indexed_columns = [c.name for c in create.find_all(exp.Column)]
                except Exception: index_parse_failed = True
            relation = alias_tables.get(alias, m.get('relation', 'unknown'))
            sql_features.update(scan_relation=relation, scan_direction=node.get('Scan Direction', 'none'),
                selected_index=node.get('Index Name', 'none'), available_indexes=float(len(m.get('indexes', []))),
                index_definition_missing=float(bool(node.get('Index Name')) and not selected),
                unknown_scan_owner=float('Scan' in kind and not alias),
                selected_index_unique=float('CREATE UNIQUE INDEX' in selected.upper()),
                selected_index_partial=float(bool(re.search(r'\bWHERE\b', selected, re.I))),
                index_parse_failed=float(index_parse_failed), selected_index_columns=float(len(set(indexed_columns))),
                predicate_index_column_matches=float(sum(a == alias and c in indexed_columns for a,c in columns)),
                available_unique_indexes=float(sum('CREATE UNIQUE INDEX' in d['definition'].upper() for d in definitions)),
                sql_parse_failed=float(sql_failed), database_column_statistics_available=0.)
            method = re.search(r'\bUSING\s+(\w+)', selected, re.I)
            sql_features['index_method'] = method[1].lower() if method else 'unknown'
            result.append({'node_path': path, 'node_type': kind, 'node_kind': 'join' if kind in JOIN_TYPES else 'scan',
                'groups': {'paper': paper, 'structure': structure, 'sql_index': sql_features},
                'local_predicate_text': ' ; '.join(normalized_local),
                'predicate_text': ' ; '.join(dict.fromkeys(normalized_all)),
                'context': {'alias': alias or None, 'relation': relation, 'covered_aliases': dict(summary['aliases']),
                            'selected_index_definition': selected or None}})
        for i,(child,sub) in enumerate(zip(children, summary['children'])):
            visit(child, sub, path+'.'+str(i), (*ancestors, node), alias or inherited)
    visit(plan, summary)
    return result
