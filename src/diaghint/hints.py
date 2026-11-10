"""Complete-set compatibility mapping; never salvage a hint subset."""
from collections import Counter
import re
from sqlglot import exp
from .plans import query,aliases,root
from .features import walk

GLOBALS={'enable_'+name for name in ['nestloop','hashjoin','mergejoin','seqscan','indexscan','indexonlyscan']}
METHODS={'Set','Leading','HashJoin','NestLoop','MergeJoin','SeqScan','IndexScan','IndexOnlyScan','BitmapScan'}


def comment(hint):
    if hint=='':return ''
    if not isinstance(hint,str):raise ValueError('Hint must be a string')
    text=hint.strip()
    if not text.startswith('/*+') or not text.endswith('*/') or '/*' in text[3:-2] or '*/' in text[3:-2]:raise ValueError('Expected one optimizer comment')
    return text


def actions(hint):
    text=comment(hint)
    if not text:return []
    body=text[3:-2].strip();result=[]
    while body:
        match=re.match(r'(\w+)\s*\(',body)
        if not match or match[1] not in METHODS:raise ValueError('Unsupported hint action')
        depth=1;i=match.end()
        while i<len(body) and depth:
            depth+=(body[i]=='(')-(body[i]==')');i+=1
        if depth:raise ValueError('Unbalanced hint action')
        args=body[match.end():i-1].strip()
        if '(' in args or ')' in args:raise ValueError('Only flat partial Leading is supported')
        tokens=args.split()
        if not tokens or any(not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9$]*',x) for x in tokens):raise ValueError('Unsupported hint identifier')
        result.append((match[1],tokens));body=body[i:].strip()
    return result


def merge(*hints):
    result=[]
    for hint in hints:
        for method,args in actions(hint):
            action=method+'('+' '.join(args)+')'
            if action not in result:result.append(action)
    return '/*+ '+' '.join(result)+' */' if result else ''


def relation_set(node):
    return frozenset(n['Alias'] for _,n in walk(node) if n.get('Alias') and 'Scan' in n.get('Node Type',''))


def graph(sql):
    parsed=query(sql);tables=aliases(sql)
    for join in parsed.find_all(exp.Join):
        if join.args.get('side') or str(join.args.get('kind','')).upper() not in {'','INNER','CROSS'} or join.args.get('using'):
            raise ValueError('Only ordinary inner-join predicate mapping is supported')
    conditions=[]
    def conjuncts(e):
        if isinstance(e,exp.And):return conjuncts(e.left)+conjuncts(e.right)
        return [e]
    where=parsed.args.get('where')
    if where:conditions+=conjuncts(where.this)
    for join in parsed.find_all(exp.Join):
        if join.args.get('on'):conditions+=conjuncts(join.args['on'])
    edges=set()
    for condition in conditions:
        if isinstance(condition,(exp.EQ,exp.LT,exp.LTE,exp.GT,exp.GTE,exp.NEQ)) and isinstance(condition.left,exp.Column) and isinstance(condition.right,exp.Column):
            a,b=condition.left,condition.right
            if not a.table or not b.table or a.table==b.table:continue
            op=type(condition).__name__;left=(a.table,a.name);right=(b.table,b.name)
            if op=='EQ' and right<left:left,right=right,left
            edges.add((left,right,op))
    return tables,edges


def connected(prefix,edges):
    if not 1<=len(prefix)<=3 or len(set(prefix))!=len(prefix):return False
    seen={prefix[0]}
    for alias in prefix[1:]:
        if not any((a[0]==alias and b[0] in seen) or (b[0]==alias and a[0] in seen) for a,b,_ in edges):return False
        seen.add(alias)
    return True


def alias_mapping(source_sql,target_sql):
    source=aliases(source_sql);target=aliases(target_sql);sc=Counter(source.values());tc=Counter(target.values());mapped={}
    for name,table in source.items():
        choices=[a for a,t in target.items() if t==table]
        if sc[table]==tc[table]==1:mapped[name]=choices[0]
        elif name in choices:
            # Self-join roles require matching incident column-labelled edges.
            _,se=graph(source_sql);_,te=graph(target_sql)
            def role(alias,edges,tables):
                return {(own[1],tables[other[0]],other[1],op) for a,b,op in edges for own,other in [(a,b),(b,a)] if own[0]==alias}
            if role(name,se,source)==role(name,te,target) and role(name,se,source):mapped[name]=name
    return mapped


def mapped_hint(source,target):
    """Return the whole mapped hint or raise; source provenance never scores reuse."""
    aa=actions(source['hint'])
    if not aa:raise ValueError('Empty hints are not reuse candidates')
    # Globals do not require source/target structural similarity.
    nonglobal=any(m!='Set' for m,_ in aa)
    mapping=alias_mapping(source['sql'],target['sql']) if nonglobal else {}
    emitted=[]
    for method,args in aa:
        if method=='Set':
            if len(args)!=2 or args[0] not in GLOBALS or args[1].lower()!='off':raise ValueError('Unsupported global restriction')
            emitted.append('Set('+' '.join(args)+')');continue
        if method=='Leading':
            st,se=graph(source['sql']);tt,te=graph(target['sql'])
            if not connected(args,se):raise ValueError('Disconnected or overlong source prefix')
            out=[mapping[a] for a in args]
            if not connected(out,te):raise ValueError('Disconnected target prefix')
            def internal(edges,selected,remap):
                result=set()
                for a,b,op in edges:
                    if a[0] not in selected or b[0] not in selected:continue
                    a=(remap[a[0]],a[1]);b=(remap[b[0]],b[1])
                    if op=='EQ' and b<a:a,b=b,a
                    result.add((a,b,op))
                return result
            if internal(se,set(args),mapping)!=internal(te,set(out),{a:a for a in tt}):raise ValueError('Leading predicate graph differs')
        elif method in {'HashJoin','NestLoop','MergeJoin'}:
            out=[mapping[a] for a in args]
            sources=[n for _,n in walk(root(source['native_plan'])) if n.get('Node Type') in {'Hash Join','Merge Join','Nested Loop'} and relation_set(n)==set(args)]
            targets=[n for _,n in walk(root(target['plan'])) if n.get('Node Type') in {'Hash Join','Merge Join','Nested Loop'} and relation_set(n)==set(out)]
            if len(sources)!=1 or len(targets)!=1:raise ValueError('No unique matching join subtree')
            a,b=sources[0],targets[0]
            if a.get('Join Type','Inner')!='Inner' or b.get('Join Type','Inner')!='Inner' or len(a.get('Plans',[]))!=2 or len(b.get('Plans',[]))!=2:raise ValueError('Unsupported join type')
            for left,right in zip(a['Plans'],b['Plans']):
                if {mapping.get(x) for x in relation_set(left)}!=relation_set(right):raise ValueError('Join child roles differ')
        else:
            alias=mapping[args[0]];out=[alias]
            if method=='SeqScan' and len(args)!=1:raise ValueError('SeqScan takes one relation')
            sm=source['metadata'][args[0]];tm=target['metadata'][alias]
            if method!='SeqScan':
                def predicate_columns(sql,alias):
                    parsed=query(sql)
                    return {(c.name,type(c.parent).__name__) for c in parsed.find_all(exp.Column) if c.table==alias and not isinstance(c.parent,exp.Select)}
                if not predicate_columns(source['sql'],args[0]) & predicate_columns(target['sql'],alias):raise ValueError('No matching scan predicate object')
            for index in args[1:]:
                original=next((i for i in sm['index_definitions'] if i['name']==index),None)
                if not original:raise ValueError('Unknown source index')
                matches=[i for i in tm['index_definitions'] if i['columns']==original['columns'] and not i.get('predicate') and not i.get('expression')]
                if not matches:raise ValueError('No provably applicable target index')
                out.append(matches[0]['name'])
            if method in {'IndexScan','IndexOnlyScan','BitmapScan'} and not tm['index_definitions']:raise ValueError('No target index')
            if method=='IndexOnlyScan':
                required=set()
                for c in query(target['sql']).find_all(exp.Column):
                    if c.table==alias:required.add(c.name)
                    elif not c.table:
                        owners=[a for a,m in target['metadata'].items() if c.name in m['columns']]
                        if len(owners)!=1:raise ValueError('Cannot resolve index coverage column')
                        if owners[0]==alias:required.add(c.name)
                if any(isinstance(n,exp.Star) for n in query(target['sql']).walk()):raise ValueError('Cannot prove star coverage')
                if not any(required<=set(i['columns']) and not i.get('predicate') and not i.get('expression') and (len(out)==1 or i['name'] in out[1:]) for i in tm['index_definitions']):raise ValueError('Index does not cover target columns')
        emitted.append(method+'('+' '.join(out)+')')
    return '/*+ '+' '.join(emitted)+' */'


def candidates(sources,target):
    found={};rejected=[]
    for source in sources:
        try:
            hint=mapped_hint(source,target)
            found.setdefault(hint,dict(hint=hint,source_ids=[]))['source_ids'].append(source['id'])
        except (ValueError,KeyError,IndexError) as error:rejected.append(dict(source_id=source['id'],reason=str(error)))
    return list(found.values()),rejected
