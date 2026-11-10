"""Estimated-only complete trees, SQL identity, diagnostics and codec."""
from collections import Counter
import copy
import hashlib
import math
import re
import numpy as np
import sqlglot
from sqlglot import exp
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from .features import estimated_only,extract_features,is_target,walk,label,CLASS_NAMES,number,log
from .storage import digest

BOUNDS=(.1,.5,2.,10.)


def root(document):
    if isinstance(document,list):return root(document[0])
    return document.get('Plan',document)


def query(sql):
    trees=[x for x in sqlglot.parse(sql,read='postgres') if x is not None and not isinstance(x,exp.Semicolon)]
    if len(trees)!=1 or not isinstance(trees[0],exp.Select):raise ValueError('Exactly one read-only SELECT is required')
    tree=trees[0]
    if tree.args.get('into') or tree.args.get('locks'):raise ValueError('SELECT INTO/locking is not supported')
    forbidden={'INSERT','UPDATE','DELETE','MERGE','CREATE','DROP','ALTER','COPY','COMMAND','TRANSACTION'}
    if any(type(n).__name__.upper() in forbidden for n in tree.walk()):raise ValueError('Modifying SQL is not allowed')
    for f in tree.find_all(exp.Func):
        name=f.name.lower() if isinstance(f,exp.Anonymous) else f.sql_name().lower()
        if name.startswith(('pg_terminate','pg_cancel','lo_','dblink')) or name in {'set_config','nextval','setval','pg_advisory_lock'}:
            raise ValueError('Side-effecting functions are not supported')
    return tree


def sql_id(sql):
    return hashlib.sha256(query(sql).sql(dialect='postgres',normalize=True,comments=False).encode()).hexdigest()


def query_name(path):
    match=re.fullmatch(r'(\d+)([a-z]+)',path.stem)
    if not match:raise ValueError('SQL filenames must be template number + letters, e.g. 13e.sql')
    return int(match[1]),match[2]


def aliases(sql):
    parsed=query(sql)
    if any(isinstance(n,(exp.Subquery,exp.CTE,exp.Lateral)) for n in parsed.walk()):raise ValueError('Subquery/CTE/LATERAL mapping is unsupported')
    result={}
    for table in parsed.find_all(exp.Table):
        name=table.alias_or_name
        if name in result:raise ValueError('Ambiguous alias')
        bare=table.copy();bare.set('alias',None);result[name]=bare.sql(dialect='postgres')
    return result


def tree(plan,sql,metadata):
    plan=estimated_only(root(plan))
    detailed={r['node_path']:r for r in extract_features(plan,sql,metadata)}
    nodes=[];children=[];paths=[];supported=[]
    def visit(n,path,depth,parent):
        if len(n.get('Plans',[]))>2:raise ValueError('Ordered Binary Tree-LSTM rejects arity >2')
        index=len(nodes);nodes.append(None);children.append([]);paths.append(path);supported.append(is_target(n))
        cs=[visit(c,path+'.'+str(i),depth+1,n.get('Node Type','')) for i,c in enumerate(n.get('Plans',[]))]
        children[index]=cs
        data=detailed.get(path)
        features={} if data is None else {group+':'+k:v for group,values in data['groups'].items() for k,v in values.items()}
        features.update(operator=n.get('Node Type','unknown'),parent=parent,depth=depth,arity=len(cs),subtree_size=sum(1 for _ in walk(n)))
        for k in ['Plan Rows','Plan Width','Startup Cost','Total Cost','Workers Planned']:
            features['estimate:'+k]=log(number(n,k));features['missing:'+k]=float(k not in n)
        for k in ['Join Type','Parent Relationship','Strategy','Partial Mode','Sort Key','Cache Key']:
            features[k]=str(n.get(k,'missing'))
        for k in ['Parallel Aware','Inner Unique','Async Capable']:features[k]=float(bool(n.get(k,False)))
        for name,count in Counter(c.get('Relation Name') for _,c in walk(n) if c.get('Relation Name')).items():features['table:'+name]=count
        for i,c in enumerate(n.get('Plans',[])):
            features['child_type:'+str(i)]=c.get('Node Type','unknown');features['child_rows:'+str(i)]=log(number(c,'Plan Rows'))
        nodes[index]=features
        return index
    visit(plan,'root',0,'')
    return dict(plan=plan,nodes=nodes,children=children,paths=paths,supported=supported)


class Codec:
    def __init__(self):self.vectorizer=DictVectorizer(sparse=False);self.scaler=StandardScaler()
    def fit(self,trees):
        rows=[r for t in trees for r in t['nodes']]
        if not rows:raise ValueError('No training feature rows')
        self.scaler.fit(self.vectorizer.fit_transform(rows));return self
    def transform(self,t):
        return self.scaler.transform(self.vectorizer.transform(t['nodes'])).astype(np.float32)
    @property
    def dim(self):return len(self.vectorizer.feature_names_)


def oracle_signals(analyzed):
    result={}
    for path,n in walk(root(analyzed)):
        if not is_target(n):continue
        y,_=label(n)
        if y:result[path]=dict(probabilities=[float(i==y['class_id']) for i in range(5)],class_id=y['class_id'],class_name=y['class_name'],available=True)
    return result


def annotate(plan,signals):
    result=estimated_only(root(plan))
    for path,n in walk(result):
        if is_target(n):n['Cardinality Estimation Bias']=signals.get(path,dict(available=False))
    return result


def accurate(plan,signals):
    relevant=[path for path,n in walk(root(plan)) if is_target(n)]
    return bool(relevant) and all(signals.get(p,{}).get('available') and signals[p]['class_id']==2 for p in relevant)


def static(document):
    runtime={'Workers','Workers Launched','Execution Time','Planning Time','Planning','Triggers','JIT',
             'Sort Method','Sort Space Used','Sort Space Type','Heap Fetches','Hash Batches','Original Hash Batches',
             'Hash Buckets','Original Hash Buckets','Peak Memory Usage','Cache Hits','Cache Misses',
             'Cache Evictions','Cache Overflows','Exact Heap Blocks','Lossy Heap Blocks',
             'Full-sort Groups','Pre-sorted Groups','HashAgg Batches','Disk Usage','Subplans Removed'}
    def clean(value):
        if isinstance(value,dict):
            return {k:clean(v) for k,v in value.items() if k not in runtime and not k.startswith(('Actual ','Rows Removed','Shared ','Local ','Temp ','I/O ','WAL '))}
        if isinstance(value,list):return [clean(v) for v in value]
        return value
    return dict(Plan=clean(root(document)),Settings=document.get('Settings',{}))


def hint_bits(hint):
    return [float(bool(re.search(x,hint,re.I))) for x in [r'\bSet\(',r'\bLeading\(',r'\b(?:HashJoin|MergeJoin|NestLoop)\(',r'\b(?:SeqScan|IndexScan|IndexOnlyScan|BitmapScan)\(']]


def physical(document):
    """Operator/access-path identity, distinct from a full measurement cache key."""
    def clean(node):
        return {k:([clean(c) for c in v] if k=='Plans' else v) for k,v in node.items()
                if k not in {'Startup Cost','Total Cost','Plan Rows','Plan Width'}}
    return clean(estimated_only(root(document)))
