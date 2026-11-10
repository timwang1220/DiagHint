"""Stage I experiences, Stage II measurements and one auditable training dataset."""
import json
import math
from pathlib import Path
import random
import time
from .config import public_snapshot,protocol_id
from .database import Database
from .features import label,walk
from .hints import actions,comment,merge,candidates,connected,graph,relation_set
from .llm import LLM
from .plans import query_name,sql_id,root,oracle_signals,accurate,annotate,static,physical
from .prompts import Prompts
from .storage import artifact_dir,save,read,digest,locked,checkpoint


def parse_response(answer):
    if not answer.get('text'):raise ValueError('No response')
    parsed=json.loads(answer['text']);hint=comment(parsed['final_hint'])
    reasons=parsed.get('reasons',parsed.get('hint_reasons',[]))
    if not isinstance(reasons,(str,list,dict)):raise ValueError('Invalid reasons')
    return hint,reasons


def context(target,signals,source=None,current=None,existing_hint=''):
    # One target schema for collection and online; only the signal provider differs.
    return dict(query_id=target['query'],sql=target['sql'],plan=annotate(target['plan'],signals),
                indexes=target['metadata'],source=source,current_plan=current or target['plan'],existing_hint=existing_hint)


def execute_candidate(db,llm,phase,target,signals,folder,source=None,conversation=(),existing_hint='',current=None,timeout_ms=300000):
    folder=Path(folder)
    if (folder/'record.json').exists():return read(folder/'record.json')
    request=llm.generate(phase,context(target,signals,source,current,existing_hint),conversation)
    record=dict(phase=phase,query=target['query'],source_id=source['id'] if source else None,
                request_id=request['request_id'],api_ms=request['api_ms'],cached_response=request['cached'],messages=request['messages'])
    try:
        hint,reasons=parse_response(request)
        record.update(hint=hint,reasons=reasons,status='generated',response=request['text'])
    except (ValueError,KeyError,TypeError):
        record.update(hint='',reasons=[],status='generation_failure',response=request.get('text'))
    save(folder/'record.json',record)
    return record


def eligible(experiences,target):
    target_id=sql_id(target['sql'])
    return [x for x in experiences if x['query']!=target['query'] and x['sql_id']!=target_id and x['hint'] and x['reasons']]


def make_experience(target,native,best,stage):
    result=best['result']
    if not (result['status']=='ok' and best['hint'] and best['reasons'] and result['execution_ms']<native['execution_ms'] and physical(result['document'])!=physical(native['document'])):return None
    return dict(id=digest(dict(query=target['query'],hint=best['hint'],plan=static(result['document']))),
        query=target['query'],stage=stage,sql=target['sql'],sql_id=sql_id(target['sql']),metadata=target['metadata'],
        native_plan=root(native['document']),best_plan=root(result['document']),optimized_plan=root(result['document']),
        native_ms=native['execution_ms'],best_ms=result['execution_ms'],hint=best['hint'],reasons=best['reasons'])


def stage_one(db,llm,target,native,folder):
    best=dict(hint='',reasons=[],result=native);signals=oracle_signals(native['document']);records=[]
    if accurate(target['plan'],signals):return records,best
    global_hint='';leading_hint='';used_prefix=set();used_nodes=set()
    for stage,count in [('global',1),('leading',3),('node',3)]:
        dialogue=[];stage_candidates=[]
        # Leading alternatives all start at the post-global plan; Node follows best.
        start=best
        for i in range(count):
            sub=Path(folder)/f'{stage}_{i+1}'
            record=execute_candidate(db,llm,stage,target,signals,sub,conversation=dialogue,
                existing_hint=best['hint'] if stage=='node' else start['hint'],current=root((best if stage=='node' else start)['result']['document']))
            dialogue=[*record['messages'],dict(role='assistant',content=record.get('response') or '')]
            records.append(record)
            if record['status'] in {'generation_failure','invalid_hint'}:continue
            try:
                proposed=record.get('proposed_hint',record['hint'])
                record['proposed_hint']=proposed
                aa=actions(proposed)
                if len(aa)!=1:raise ValueError('Each Stage-I round proposes exactly one action')
                method,args=aa[0]
                if stage=='global':
                    if method!='Set' or len(args)!=2 or args[1]!='off':raise ValueError('Invalid global candidate')
                    # Full compatibility validation applies even to Stage-I globals.
                    from .hints import mapped_hint
                    mapped_hint(dict(hint=proposed),target)
                elif stage=='leading':
                    _,edges=graph(target['sql']);key=tuple(args)
                    if method!='Leading' or not connected(args,edges) or key in used_prefix:raise ValueError('Invalid or repeated prefix')
                    used_prefix.add(key)
                else:
                    if method in {'Set','Leading'}:raise ValueError('Expected one local repair')
                    key=('join',tuple(sorted(args))) if method in {'HashJoin','NestLoop','MergeJoin'} else ('scan',args[0])
                    if key in used_nodes:raise ValueError('Repeated repair node')
                    current=root(best['result']['document'])
                    if key[0]=='join':
                        matches=[n for _,n in walk(current) if n.get('Node Type') in {'Hash Join','Merge Join','Nested Loop'} and relation_set(n)==set(args)]
                        if len(matches)!=1:raise ValueError('Repair must identify one existing join node')
                    else:
                        if args[0] not in target['metadata']:raise ValueError('Unknown scan alias')
                        if any(index not in target['metadata'][args[0]]['indexes'] for index in args[1:]):raise ValueError('Unknown scan index')
                    used_nodes.add(key)
                combined=merge(best['hint'] if stage=='node' else start['hint'],proposed)
                result=db.measure(target['sql'],combined,sub/'execution')
                record.update(hint=combined,result=result,status=result['status'])
                if result['status']=='ok':
                    candidate=dict(hint=combined,reasons=[*(best if stage=='node' else start)['reasons'],record['reasons']],result=result)
                    stage_candidates.append(candidate)
                    if stage=='node' and result['execution_ms']<best['result']['execution_ms']:best=candidate
            except (ValueError,KeyError) as error:record.update(status='invalid_hint',reason=str(error))
            save(sub/'record.json',record)
        if stage!='node' and stage_candidates:
            winner=min(stage_candidates,key=lambda x:x['result']['execution_ms'])
            if winner['result']['execution_ms']<best['result']['execution_ms']:best=winner
    return records,best


def measured_record(db,record,target,folder,timeout):
    if record['status']=='generated':
        result=db.measure(target['sql'],record['hint'],Path(folder)/'execution',timeout)
        record.update(result=result,status=result['status'])
    save(Path(folder)/'record.json',record)
    return record


def generate(config,workload,output):
    output=artifact_dir(output);paths=sorted(Path(workload).glob('*.sql'),key=query_name)
    if not paths:raise ValueError('No training SQL files')
    queries={p.stem:p.read_text() for p in paths}
    fingerprint=digest(dict(queries=queries,config=public_snapshot(config)))
    if (output/'manifest.json').exists() and read(output/'manifest.json')['fingerprint']!=fingerprint:raise ValueError('Input workload/config changed; use a new output directory')
    prompts=Prompts()
    if (output/'manifest.json').exists() and read(output/'manifest.json')['prompt_id']!=prompts.fingerprint:raise ValueError('Private prompt version changed')
    with locked(output/'run.lock'):
        save(output/'manifest.json',dict(fingerprint=fingerprint,queries=queries,query_sql_ids={q:sql_id(s) for q,s in queries.items()},
            config=public_snapshot(config),protocol=protocol_id(config),prompt_id=prompts.fingerprint,label_boundaries=[.1,.5,2,10],
            stage2_rounds=dict(nodemo=1,demo=6,reuse_maximum=5),training_queries_only=True))
        db=Database(config,output);llm=LLM(config,output,prompts);seeds={}
        for p in paths:seeds.setdefault(query_name(p)[0],p.stem)
        experiences=[]
        try:
            for q in [*seeds.values(),*[q for q in queries if q not in seeds.values()]]:
                folder=output/'queries'/q
                if (folder/'complete.json').exists():
                    experience=read(folder/'experience.json') if (folder/'experience.json').exists() else None
                    if experience:experiences.append(experience)
                    continue
                checkpoint(output,'generating_pool',query=q)
                sql=queries[q];metadata=db.metadata(sql);probe=db.explain(sql)
                if probe['status']!='ok':raise RuntimeError('Native EXPLAIN failed: '+q)
                target=dict(query=q,sql=sql,metadata=metadata,plan=root(probe['document']))
                save(folder/'input.json',target)
                native=db.measure(sql,'',folder/'native',1800000)
                save(folder/'native.json',native)
                if native['status']!='ok':
                    save(folder/'complete.json',dict(status=native['status']));continue
                best=dict(hint='',reasons=[],result=native);records=[];signals=oracle_signals(native['document'])
                if q in seeds.values():records,best=stage_one(db,llm,target,native,folder)
                elif not accurate(target['plan'],signals):
                    sources=eligible([x for x in experiences if x['stage']==1],target)
                    save(folder/'frozen_sources.json',[x['id'] for x in sources])
                    timeout=min(300000,max(1000,int(2*native['wall_ms'])))
                    nodemo=execute_candidate(db,llm,'optimize',target,signals,folder/'nodemo')
                    records.append(measured_record(db,nodemo,target,folder/'nodemo',timeout))
                    rng=random.Random(int(digest(dict(seed=config['pipeline']['seed'],query=q))[:16],16))
                    # Sample SQL identities first: multiple experiences cannot multiply a source's chance.
                    groups={}
                    for source in sources:groups.setdefault(source['sql_id'],[]).append(source)
                    chosen=rng.sample(sorted(groups),min(6,len(groups)))
                    for i,group in enumerate(chosen):
                        source=rng.choice(groups[group]);sub=folder/f'demo_{i+1}'
                        record=execute_candidate(db,llm,'optimize',target,signals,sub,source=source)
                        records.append(measured_record(db,record,target,sub,timeout))
                    reusable,rejected=candidates(sources,target);save(folder/'reuse_rejections.json',rejected)
                    rng.shuffle(reusable)
                    for i,candidate in enumerate(reusable[:5]):
                        sub=folder/f'reuse_{i+1}';result=db.measure(sql,candidate['hint'],sub/'execution',timeout)
                        record=dict(query=q,phase='reuse',hint=candidate['hint'],source_ids=candidate['source_ids'],
                                    result=result,status=result['status'],reasons=[],api_ms=0)
                        save(sub/'record.json',record);records.append(record)
                    for record in records:
                        r=record.get('result',{})
                        if r.get('status')=='ok' and r['execution_ms']<best['result']['execution_ms']:
                            best=dict(hint=record['hint'],reasons=record['reasons'],result=r)
                save(folder/'records.json',records)
                experience=make_experience(target,native,best,1 if q in seeds.values() else 2)
                if experience:save(folder/'experience.json',experience);experiences.append(experience)
                save(folder/'complete.json',dict(status='ok',accurate_skip=accurate(target['plan'],signals)))
                save(output/'experiences.json',experiences)
            save(output/'experiences.json',experiences)
            save(output/'COMPLETE.json',dict(queries=len(queries),experiences=len(experiences),time=time.time()))
            checkpoint(output,'complete')
        finally:db.close()
