"""The single online route: accurate PG -> reuse1.20 -> positive Demo -> PG."""
import time
from pathlib import Path
from .database import Database
from .hints import candidates
from .learning import Cardinality,Utility
from .llm import LLM
from .plans import tree,root,accurate,static,sql_id
from .pool import context,parse_response,eligible
from .prompts import Prompts
from .storage import artifact_dir,read,save,digest,locked,checkpoint


def optimize(config,sql_path,pool,cardinality,utility,output):
    output=artifact_dir(output);pool=Path(pool)
    with locked(output/'run.lock'):
        if (output/'result.json').exists():raise ValueError('This online run already has a result; use a new output directory')
        began=time.perf_counter();sql=Path(sql_path).read_text()
        card=Cardinality(cardinality);model=Utility(utility,card)
        if model.manifest['pool_manifest']!=digest(read(pool/'manifest.json')):raise ValueError('Utility/pool provenance mismatch')
        prompts=Prompts()
        if prompts.fingerprint!=read(pool/'manifest.json')['prompt_id']:raise ValueError('Collection/online prompt versions differ')
        db=Database(config,output)
        try:
            if db.scope!=model.manifest['pool_scope']:raise ValueError('Model/pool database scope differs from the online database')
            probe=db.explain(sql)
            if probe['status']!='ok':raise RuntimeError('Native EXPLAIN failed')
            target=dict(query=Path(sql_path).stem,sql=sql,plan=root(probe['document']),metadata=db.metadata(sql))
            t=tree(target['plan'],sql,target['metadata']);signals=card.signals(t)
            save(output/'input.json',target);save(output/'bias.json',signals)
            route='PG';hint='';generation=None;source_id=None;reuse_scores=[];demo_scores=[];rejections=[]
            skip=accurate(target['plan'],signals);reason='all_accurate' if skip else 'no_positive_demo'
            if not skip:
                a=model.encode(target['plan'],sql,target['metadata'])
                # Offline value retained for audit; NEVER opens an online NoDemo branch.
                save(output/'nodemo_value_audit.json',model.score('nodemo',[a]))
                sources=eligible(read(pool/'experiences.json'),target)
                mapped,rejections=candidates(sources,target);seen={digest(static(probe['document']))}
                for candidate in mapped:
                    trial=db.explain(sql,candidate['hint'])
                    if trial['status']!='ok':
                        rejections.append(dict(hint=candidate['hint'],reason=trial['status']));continue
                    signature=digest(static(trial['document']))
                    if signature in seen:continue
                    seen.add(signature)
                    h=model.encode(trial['document'],sql,target['metadata'])
                    reuse_scores.append(dict(**candidate,**model.score('reuse',[a,h],candidate['hint'])))
                best=max(reuse_scores,key=lambda x:(x['log_utility'],x['hint'])) if reuse_scores else None
                if best and best['predicted_speedup']>=1.20:
                    route='Reuse';hint=best['hint'];reason='reuse_threshold_met'
                else:
                    for source in sources:
                        b=model.encode(source['native_plan'],source['sql'],source['metadata'])
                        c=model.encode(source['best_plan'],source['sql'],source['metadata'])
                        demo_scores.append(dict(source_id=source['id'],**model.score('demo',[a,b,c],source['hint'])))
                    chosen=max(demo_scores,key=lambda x:(x['log_utility'],x['source_id'])) if demo_scores else None
                    if chosen and chosen['log_utility']>0:
                        source_id=chosen['source_id'];source=next(x for x in sources if x['id']==source_id)
                        generation=LLM(config,output,prompts,cache_roots=[pool]).generate('optimize',context(target,signals,source))
                        try:
                            hint,reasons=parse_response(generation);route='Demo' if hint.replace('/*+','').replace('*/','').strip() else 'PG'
                            reason='positive_demo' if route=='Demo' else 'empty_llm_hint'
                            # A valid nonempty response is inspected, not benchmarked, before selection.
                            candidate=db.explain(sql,hint)
                            if candidate['status']!='ok':hint='';route='PG';reason='invalid_generated_plan'
                            save(output/'candidate.explain.json',candidate)
                        except (ValueError,KeyError,TypeError):hint='';route='PG';reason='llm_failure'
            save(output/'routing.json',dict(route=route,reason=reason,all_accurate=skip,source_id=source_id,hint=hint,
                 reuse_scores=reuse_scores,demo_scores=demo_scores,rejections=rejections,nodemo_online=False))
            # Import only same-scope measured plans. Cache reads do not influence routing.
            cache=pool/'plan_cache'
            for p in cache.glob('*.json'):
                destination=output/'plan_cache'/p.name
                if not destination.exists():save(destination,read(p))
            result=db.measure(sql,hint,output/'selected_execution',300000)
            elapsed=1000*(time.perf_counter()-began)
            reconstructed=elapsed+(result.get('wall_ms',0) if result.get('cached') else 0)
            if generation and generation['cached']:reconstructed+=generation['api_ms']
            record=dict(query=target['query'],route=route,reason=reason,source_id=source_id,hint=hint,status=result['status'],
                execution_ms=result.get('execution_ms'),planning_ms=result.get('planning_ms'),
                observed_end_to_end_ms=elapsed,logical_end_to_end_ms=reconstructed,
                cached_plan=bool(result.get('cached')),time_source=result.get('time_source'),
                api_ms=generation['api_ms'] if generation else 0,new_llm_call=bool(generation and not generation['cached']),
                label_boundaries=[.1,.5,2,10],nodemo_online=False)
            save(output/'result.json',record);checkpoint(output,'complete',status=result['status'])
            return record
        finally:db.close()
