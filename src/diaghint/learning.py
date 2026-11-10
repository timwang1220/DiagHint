"""Offline-only learning from a generated pool, fixed target holdout, no folds."""
from collections import defaultdict
import copy
import math
from pathlib import Path
import random
import joblib
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix,f1_score,recall_score
from .features import label,walk,CLASS_NAMES
from .model import CardinalityClassifier,ExperienceUtility
from .plans import BOUNDS,Codec,tree,root,hint_bits
from .storage import artifact_dir,read,save,digest,checkpoint


def split_queries(pool,seed):
    manifest=read(Path(pool)/'manifest.json');groups=defaultdict(list)
    for q,h in manifest['query_sql_ids'].items():groups[h].append(q)
    hashes=sorted(groups);random.Random(seed).shuffle(hashes)
    if len(hashes)<5:raise ValueError('At least five distinct target SQL groups are required')
    validation=set(hashes[:max(1,math.ceil(.2*len(hashes)))])
    return {q:('validation' if h in validation else 'train') for h,qs in groups.items() for q in qs}


def observations(pool):
    for folder in sorted((Path(pool)/'queries').iterdir()):
        if not (folder/'native.json').exists():continue
        target=read(folder/'input.json');native=read(folder/'native.json')
        if native['status']!='ok':continue
        yield target,native,read(folder/'records.json') if (folder/'records.json').exists() else []


def classification_metrics(probability,labels):
    y=np.asarray(labels);p=np.asarray(probability);pred=p.argmax(1)
    matrix=confusion_matrix(y,pred,labels=list(range(5)))
    return dict(weighted_recall=float(recall_score(y,pred,average='weighted',zero_division=0)),
        macro_recall=float(recall_score(y,pred,labels=list(range(5)),average='macro',zero_division=0)),
        macro_f1=float(f1_score(y,pred,labels=list(range(5)),average='macro',zero_division=0)),
        class_recall={name:float(matrix[i,i]/max(1,matrix[i].sum())) for i,name in enumerate(CLASS_NAMES)},confusion_matrix=matrix.tolist())


def train_cardinality(config,pool,output):
    output=artifact_dir(output);pool=Path(pool);split=split_queries(pool,config['pipeline']['seed']);items=[];seen=set()
    if not (pool/'COMPLETE.json').exists():raise ValueError('Finish pool generation before training')
    if (output/'classifier.pt').exists():raise ValueError('Refusing to overwrite a classifier; choose a new output directory')
    for target,native,records in observations(pool):
        for result in [native,*[r['result'] for r in records if r.get('result',{}).get('status')=='ok']]:
            identity=(target['query'],result.get('key') or digest(result['document']))
            if identity in seen:continue
            seen.add(identity);t=tree(result['document'],target['sql'],target['metadata'])
            items.append((target['query'],t,root(result['document'])))
    codec=Codec().fit([t for q,t,_ in items if split[q]=='train'])
    arrays={k:[] for k in ['train','validation']};audit=[]
    for q,t,actual in items:
        x=codec.transform(t)
        for i,(path,n) in enumerate(walk(actual)):
            outcome,reason=label(n)
            if not t['supported'][i] or outcome is None:continue
            arrays[split[q]].append((x[i],outcome['class_id'],outcome['log_rho']))
            audit.append(dict(query=q,path=path,split=split[q],plan=digest(t['plan']),**outcome))
    if not all(arrays.values()):raise ValueError('Both target partitions must contain executed Join/Scan nodes')
    def tensors(rows):return torch.tensor(np.stack([r[0] for r in rows])),torch.tensor([r[1] for r in rows]),torch.tensor([r[2] for r in rows],dtype=torch.float32)
    x,y,r=tensors(arrays['train']);vx,vy,vr=tensors(arrays['validation'])
    torch.manual_seed(config['pipeline']['seed']);net=CardinalityClassifier(codec.dim)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);best=float('inf');patience=0;history=[]
    def loss(out,y,r):return F.cross_entropy(out['logits'],y)+.1*F.huber_loss(out['log_rho'],r)
    for epoch in range(150):
        net.train();optimizer.zero_grad();value=loss(net(x),y,r);value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();net.eval()
        with torch.no_grad():validation=float(loss(net(vx),vy,vr))
        history.append(dict(epoch=epoch+1,validation_loss=validation))
        if validation<best:best=validation;patience=0;state=copy.deepcopy(net.state_dict());best_epoch=epoch+1
        else:patience+=1
        if patience>=20:break
    net.load_state_dict(state);net.eval()
    with torch.no_grad():prob=net(vx)['logits'].softmax(-1).numpy()
    torch.save(dict(input_dim=codec.dim,state=state,label_boundaries=BOUNDS),output/'classifier.pt')
    joblib.dump(codec,output/'codec.joblib')
    save(output/'manifest.json',dict(schema=1,kind='cardinality',pool_manifest=digest(read(pool/'manifest.json')),split=split,
        label_boundaries=BOUNDS,best_epoch=best_epoch,model_id=digest({k:v.tolist() for k,v in state.items()})))
    save(output/'validation.json',classification_metrics(prob,vy.numpy()));save(output/'history.json',history);save(output/'labels.json',audit)
    checkpoint(output,'complete')


class Cardinality:
    def __init__(self,folder):
        self.folder=Path(folder);self.manifest=read(self.folder/'manifest.json')
        ck=torch.load(self.folder/'classifier.pt',map_location='cpu',weights_only=True)
        if tuple(ck['label_boundaries'])!=BOUNDS:raise ValueError('Classifier label boundaries differ from the fixed 0.1 protocol')
        self.codec=joblib.load(self.folder/'codec.joblib');self.net=CardinalityClassifier(ck['input_dim'])
        self.net.load_state_dict(ck['state']);self.net.eval().requires_grad_(False)

    def signals(self,t):
        x=torch.tensor(self.codec.transform(t));result={}
        with torch.no_grad():p=self.net(x)['logits'].softmax(-1).numpy()
        for i,path in enumerate(t['paths']):
            if t['supported'][i] and np.isfinite(p[i]).all():
                c=int(p[i].argmax());result[path]=dict(available=True,class_id=c,class_name=CLASS_NAMES[c],probabilities=p[i].tolist())
        return result


def utility_records(pool):
    pool=Path(pool);experiences={x['id']:x for x in read(pool/'experiences.json')};records=[];trees={}
    def put(plan,target):
        t=tree(plan,target['sql'],target['metadata']);key=digest(dict(plan=t['plan'],sql=target['sql'],metadata=target['metadata']))
        trees[key]=t;return key
    for target,native,outcomes in observations(pool):
        a=put(native['document'],target);nd=next((r for r in outcomes if r['phase']=='optimize' and r.get('source_id') is None and r.get('result',{}).get('status')=='ok'),None)
        for r in outcomes:
            result=r.get('result',{})
            if result.get('status')!='ok' or not result.get('execution_ms') or result['execution_ms']<=0:continue
            pg=native['execution_ms'];ms=result['execution_ms'];source_id=r.get('source_id');mode=None
            if r['phase']=='reuse':mode='reuse';keys=[a,put(result['document'],target)];value=math.log(pg/ms)
            elif r['phase']=='optimize' and source_id is None:mode='nodemo';keys=[a];value=math.log(pg/ms)
            elif r['phase']=='optimize' and source_id and nd:
                source=experiences.get(source_id)
                if not source:raise ValueError('Missing frozen source provenance')
                mode='demo';keys=[a,put(source['native_plan'],source),put(source['best_plan'],source)]
                value=math.log(nd['result']['execution_ms']/ms)
            if mode:
                records.append(dict(query=target['query'],mode=mode,trees=keys,y=value,source_id=source_id,
                    hint_bits=hint_bits(r['hint'] if mode=='reuse' else experiences[source_id]['hint'] if source_id else ''),
                    plan_key=result.get('key') or digest(result['document']),time_source=result['time_source'],cached=bool(result.get('cached')),
                    execution_ms=ms,pg_ms=pg,nodemo_ms=nd['result']['execution_ms'] if nd else None))
    return records,trees


def train_utility(config,pool,cardinality,output):
    output=artifact_dir(output);pool=Path(pool);card=Cardinality(cardinality)
    if not (pool/'COMPLETE.json').exists():raise ValueError('Finish pool generation before training')
    if (output/'utility.pt').exists():raise ValueError('Refusing to overwrite a utility model; choose a new output directory')
    split=split_queries(pool,config['pipeline']['seed'])
    if split!=card.manifest['split'] or card.manifest['pool_manifest']!=digest(read(pool/'manifest.json')):raise ValueError('Classifier and utility must use the same pool and target split')
    records,trees=utility_records(pool);fit_queries=sorted({r['query'] for r in records if split[r['query']]=='train'})
    random.Random(config['pipeline']['seed']).shuffle(fit_queries)
    if len(fit_queries)<3:raise ValueError('Not enough target queries with exact utility supervision')
    cal_queries=set(fit_queries[:max(1,math.ceil(.2*len(fit_queries)))])
    for r in records:r['split']='validation' if split[r['query']]=='validation' else 'calibration' if r['query'] in cal_queries else 'fit'
    fit_keys={k for r in records if r['split']=='fit' for k in r['trees']}
    codec=Codec().fit([trees[k] for k in sorted(fit_keys)])
    packed={}
    for key,t in trees.items():
        signals=card.signals(t);diag=np.zeros((len(t['nodes']),6),dtype=np.float32)
        for i,path in enumerate(t['paths']):
            if path in signals:diag[i,:5]=signals[path]['probabilities'];diag[i,5]=1
        packed[key]=(torch.tensor(codec.transform(t)),t['children'],torch.tensor(diag))
    by=defaultdict(list)
    for r in records:by[r['query']].append(r)
    for rr in by.values():
        count=defaultdict(int)
        for r in rr:count[(r['mode'],r['plan_key'])]+=1
        for r in rr:r['weight']=(.5 if r['cached'] else 1.)/count[(r['mode'],r['plan_key'])]
    train=[r for r in records if r['split']=='fit'];val=[r for r in records if r['split']=='validation'];cal=[r for r in records if r['split']=='calibration']
    for mode in ['reuse','demo','nodemo']:
        if not all(any(r['mode']==mode for r in group) for group in [train,val,cal]):raise ValueError('Each task needs exact fit/calibration/validation labels: '+mode)
    torch.manual_seed(config['pipeline']['seed']);net=ExperienceUtility(codec.dim)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);best=float('inf');patience=0;history=[]
    def predict(rows,grad=False):
        # This cache lives for ONE batch/optimizer step only.
        embeddings={};values=[]
        for r in rows:
            for k in r['trees']:
                if k not in embeddings:embeddings[k]=net.encode(*packed[k])
            values.append(net.score(r['mode'],[embeddings[k] for k in r['trees']],torch.tensor(r['hint_bits'])))
        return torch.stack(values)
    def balanced_error(rows,pred):
        groups=defaultdict(list)
        for i,r in enumerate(rows):groups[(r['query'],r['mode'])].append(i)
        return float(np.mean([np.average([abs(float(pred[i])-rows[i]['y']) for i in ids],weights=[rows[i]['weight'] for i in ids]) for ids in groups.values()]))
    train_queries=sorted({r['query'] for r in train});rng=random.Random(config['pipeline']['seed'])
    for epoch in range(200):
        rng.shuffle(train_queries);net.train()
        for start in range(0,len(train_queries),8):
            batch=[r for r in train if r['query'] in set(train_queries[start:start+8])]
            optimizer.zero_grad();pred=predict(batch);losses=[]
            for mode in ['reuse','demo','nodemo']:
                groups=defaultdict(list)
                for i,r in enumerate(batch):
                    if r['mode']==mode:groups[r['query']].append(i)
                terms=[]
                for ids in groups.values():
                    y=pred.new_tensor([batch[i]['y'] for i in ids]);w=pred.new_tensor([batch[i]['weight'] for i in ids])
                    terms.append((F.huber_loss(pred[ids],y,reduction='none')*w).sum()/w.sum())
                if terms:losses.append(torch.stack(terms).mean())
            loss=torch.stack(losses).mean();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step()
        net.eval()
        with torch.no_grad():value=balanced_error(val,predict(val))
        history.append(dict(epoch=epoch+1,validation_mae=value));checkpoint(output,'training',epoch=epoch+1,validation_mae=value)
        if value<best:best=value;patience=0;state=copy.deepcopy(net.state_dict());best_epoch=epoch+1
        else:patience+=1
        if epoch>=39 and patience>=30:break
    net.load_state_dict(state);net.eval()
    with torch.no_grad():raw_cal=predict(cal).numpy();raw_val=predict(val).numpy()
    calibration={}
    for mode in ['reuse','demo','nodemo']:
        ids=[i for i,r in enumerate(cal) if r['mode']==mode];p=raw_cal[ids];y=np.array([cal[i]['y'] for i in ids]);w=np.array([cal[i]['weight'] for i in ids])
        model=Ridge(alpha=10).fit(p[:,None],y,sample_weight=w);slope=float(model.coef_[0])
        supported=slope>0 and np.std(p)>1e-8
        slope=min(slope,2.) if supported else 1.
        calibration[mode]=dict(slope=slope,intercept=float(np.average(y-slope*p,weights=w)),supported=bool(supported))
    predictions=[dict(**r,raw_prediction=float(raw_val[i]),prediction=calibration[r['mode']]['slope']*float(raw_val[i])+calibration[r['mode']]['intercept']) for i,r in enumerate(val)]
    torch.save(dict(input_dim=codec.dim,state=state),output/'utility.pt');joblib.dump(codec,output/'codec.joblib')
    save(output/'manifest.json',dict(schema=1,kind='utility',cardinality_id=card.manifest['model_id'],label_boundaries=BOUNDS,
        pool_manifest=digest(read(pool/'manifest.json')),pool_scope=read(pool/'database_scope.json')['scope'],
        calibration=calibration,best_epoch=best_epoch,split=split,calibration_targets=sorted(cal_queries),
        tasks=dict(reuse='log(PG/Reuse)',demo='log(NoDemo/Demo)',nodemo='log(PG/NoDemo)'),online_nodemo=False))
    save(output/'training_labels.json',records);save(output/'validation_predictions.json',predictions);save(output/'history.json',history)
    checkpoint(output,'complete',validation_mae=best)


class Utility:
    def __init__(self,folder,card):
        folder=Path(folder);self.manifest=read(folder/'manifest.json');self.card=card
        if self.manifest['cardinality_id']!=card.manifest['model_id']:raise ValueError('Classifier/utility checkpoint mismatch')
        ck=torch.load(folder/'utility.pt',map_location='cpu',weights_only=True)
        self.codec=joblib.load(folder/'codec.joblib');self.net=ExperienceUtility(ck['input_dim'])
        self.net.load_state_dict(ck['state']);self.net.eval();self.encodings={}
    def encode(self,plan,sql,metadata):
        t=tree(plan,sql,metadata);key=digest(dict(plan=t['plan'],sql=sql,metadata=metadata))
        if key not in self.encodings:
            signals=self.card.signals(t);diag=torch.zeros(len(t['nodes']),6)
            for i,path in enumerate(t['paths']):
                if path in signals:diag[i]=torch.tensor([*signals[path]['probabilities'],1.])
            with torch.no_grad():self.encodings[key]=self.net.encode(torch.tensor(self.codec.transform(t)),t['children'],diag)
        return self.encodings[key]
    def score(self,mode,embeddings,hint=''):
        with torch.no_grad():raw=float(self.net.score(mode,embeddings,torch.tensor(hint_bits(hint))))
        cal=self.manifest['calibration'][mode];mu=cal['slope']*raw+cal['intercept']
        return dict(log_utility=mu,raw_log_utility=raw,predicted_speedup=math.exp(max(-30,min(30,mu))),calibration_supported=cal['supported'])
