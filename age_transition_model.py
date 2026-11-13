"""Age-proxy analogue extension. Stdlib inference; no global UID cutoff.
First-seen elapsed time is explicitly left-censored for archive-initial games.
UID-age feature is relative to source chart's current maximum, not future max.
"""
import math
import transition_model as tm
# Frozen by earliest-window selection; elapsed first-seen age was NOT selected.
DEFAULT={'age_weight':0.0,'uid_weight':0.5,'age_bw':1.0,'uid_bw':1.0}

def training(rows,config=None):
    cfg={**tm.DEFAULT,**DEFAULT,**(config or {})}
    samples,tails,meta,current=tm.training(rows,cfg)
    rows,times,charts,features=tm.prepare(rows)
    first={};age_features=[]
    initial=set(charts[0])
    for i,c in enumerate(charts):
        maximum=max(c,default=0);f={}
        for uid in c:
            first.setdefault(uid,times[i])
            f[uid]=(min(7.0,(times[i]-first[uid])/24),float(uid in initial),min(100.0,max(0,maximum-uid)/1000000))
        age_features.append(f)
    aug=[];n=0
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            lag=times[j]-times[i]
            if lag<6:continue
            if lag>72:break
            for uid,f in features[i].items():
                if f[0]>60:continue
                aug.append(samples[n]+age_features[i][uid]);n+=1
    assert n==len(samples)
    meta['uncensored_current_games']=sum(not a[1] for a in age_features[-1].values())
    return aug,tails,meta,{u:current[u]+age_features[-1][u] for u in current}

def age_distance(a,b):
    # Both exact ages: absolute difference. A left-censored value is only
    # a lower bound, so penalize only violations of that lower bound.
    age,censor=a;other,ocensor=b
    if censor and ocensor:return 0.0
    if censor:return max(0,age-other)
    if ocensor:return max(0,other-age)
    return abs(age-other)

def forecast(rows,places=10,config=None,return_meta=False):
    cfg={**tm.DEFAULT,**DEFAULT,**(config or {})}
    samples,tails,meta,current=training(rows,cfg)
    if cfg['age_weight'] == 0:
        # Identical sufficient statistics share a kernel; sum their weights once.
        # This preserves the estimator, unlike discarding dense hourly reads.
        grouped = {}
        for sr,sm,sp,b,tr,wt,age,censor,gap in samples:
            key = (sr,sm,sp,b,tr,gap)
            grouped[key] = grouped.get(key,0.0) + wt
        samples = [(sr,sm,sp,b,tr,wt,0.0,0.0,gap)
                   for (sr,sm,sp,b,tr,gap),wt in grouped.items()]
        meta['kernel_rows'] = len(samples)
    candidates={u:f for u,f in current.items() if f[0]<=cfg['depth']}
    raw={u:[0.0]*places for u in candidates}
    kernels={r:[math.exp(-abs(r-p)/cfg['outcome_bw']) if r<100 else 0 for p in range(1,places+1)] for r in range(1,121)}
    for u,f in candidates.items():
        norm=[0.0]*6;vals=[[0.0]*places for _ in range(6)]
        for sr,sm,sp,b,tr,wt,age,censor,gap in samples:
            d=abs(f[0]-sr)/cfg['rank_bw']
            if d>6:continue
            d+=cfg['momentum']*(abs(f[1]-sm)/cfg['momentum_bw']+abs(f[2]-sp)/cfg['play_bw'])
            if cfg['age_weight']:
                d+=cfg['age_weight']*age_distance(f[3:5],(age,censor))/cfg['age_bw']
            d+=cfg['uid_weight']*abs(f[5]-gap)/cfg['uid_bw']
            w=wt*math.exp(-d);norm[b]+=w;k=kernels[tr]
            for p in range(places):vals[b][p]+=w*k[p]
        for p in range(places):
            components=[vals[b][p]/norm[b] for b in range(6) if norm[b]>0]
            raw[u][p]=sum(components)/max(1,len(components))+1e-12
    probabilities=[];total={u:0 for u in candidates}
    for p in range(places):
        norm=sum(v[p] for v in raw.values())
        tail=max(.08,min(.65,tails[p][0]/tails[p][1])) if cfg['tail']=='learned' else float(cfg['tail'])
        for u,v in raw.items():
            prob=round((1-tail)*v[p]/max(norm,1e-30),6)
            if prob>0:probabilities.append(dict(universe_id=u,place=p+1,p=prob));total[u]+=prob
    out=dict(probabilities=probabilities,top10=sorted(candidates,key=lambda u:(-total[u],current[u][0],u))[:places])
    return (out,meta) if return_meta else out


def main():
    import json
    import os
    inp=os.environ.get('MAKEFUN_INPUT_DIR','/input')
    out=os.environ.get('MAKEFUN_OUTPUT_DIR','/output')
    with open(os.path.join(inp,'task.json'),encoding='utf-8') as f:task=json.load(f)
    with open(os.path.join(inp,'history.ndjson'),encoding='utf-8') as f:rows=[json.loads(s) for s in f if s.strip()]
    payload=forecast(rows,int(task['places']))
    os.makedirs(out,exist_ok=True)
    with open(os.path.join(out,task['output']),'w',encoding='utf-8') as f:json.dump(payload,f,separators=(',',':'))

if __name__=='__main__':main()
