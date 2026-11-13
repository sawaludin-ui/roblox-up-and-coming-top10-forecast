"""Causal 72-hour analogue forecaster; standard library only, no network.

Each real source/target read pair has a 6..72h lag and ends before forecast
cutoff. Source features use only source-and-earlier reads. Rank/momentum
analogue kernels transfer future ranks, never universe identity. Six horizon
bins are averaged equally. Per-place named probabilities have an unnamed tail.
"""
import bisect
import json
import math
import os
from datetime import datetime

# Frozen using ONLY the earliest chronological development window.
DEFAULT = dict(rank_bw=3.0, momentum_bw=8.0, play_bw=1.0, outcome_bw=2.0,
               momentum=1.0, tail=0.2, depth=40, recency=120.0)

def hours(stamp):
    return datetime.fromisoformat(stamp.replace('Z','+00:00')).timestamp()/3600

def prepare(rows):
    unique = {r['at']:r for r in rows}
    rows = sorted(unique.values(),key=lambda r:hours(r['at']))
    times=[hours(r['at']) for r in rows]
    charts=[{int(g['universe_id']):g for g in r.get('up_and_coming',[])} for r in rows]
    features=[]
    for i,c in enumerate(charts):
        # Nearest read at least six hours old, preferably around twelve hours.
        j=bisect.bisect_right(times,times[i]-12)-1
        if j<0: j=0
        prev=charts[j]
        dt=max(6,times[i]-times[j])
        f={}
        for uid,g in c.items():
            rank=float(g['rank']); old=prev.get(uid)
            delta=(rank-float(old['rank']))*12/dt if old else -12.0
            p=float(g.get('playing') or 0); op=float((old or {}).get('playing') or 0)
            growth=(math.log(max(1,p))-math.log(max(1,op)))*12/dt if old else 0.0
            f[uid]=(rank,max(-30,min(30,delta)),max(-3,min(3,growth)))
        features.append(f)
    return rows,times,charts,features

def training(rows, config=None):
    cfg=dict(DEFAULT);cfg.update(config or {})
    rows,times,charts,features=prepare(rows)
    samples=[]; tails=[[1.0,2.0] for _ in range(10)]
    pairs=0; horizon_pairs=[0]*6
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            lag=times[j]-times[i]
            if lag<6:continue
            if lag>72:break
            b=min(5,int((lag-0.000001)//12))
            weight=0.5**((times[-1]-times[j])/cfg['recency'])
            pairs+=1;horizon_pairs[b]+=1
            # Known candidate set at this historical source, not at target.
            known={u for u,f in features[i].items() if f[0]<=cfg['depth']}
            for u,g in charts[j].items():
                p=int(g['rank'])
                if p<=10:
                    tails[p-1][0]+=weight*(u not in known)
                    tails[p-1][1]+=weight
            for uid,f in features[i].items():
                if f[0]>60:continue
                target=charts[j].get(uid)
                rank=int(target['rank']) if target else 120
                samples.append((f[0],f[1],f[2],b,rank,weight))
    return samples,tails,dict(reads=len(rows),pairs=pairs,analogue_rows=len(samples),horizon_pairs=horizon_pairs),features[-1]

def forecast(rows, places=10, config=None, return_meta=False):
    cfg=dict(DEFAULT);cfg.update(config or {})
    samples,tails,meta,current=training(rows,cfg)
    candidates={u:f for u,f in current.items() if f[0]<=cfg['depth']}
    raw={u:[0.0]*places for u in candidates}
    kernels={r:[math.exp(-abs(r-p)/cfg['outcome_bw']) if r<100 else 0.0 for p in range(1,places+1)] for r in range(1,121)}
    for uid,f in candidates.items():
        norm=[0.0]*6; vals=[[0.0]*places for _ in range(6)]
        for sr,sm,sp,b,tr,wt in samples:
            distance=abs(f[0]-sr)/cfg['rank_bw']
            if distance>6:continue
            distance+=cfg['momentum']*(abs(f[1]-sm)/cfg['momentum_bw']+abs(f[2]-sp)/cfg['play_bw'])
            w=wt*math.exp(-distance)
            norm[b]+=w
            k=kernels.get(tr,kernels[120])
            for p in range(places): vals[b][p]+=w*k[p]
        for p in range(places):
            components=[vals[b][p]/norm[b] for b in range(6) if norm[b]>0]
            raw[uid][p]=sum(components)/max(1,len(components))+1e-12
    probabilities=[];total_prob={u:0.0 for u in candidates}
    for p in range(places):
        norm=sum(v[p] for v in raw.values())
        tail=max(.08,min(.65,tails[p][0]/tails[p][1])) if cfg['tail']=='learned' else float(cfg['tail'])
        for uid,v in raw.items():
            prob=round((1-tail)*v[p]/max(1e-30,norm),6)
            if prob>0:
                probabilities.append(dict(universe_id=uid,place=p+1,p=prob))
                total_prob[uid]+=prob
    top=sorted(candidates,key=lambda u:(-total_prob[u],current[u][0],u))[:places]
    for u in sorted(current,key=lambda u:(current[u][0],u)):
        if len(top)>=places:break
        if u not in top:top.append(u)
    out=dict(probabilities=probabilities,top10=top)
    meta['estimated_tail']=[round(max(.08,min(.65,a/b)),4) for a,b in tails]
    meta['tail']=meta['estimated_tail'] if cfg['tail']=='learned' else [float(cfg['tail'])]*places
    return (out,meta) if return_meta else out

def main():
    inp=os.environ.get('MAKEFUN_INPUT_DIR','/input');out=os.environ.get('MAKEFUN_OUTPUT_DIR','/output')
    with open(os.path.join(inp,'task.json'),encoding='utf-8') as f:task=json.load(f)
    with open(os.path.join(inp,'history.ndjson'),encoding='utf-8') as f:rows=[json.loads(s) for s in f if s.strip()]
    payload=forecast(rows,int(task['places']))
    os.makedirs(out,exist_ok=True)
    with open(os.path.join(out,task['output']),'w',encoding='utf-8') as f:json.dump(payload,f,separators=(',',':'))
if __name__=='__main__':main()
