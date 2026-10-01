import json
from collections import Counter
R=json.load(open('rows.json'))
def q(xs,p):
    xs=sorted(xs); k=(len(xs)-1)*p; f=int(k); c=min(f+1,len(xs)-1); return xs[f]+(xs[c]-xs[f])*(k-f)
full=[r for r in R if r['full'] and not r['me']]
# референс: все режимы вместе, пороги p97.5 / p2.5 — шире, чем FACEIT-only из-за малой выборки
F=[r['s'] for r in full]
th={'ttd':q([s['reaction_time'] for s in F],.025),'pre':q([s['preaim'] for s in F],.025),
    'acc':q([s['accuracy_enemy_spotted'] for s in F],.975),'hs':q([s['accuracy_head'] for s in F],.975),'adr':q([s['dpr'] for s in F],.975)}
print({k:round(v,3) for k,v in th.items()})
def flags(s):
    f=[]
    if s['reaction_time']<=th['ttd']: f.append('ttd')
    if s['preaim']<=th['pre']: f.append('preaim')
    if s['accuracy_enemy_spotted']>=th['acc']: f.append('acc')
    if s['accuracy_head']>=th['hs'] and s['shots_hit_foe']>=30: f.append('head')
    if s['dpr']>=th['adr']: f.append('adr')
    return f
by=Counter(); multi=Counter(); tot=Counter(); lst=[]
for r in full:
    fl=flags(r['s']); tot[r['mode']]+=1
    if fl: by[r['mode']]+=1
    if len(fl)>=2: multi[r['mode']]+=1; lst.append((r['mode'],r['s']['name'],r['s']['steam64_id'],fl,r['mid'][:8],round(r['s']['leetify_rating'],3)))
for m in tot: print(f"{m:12} строк {tot[m]:4}  хоть 1 флаг {by[m]/tot[m]*100:4.1f}%  2+ флага {multi[m]:3} ({multi[m]/tot[m]*100:.1f}%)")
print()
for x in sorted(lst): print(x)
