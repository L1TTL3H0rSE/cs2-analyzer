import json, statistics as st, random
d=json.load(open('hist.json'))
rows=[]
for m in d:
    s=m['stats'][0]; sc=[t['score'] for t in m['team_scores']]
    complete = max(sc)>=13 or sc==[12,12]
    tot=sum(sc)
    full = s['rounds_count']>=0.9*tot
    if not (complete and full): continue
    rows.append(dict(t=m['finished_at'], src=m['data_source'], map=m['map_name'], r=s))
rows.sort(key=lambda x:x['t'])
def per(t):
    return 'A' if t<'2026-07-26' else ('B' if t<'2026-09-14' else 'C')
for x in rows: x['p']=per(x['t'])
print("полных матчей:",len(rows), {p:sum(1 for x in rows if x['p']==p) for p in 'ABC'})
from collections import Counter
for p in 'ABC': print(p, Counter(x['src'] for x in rows if x['p']==p))
# aggregate metrics per period
def agg(sub):
    R=[x['r'] for x in sub]
    g=lambda k:sum(r[k] for r in R)
    return {
     'ttd_ms (медиана)': st.median(r['reaction_time']*1000 for r in R),
     'preaim ° (медиана)': st.median(r['preaim'] for r in R),
     'точн. при замеч., %': 100*g('shots_hit_enemy_spotted')/g('shots_fired_enemy_spotted'),
     'точн. в голову, %': 100*g('shots_hit_foe_head')/max(1,g('shots_hit_foe')) if g('shots_hit_foe_head') else 100*st.mean(r['accuracy_head'] for r in R),
     'контрстрейф, %': 100*g('counter_strafing_shots_good')/g('counter_strafing_shots_all'),
     'спрей, % (ср.)': 100*st.mean(r['spray_accuracy'] for r in R),
     'leetify rating (ср.)': st.mean(r['leetify_rating'] for r in R),
     'урон за раунд': g('total_damage')/g('rounds_count'),
     'K/D': g('total_kills')/max(1,g('total_deaths')),
    }
def boot(sub1,sub2,key,n=4000):
    random.seed(1); diffs=[]
    for _ in range(n):
        a=[random.choice(sub1) for _ in sub1]; b=[random.choice(sub2) for _ in sub2]
        diffs.append(agg(b)[key]-agg(a)[key])
    diffs.sort(); return diffs[int(.025*n)], diffs[int(.975*n)]
P={p:[x for x in rows if x['p']==p] for p in 'ABC'}
keys=list(agg(rows).keys())
print(f"{'метрика':24}{'A до':>9}{'B клава':>9}{'C +моник':>10}   B−A [95% CI]            C−B [95% CI]           C−A [95% CI]")
for k in keys:
    a,b,c=(agg(P[p])[k] for p in 'ABC')
    ba=boot(P['A'],P['B'],k); cb=boot(P['B'],P['C'],k); ca=boot(P['A'],P['C'],k)
    f=lambda v: f"{v:.3f}" if 'rating' in k else f"{v:.1f}"
    print(f"{k:24}{f(a):>9}{f(b):>9}{f(c):>10}   [{f(ba[0])}, {f(ba[1])}]   [{f(cb[0])}, {f(cb[1])}]   [{f(ca[0])}, {f(ca[1])}]")
json.dump([dict(t=x['t'],p=x['p'],src=x['src'],ttd=x['r']['reaction_time']*1000,pre=x['r']['preaim'],cs=x['r']['counter_strafing_shots_good_ratio']*100,acc=x['r']['accuracy_enemy_spotted']*100,rt=x['r']['leetify_rating']) for x in rows],open('series.json','w'))

print("\n--- только Premier (matchmaking) ---")
Q={p:[x for x in rows if x['p']==p and x['src']=='matchmaking'] for p in 'ABC'}
print({p:len(Q[p]) for p in 'ABC'})
for k in ['ttd_ms (медиана)','точн. при замеч., %','контрстрейф, %','урон за раунд']:
    print(f"{k:24}", *(f"{agg(Q[p])[k]:8.1f}" for p in 'ABC'), " C−A CI", [round(v,1) for v in boot(Q['A'],Q['C'],k)])
print("\n--- тренд ВНУТРИ периода A (не было ли улучшения ещё до клавы) ---")
import datetime as dt
def days(t): return (dt.datetime.fromisoformat(t[:19])-dt.datetime(2026,5,1)).days
for key,f in [('ttd',lambda r:r['reaction_time']*1000),('cs',lambda r:r['counter_strafing_shots_good_ratio']*100),('acc',lambda r:r['accuracy_enemy_spotted']*100)]:
    for p in 'AC':
        xs=[days(x['t']) for x in P[p]]; ys=[f(x['r']) for x in P[p]]
        mx,my=st.mean(xs),st.mean(ys); sl=sum((a-mx)*(b-my) for a,b in zip(xs,ys))/sum((a-mx)**2 for a in xs)
        print(key,p,f"наклон {sl*30:+.1f} за 30 дней, диапазон дат {min(xs)}–{max(xs)} дней")
print("\n--- первые 5 vs остальные матчи после монитора (привыкание) ---")
C=P['C']
for k in ['ttd_ms (медиана)','точн. при замеч., %','контрстрейф, %']:
    print(k, round(agg(C[:5])[k],1), round(agg(C[5:])[k],1))
print("\n--- то же после клавиатуры ---")
B=P['B']
for k in ['ttd_ms (медиана)','контрстрейф, %']:
    print(k, round(agg(B[:5])[k],1), round(agg(B[5:])[k],1))
