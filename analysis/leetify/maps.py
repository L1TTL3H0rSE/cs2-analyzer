import json, statistics as st, math
from collections import defaultdict
R=json.load(open('rows.json'))
S5='2026-07-06'
POOL_S4={'de_ancient','de_anubis','de_dust2','de_inferno','de_mirage','de_nuke','de_overpass'}
POOL_S5={'de_ancient','de_anubis','de_dust2','de_inferno','de_mirage','de_nuke','de_cache'}
mt={}
import glob
for f in glob.glob('m/*.json'):
    m=json.load(open(f)); mt[m['id']]=(m['finished_at'],m['map_name'])
def inpool(mid):
    t,mp=mt[mid]; return mp in (POOL_S5 if t>=S5 else POOL_S4)
me=[r for r in R if r['me'] and r['complete']]
ex=[r for r in me if not inpool(r['mid'])]
print("исключено вне пула:", [(mt[r['mid']][1],mt[r['mid']][0][:10],r['mode']) for r in ex])
me=[r for r in me if inpool(r['mid'])]
by=defaultdict(list)
for r in me: by[mt[r['mid']][1]].append(r['s'])
print(f"\n{'карта':11} n  винр  rating   CT     T    ADR  TTD  точн  голова  HS-килл  тебя разм.")
for mp,L in sorted(by.items(),key=lambda x:-len(x[1])):
    w=sum(s['rounds_won']>s['rounds_lost'] for s in L)
    tr=[s['traded_deaths_success_percentage'] for s in L if s['traded_death_opportunities']]
    print(f"{mp[3:]:11}{len(L):2} {w/len(L)*100:4.0f}% {st.mean(s['leetify_rating'] for s in L):+.3f} {st.mean(s['ct_leetify_rating'] for s in L):+.3f} {st.mean(s['t_leetify_rating'] for s in L):+.3f} {sum(s['total_damage'] for s in L)/sum(s['rounds_count'] for s in L):5.1f} {st.median(s['reaction_time']*1000 for s in L):4.0f} {sum(s['shots_hit_enemy_spotted'] for s in L)/sum(s['shots_fired_enemy_spotted'] for s in L)*100:5.1f} {st.mean(s['accuracy_head'] for s in L)*100:6.1f} {sum(s['total_hs_kills'] for s in L)/sum(s['total_kills'] for s in L)*100:7.0f} {st.mean(tr)*100:8.0f}")
# lobby per-map CT/T for reference
