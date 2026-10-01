import json, os, statistics as st, datetime as dt, math
from collections import defaultdict
R=json.load(open('rows.json'))
ME='76561198105241814'; FR=os.environ.get('FRIEND_STEAM_ID','')  # steamid частого тиммейта — в env, не в git
bym=defaultdict(list)
for r in R: bym[r['mid']].append(r)
G=[]
for mid,rs in bym.items():
    me=[r for r in rs if r['me']][0]
    if not me['complete']: continue
    s=me['s']
    m=json.load(open(f'm/{mid}.json'))
    t=dt.datetime.fromisoformat(m['finished_at'][:19])+dt.timedelta(hours=3)
    mates={r['s']['steam64_id'] for r in rs if r['team']=='mate'}
    G.append(dict(t=t,mode=me['mode'],map=m['map_name'],win=s['rounds_won']>s['rounds_lost'],tie=s['rounds_won']==s['rounds_lost'],s=s,
        mate_rating=st.mean(r['s']['leetify_rating'] for r in rs if r['team']=='mate'),
        opp_rating=st.mean(r['s']['leetify_rating'] for r in rs if r['team']=='opp'),
        friend=FR in mates, stack=sum(1 for r in rs if r['team']=='mate' and r['s']['steam64_id'] in {FR}) ))
G.sort(key=lambda g:g['t'])
# sessions
idx=0; prev=None
for g in G:
    idx = idx+1 if prev and (g['t']-prev).total_seconds()<75*60 else 1
    g['sidx']=idx; prev=g['t']
def summ(L):
    n=len(L); w=sum(g['win'] for g in L)
    return f"n={n:2} винрейт {w/n*100:3.0f}%  rating {st.mean(g['s']['leetify_rating'] for g in L):+.3f}  TTD {st.median(g['s']['reaction_time']*1000 for g in L):3.0f}  точн {st.mean(g['s']['accuracy_enemy_spotted']*100 for g in L):4.1f}"
print("=== номер матча в сессии (пауза < 75 мин) ===")
for k,f in [('1-й',lambda g:g['sidx']==1),('2-й',lambda g:g['sidx']==2),('3-й',lambda g:g['sidx']==3),('4-й+',lambda g:g['sidx']>=4)]:
    print(k, summ([g for g in G if f(g)]))
print("\n=== время суток (МСК) ===")
for k,f in [('утро 5–12',lambda h:5<=h<12),('день 12–18',lambda h:12<=h<18),('вечер 18–24',lambda h:18<=h),('ночь 0–5',lambda h:h<5)]:
    L=[g for g in G if f(g['t'].hour)]; 
    if L: print(k, summ(L))
print("\n=== будни / выходные ===")
print('будни   ', summ([g for g in G if g['t'].weekday()<5])); print('выходные', summ([g for g in G if g['t'].weekday()>=5]))
print("\n=== с другом (FRIEND_STEAM_ID) / без ===")
print('с ним ', summ([g for g in G if g['friend']])); print('без  ', summ([g for g in G if not g['friend']]))
print("\n=== карты ===")
maps=defaultdict(list)
for g in G: maps[g['map']].append(g)
for k,L in sorted(maps.items(),key=lambda x:-len(x[1])):
    if len(L)>=4: print(f"{k:12}", summ(L), f" CT {st.mean(g['s']['ct_leetify_rating'] for g in L):+.3f} T {st.mean(g['s']['t_leetify_rating'] for g in L):+.3f}")
print("\n=== что отличает твои победы от поражений (размер эффекта d) ===")
W=[g for g in G if g['win']]; Lo=[g for g in G if not g['win'] and not g['tie']]
feats={'урон за раунд':lambda s:s['dpr'],'точн. при замеч.':lambda s:s['accuracy_enemy_spotted'],'время до урона':lambda s:-s['reaction_time'],'preaim':lambda s:-s['preaim'],
 'в голову':lambda s:s['accuracy_head'],'контрстрейф':lambda s:s['counter_strafing_shots_good_ratio'],'выживаемость':lambda s:s['rounds_survived_percentage'],
 'разменял тиммейта %':lambda s:(s['trade_kills_success_percentage'] if s['trade_kill_opportunities'] else None),'тебя разменяли %':lambda s:(s['traded_deaths_success_percentage'] if s['traded_death_opportunities'] else None),
 'флешки: враги/флешка':lambda s:s['flashbang_hit_foe']/s['flashbang_thrown'] if s['flashbang_thrown'] else 0,'гранат за раунд':lambda s:(s['flashbang_thrown']+s['he_thrown']+s['molotov_thrown']+s['smoke_thrown'])/s['rounds_count'],
 'утилита при смерти':lambda s:-s['utility_on_death_avg'],'рейтинг за CT':lambda s:s['ct_leetify_rating'],'рейтинг за T':lambda s:s['t_leetify_rating']}
res=[]
for k,f in feats.items():
    a=[v for v in (f(g["s"]) for g in W) if v is not None]; b=[v for v in (f(g["s"]) for g in Lo) if v is not None]
    sd=math.sqrt((st.pvariance(a)+st.pvariance(b))/2); res.append((( st.mean(a)-st.mean(b))/sd,k))
for d,k in sorted(res,reverse=True): print(f"{k:22} d={d:+.2f}")
print(f"\nпобед {len(W)}, поражений {len(Lo)}")
print("\n=== насколько исход зависит от тебя vs тиммейтов ===")
for lab,L in [('победы',W),('поражения',Lo)]:
    print(lab, f"твой rating {st.mean(g['s']['leetify_rating'] for g in L):+.3f}  средний тиммейт {st.mean(g['mate_rating'] for g in L):+.3f}  средний соперник {st.mean(g['opp_rating'] for g in L):+.3f}")
