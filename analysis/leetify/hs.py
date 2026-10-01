import json, statistics as st, math
R=json.load(open('rows.json'))
full=[r for r in R if r['full']]
oth=[r['s'] for r in full if not r['me']]; me=[r['s'] for r in full if r['me']]
def corr(x,y):
    mx,my=st.mean(x),st.mean(y); return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
def pct(v,xs,low=False): return sum(1 for x in xs if (x>v if low else x<v))/len(xs)*100
F={'точн. при замеч.':lambda s:s['accuracy_enemy_spotted'],'спрей':lambda s:s['spray_accuracy'],'preaim':lambda s:s['preaim'],'время до урона':lambda s:s['reaction_time'],
   'выстрелов на убийство':lambda s:s['shots_fired']/max(1,s['total_kills']),'доля выстрелов по видимому':lambda s:s['shots_fired_enemy_spotted']/max(1,s['shots_fired']),'контрстрейф':lambda s:s['counter_strafing_shots_good_ratio']}
print("корреляция с точностью в голову по всем игрокам (n=%d):"%len(oth))
for k,f in F.items(): print(f"  {k:28} r={corr([f(s) for s in oth],[s['accuracy_head'] for s in oth]):+.2f}")
print("\nты vs лобби:")
for k,f in F.items():
    v=st.median(f(s) for s in me); print(f"  {k:28} ты {v:.3f}  медиана лобби {st.median(f(s) for s in oth):.3f}  перцентиль {pct(v,[f(s) for s in oth]):.0f}")
# роли: прокси
print("\nролевые прокси (на раунд), ты vs лобби:")
P={'твою смерть можно было разменять (ты первый в контакте)':lambda s:s['traded_deaths_opportunities_per_round'],
   'ты мог разменять тиммейта (ты второй)':lambda s:s['trade_kill_opportunities_per_round'],
   'выживаемость':lambda s:s['rounds_survived_percentage'],'флешек за раунд':lambda s:s['flashbang_thrown']/s['rounds_count'],
   'смертей за раунд':lambda s:s['total_deaths']/s['rounds_count'],'мультикиллов (2k+) за раунд':lambda s:(s['multi2k']+s['multi3k']+s['multi4k']+s['multi5k'])/s['rounds_count'],
   'ассистов за раунд':lambda s:s['total_assists']/s['rounds_count']}
for k,f in P.items():
    o=[x for x in (f(s) for s in oth) if x is not None]; v=st.median(x for x in (f(s) for s in me) if x is not None)
    print(f"  {k:58} ты {v:.3f}  лобби {st.median(o):.3f}  перцентиль {pct(v,o):.0f}")
