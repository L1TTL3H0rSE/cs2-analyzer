import json, statistics as st
R=json.load(open('rows.json'))
def q(xs,p):
    xs=sorted(xs); k=(len(xs)-1)*p; f=int(k); c=min(f+1,len(xs)-1); return xs[f]+(xs[c]-xs[f])*(k-f)
MET=[('Время до урона, мс',lambda s:s['reaction_time']*1000,True),
     ('Preaim, °',lambda s:s['preaim'],True),
     ('Точн. при замеч., %',lambda s:s['accuracy_enemy_spotted']*100,False),
     ('Точн. в голову, %',lambda s:s['accuracy_head']*100,False),
     ('Контрстрейф, %',lambda s:s['counter_strafing_shots_good_ratio']*100,False),
     ('Спрей, %',lambda s:s['spray_accuracy']*100,False),
     ('Урон за раунд',lambda s:s['dpr'],False),
     ('Leetify rating',lambda s:s['leetify_rating'],False),
     ('Флешек за раунд',lambda s:s['flashbang_thrown']/s['rounds_count'],False),
     ('Ослепл. врагов на флешку',lambda s:s['flashbang_hit_foe']/s['flashbang_thrown'] if s['flashbang_thrown'] else None,False),
     ('Разменяли смерть, %',lambda s:s['traded_deaths_success_percentage']*100 if s['traded_death_opportunities'] else None,False)]
for mode in ['Premier','Competitive','FACEIT']:
    oth=[r['s'] for r in R if r['full'] and not r['me'] and r['mode']==mode]
    me=[r['s'] for r in R if r['full'] and r['me'] and r['mode']==mode]
    print(f"\n=== {mode}: других {len(oth)} строк, моих {len(me)}")
    for name,f,low in MET:
        o=[v for v in (f(s) for s in oth) if v is not None]; m=[v for v in (f(s) for s in me) if v is not None]
        mm=st.median(m); pct=sum(1 for x in o if (x>mm if low else x<mm))/len(o)*100
        fmt=(lambda v:f"{v:.3f}") if 'rating' in name else ((lambda v:f"{v:.2f}") if 'флешк' in name.lower() or 'Ослепл' in name else (lambda v:f"{v:.1f}"))
        print(f"{name:26} p5 {fmt(q(o,.05)):>7} p25 {fmt(q(o,.25)):>7} мед {fmt(q(o,.5)):>7} p75 {fmt(q(o,.75)):>7} p95 {fmt(q(o,.95)):>7} p99 {fmt(q(o,.99) if not low else q(o,.01)):>7} | я мед {fmt(mm):>7} лучше {pct:3.0f}%")
