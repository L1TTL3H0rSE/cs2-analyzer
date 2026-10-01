from demoparser2 import DemoParser
import pandas as pd, numpy as np, math
from collections import Counter
ME=76561198105241814
p=DemoParser('match.dem')
start=p.parse_event('begin_new_match')['tick'].max()
fe=p.parse_event('round_freeze_end'); fe=fe[fe.tick>=start].reset_index(drop=True)
re_=p.parse_event('round_end',other=['total_rounds_played']) if 'round_end' in p.list_game_events() else None
d=p.parse_event('player_death', player=['last_place_name','team_num'], other=['total_rounds_played'])
d=d[d.tick>start].copy()
d['user_steamid']=pd.to_numeric(d.user_steamid,errors='coerce'); d['attacker_steamid']=pd.to_numeric(d.attacker_steamid,errors='coerce')
def rnd(t):
    k=fe[fe.tick<=t]; return len(k)  # номер раунда 1..
d['round']=d.tick.apply(rnd)
d['tsec']=d.apply(lambda r:(r.tick-fe.tick.iloc[r['round']-1])/64,axis=1)
print("раундов:",len(fe),"смертей:",len(d))
me_k=d[d.attacker_steamid==ME]; me_d=d[d.user_steamid==ME]
print(f"\nУБИЙСТВА {len(me_k)}, смертей {len(me_d)}")
w=me_k.groupby('weapon').agg(kills=('headshot','size'),hs=('headshot','sum'))
w['hs%']=(w.hs/w.kills*100).round(0); print(w.sort_values('kills',ascending=False))
side=lambda team:'CT' if team==3 else 'T'
# смерти: размен и первая смерть раунда
print("\nТВОИ СМЕРТИ:")
rows=[]
for _,r in me_d.iterrows():
    after=d[(d.tick>r.tick)&(d.tick<=r.tick+5*64)&(d.user_steamid==r.attacker_steamid)]
    first=d[d['round']==r['round']].tick.min()==r.tick
    rows.append((r['round'],side(r.user_team_num),round(r.tsec),r.user_last_place_name,r.attacker_name,r.weapon,'размен' if len(after) else 'БЕЗ размена','ПЕРВАЯ смерть раунда' if first else ''))
for x in rows: print(x)
print("без размена:",sum(1 for x in rows if x[6]=='БЕЗ размена'),"из",len(rows))
# опенинги
print("\nОПЕНИНГИ (первое убийство раунда с твоим участием):")
op=d.sort_values('tick').groupby('round').head(1)
mine=op[(op.attacker_steamid==ME)|(op.user_steamid==ME)]
for _,r in mine.iterrows(): print(r['round'], 'выиграл' if r.attacker_steamid==ME else 'проиграл', side(r.user_team_num if r.user_steamid==ME else r.attacker_team_num), r.user_last_place_name if r.user_steamid==ME else r.attacker_last_place_name, r.weapon, f"{r.tsec:.0f}с")
print(f"участвовал в {len(mine)} опенингах из {len(op)}")
me_k.to_pickle('mek.pkl'); d.to_pickle('d.pkl'); fe.to_pickle('fe.pkl')
