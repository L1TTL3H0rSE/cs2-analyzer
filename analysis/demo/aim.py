from demoparser2 import DemoParser
import pandas as pd, numpy as np, math
ME='76561198105241814'
RIFLES={'ak47','m4a1_silencer','m4a1','galilar','famas','aug','sg556','m4a4'}
p=DemoParser('match.dem')
start=p.parse_event('begin_new_match')['tick'].max()
wf=p.parse_event('weapon_fire'); wf=wf[wf.tick>start].copy()
hu=p.parse_event('player_hurt'); hu=hu[hu.tick>start].copy()
for df,c in [(wf,'user_steamid'),(hu,'attacker_steamid'),(hu,'user_steamid')]: df[c]=df[c].astype(str)
wf['weapon']=wf.weapon.str.replace('weapon_','')
hu=hu[hu.weapon.isin(RIFLES)]
eng=[]; last={}
for _,h in hu.sort_values('tick').iterrows():
    key=(h.attacker_steamid,h.user_steamid)
    if key in last and h.tick-last[key]<3*64: last[key]=h.tick; continue
    last[key]=h.tick
    shots=wf[(wf.user_steamid==h.attacker_steamid)&(wf.tick<=h.tick)&(wf.tick>=h.tick-2*64)].tick.sort_values().tolist()
    if not shots: continue
    # первый выстрел серии: идём назад, пока разрыв < 0.6с
    s=shots[-1]
    for t in reversed(shots[:-1]):
        if s-t<0.6*64: s=t
        else: break
    eng.append(dict(att=h.attacker_steamid,vic=h.user_steamid,first=int(s),pre=int(s-int(0.3*64)),hitgroup=h.hitgroup))
E=pd.DataFrame(eng); print("дуэлей с рифла:",len(E))
ticks=sorted(set(E['first'])|set(E['pre']))
tk=p.parse_ticks(['X','Y','Z','pitch','yaw','duck_amount'],ticks=ticks)
tk['steamid']=tk.steamid.astype(str); tk=tk.set_index(['tick','steamid'])
def err(att,vic,t):
    try: a=tk.loc[(t,att)]; v=tk.loc[(t,vic)]
    except KeyError: return None
    ea=64-18*a.duck_amount; ev=64-18*v.duck_amount-2  # глаза ≈ голова
    dx,dy,dz=v.X-a.X,v.Y-a.Y,(v.Z+ev)-(a.Z+ea); hz=math.hypot(dx,dy)
    if hz<50: return None
    need_p=-math.degrees(math.atan2(dz,hz)); need_y=math.degrees(math.atan2(dy,dx))
    ey=(a.yaw-need_y+180)%360-180
    return a.pitch-need_p, ey, hz
out=[]
for _,e in E.iterrows():
    for when,t in [('first',e['first']),('pre',e['pre'])]:
        r=err(e.att,e.vic,t)
        if r: out.append(dict(att=e.att,when=when,v=r[0],h=r[1],dist=r[2],hg=e.hitgroup))
O=pd.DataFrame(out)
names=p.parse_player_info(); names['steamid']=names.steamid.astype(str); nm=dict(zip(names.steamid,names.name))
f=O[O.when=='first']; pr=O[O.when=='pre']
g=f.groupby('att').agg(n=('v','size'),vert_med=('v','median'),horiz_abs_med=('h',lambda x:np.median(np.abs(x))))
g2=pr.groupby('att').agg(vert_pre=('v','median'),horiz_pre=('h',lambda x:np.median(np.abs(x))))
g=g.join(g2); g['name']=g.index.map(nm); g=g[g.n>=8].sort_values('vert_med')
pd.set_option('display.width',200); print(g.round(2).to_string())
m=f[f.att==ME]
print("\nты: вертикаль на первом выстреле — ниже головы (>0.5°): %.0f%%, выше (<-0.5°): %.0f%%"%((m.v>0.5).mean()*100,(m.v<-0.5).mean()*100))
print("по дистанции:"); m=m.assign(b=pd.cut(m.dist,[0,500,1000,1600,5000]))
print(m.groupby('b',observed=True).v.agg(['size','median']).round(2))
