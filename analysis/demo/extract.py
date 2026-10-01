import sys, os, json, math, pickle, traceback
import pandas as pd, numpy as np
from demoparser2 import DemoParser
ME='76561198105241814'
RIFLES={'ak47','m4a1_silencer','m4a1','galilar','famas','aug','sg556','m4a4'}
def dec(fn):
    # .zst → временный .dem; исходник остаётся (демки храним сжатыми)
    if not fn.endswith('.zst'): return fn
    import zstandard as z, tempfile
    fd,out=tempfile.mkstemp(suffix='.dem')
    with open(fn,'rb') as i, os.fdopen(fd,'wb') as o: z.ZstdDecompressor().copy_stream(i,o)
    return out
def S(df,cols):
    for c in cols:
        if c in df: df[c]=df[c].astype(str).str.replace(r'\.0$','',regex=True)
    return df
def run(fn):
    p=DemoParser(fn); R={}
    R['map']=p.parse_header()['map_name']
    start=int(p.parse_event('begin_new_match')['tick'].max())
    fe=p.parse_event('round_freeze_end'); fe=fe[fe.tick>=start].tick.astype(int).tolist(); R['freeze']=fe
    roe=p.parse_event('round_officially_ended'); R['round_end']=roe[roe.tick>start].tick.astype(int).tolist()
    d=p.parse_event('player_death',player=['X','Y','Z','last_place_name','team_num'],other=['total_rounds_played'])
    d=S(d[d.tick>start].copy(),['user_steamid','attacker_steamid','assister_steamid']); R['deaths']=d
    info=S(p.parse_player_info(),['steamid']); R['players']=info
    # команда игроков и экономика на каждом freeze_end
    eco=S(p.parse_ticks(['team_num','balance','current_equip_value','crosshair_code','team_rounds_total'],ticks=fe),['steamid']); R['eco']=eco
    # итоговый счёт — на последнем round_end: round_officially_ended после финального раунда не приходит
    last=int(p.parse_event('round_end').tick.max())
    R['final']=S(p.parse_ticks(['team_num','team_rounds_total'],ticks=[last]),['steamid'])
    # гранаты и ослепления
    g={}
    for ev in ['flashbang_detonate','hegrenade_detonate','smokegrenade_detonate','inferno_startburn']:
        try: e=p.parse_event(ev,player=['team_num']); g[ev]=S(e[e.tick>start].copy(),['user_steamid'])
        except Exception: pass
    R['nades']=g
    b=p.parse_event('player_blind',player=['team_num']); R['blind']=S(b[b.tick>start].copy(),['user_steamid','attacker_steamid'])
    # дуэли и прицел на первом выстреле
    wf=S(p.parse_event('weapon_fire'),['user_steamid']); wf=wf[wf.tick>start]; wf['weapon']=wf.weapon.str.replace('weapon_','')
    hu=S(p.parse_event('player_hurt',player=['team_num']),['attacker_steamid','user_steamid']); hu=hu[hu.tick>start]
    # для report.py: урон, победители раундов, выстрелы, бомба, позиции каждые 5 с первой минуты раунда
    R['hurt']=hu[['tick','attacker_steamid','user_steamid','attacker_team_num','user_team_num','weapon','dmg_health','health','hitgroup']].copy()
    re_=p.parse_event('round_end'); R['rounds']=re_[re_.tick>start][['tick','winner','reason']].copy()
    R['shots']=wf.groupby(['user_steamid','weapon']).size().rename('n').reset_index()
    R['bomb']={}
    for ev in ['bomb_planted','bomb_defused']:
        try: e=p.parse_event(ev); R['bomb'][ev]=S(e[e.tick>start].copy(),['user_steamid'])
        except Exception: pass
    R['pos']=S(p.parse_ticks(['X','Y','Z','last_place_name','team_num','health'],ticks=[f+s*64 for f in fe for s in range(5,65,5)]),['steamid'])
    hu=hu[hu.weapon.isin(RIFLES|{'deagle','usp_silencer','glock','p250','hkp2000','tec9','fiveseven','cz75a','mac10','mp9','awp','ssg08'})]
    eng=[]; lastk={}
    for h in hu.sort_values('tick').itertuples():
        key=(h.attacker_steamid,h.user_steamid)
        if key in lastk and h.tick-lastk[key]<3*64: lastk[key]=h.tick; continue
        lastk[key]=h.tick
        shots=wf[(wf.user_steamid==h.attacker_steamid)&(wf.tick<=h.tick)&(wf.tick>=h.tick-2*64)].tick.sort_values().tolist()
        if not shots: continue
        s=shots[-1]
        for t in reversed(shots[:-1]):
            if s-t<0.6*64: s=t
            else: break
        eng.append(dict(att=h.attacker_steamid,vic=h.user_steamid,first=int(s),weapon=h.weapon,hitgroup=h.hitgroup,hit_tick=int(h.tick)))
    E=pd.DataFrame(eng)
    ticks=sorted(set(E['first'])|set(E['first']-1)|set(E['first']-19))
    tk=S(p.parse_ticks(['X','Y','Z','pitch','yaw','duck_amount'],ticks=ticks),['steamid']).set_index(['tick','steamid'])
    rows=[]
    for e in E.itertuples():
        for when,t in [('first',e.first),('pre',e.first-19)]:
            try: a=tk.loc[(t,e.att)]; v=tk.loc[(t,e.vic)]
            except KeyError: continue
            ea=64-18*a.duck_amount; ev=64-18*v.duck_amount-2
            dx,dy,dz=v.X-a.X,v.Y-a.Y,(v.Z+ev)-(a.Z+ea); hz=math.hypot(dx,dy)
            if hz<50: continue
            ep=a.pitch+math.degrees(math.atan2(dz,hz)); ey=(a.yaw-math.degrees(math.atan2(dy,dx))+180)%360-180
            spd=None
            if when=='first':
                try: a0=tk.loc[(t-1,e.att)]; spd=math.hypot(a.X-a0.X,a.Y-a0.Y)*64
                except KeyError: pass
            rows.append(dict(att=e.att,vic=e.vic,when=when,v=ep,h=ey,dist=hz,weapon=e.weapon,hitgroup=e.hitgroup,speed=spd,tick=t))
    R['aim']=pd.DataFrame(rows)
    return R
if __name__=='__main__':
    # python extract.py <out_dir> <demo.dem[.zst]>... — уже разобранные пропускает
    out_dir=sys.argv[1]; os.makedirs(out_dir,exist_ok=True)
    for fn in sys.argv[2:]:
        name=os.path.basename(fn).split('.')[0]; res=os.path.join(out_dir,f'res_{name}.pkl')
        if os.path.exists(res): continue
        dem=dec(fn)
        try:
            R=run(dem); pickle.dump(R,open(res,'wb')); print(name,'OK',R['map'],flush=True)
        except Exception as ex:
            traceback.print_exc(); print(name,'FAIL',ex,flush=True)
        finally:
            if dem!=fn: os.remove(dem)
