import pickle, glob, pandas as pd, numpy as np
from collections import Counter, defaultdict
ME='76561198105241814'
RIFLES={'ak47','m4a1_silencer','m4a1','galilar','famas','aug','sg556','m4a4'}
res={f[4:-4]:pickle.load(open(f,'rb')) for f in sorted(glob.glob('res_*.pkl'))}
allaim=[]; summary=[]; deaths_me=[]; untr=Counter(); tot=Counter(); op=Counter(); opw=Counter()
eco_rows=[]; clutch=[]; nade=Counter(); nade_lobby=Counter(); rounds_me=0; rounds_all=0; codes=Counter(); kills_w=Counter(); hs_w=Counter()
for name,R in res.items():
    fe=R['freeze']; eco=R['eco']; eco['steamid']=eco.steamid.astype(str)
    fin=R['final']; fin['steamid']=fin.steamid.astype(str)
    myteam_by_round={}
    for i,t in enumerate(fe):
        e=eco[eco.tick==t]; m=e[e.steamid==ME]
        if len(m): myteam_by_round[i+1]=int(m.team_num.iloc[0])
    # счёт и победитель раунда
    def score_at(df,team):
        x=df[df.team_num==team].team_rounds_total
        return int(x.iloc[0]) if len(x) else None
    winners={}
    for i in range(len(fe)):
        cur=eco[eco.tick==fe[i]]; nxt=eco[eco.tick==fe[i+1]] if i+1<len(fe) else fin
        for team in (2,3):
            a,b=score_at(cur,team),score_at(nxt,team)
            if a is not None and b is not None and b>a: winners[i+1]=team
    mt=myteam_by_round.get(len(fe),myteam_by_round.get(1))
    my_final=score_at(fin,mt); op_final=score_at(fin,5-mt)
    d=R['deaths'].copy(); d['round']=d.tick.apply(lambda t:sum(1 for x in fe if x<=t)); d=d[d['round']>0]
    codes[str(eco[eco.steamid==ME].crosshair_code.iloc[0])]+=1
    k=d[d.attacker_steamid==ME]; dd=d[d.user_steamid==ME]
    for w,h in zip(k.weapon,k.headshot): kills_w[w]+=1; hs_w[w]+=int(h)
    summary.append((name,R['map'][3:],f"{my_final}:{op_final}",len(k),len(dd)))
    # размены и опенинги для всех игроков
    for r in d.itertuples():
        after=d[(d.tick>r.tick)&(d.tick<=r.tick+320)&(d.user_steamid==r.attacker_steamid)]
        who='me' if r.user_steamid==ME else 'other'
        tot[who]+=1; untr[who]+= 0 if len(after) else 1
    first=d.sort_values('tick').groupby('round').head(1)
    for r in first.itertuples():
        for sid,won in ((r.attacker_steamid,1),(r.user_steamid,0)):
            who='me' if sid==ME else 'other'; op[who]+=1; opw[who]+=won
    # экономика раунда: средняя стоимость снаряжения команды на freeze_end
    for i,t in enumerate(fe):
        e=eco[eco.tick==t]; rnd=i+1
        if rnd not in myteam_by_round or rnd not in winners: continue
        team=myteam_by_round[rnd]; eq=e[e.team_num==team].current_equip_value.mean(); eqo=e[e.team_num==5-team].current_equip_value.mean()
        typ='пистолетка' if rnd in (1,13) else ('эко' if eq<1500 else ('форс/полу' if eq<3500 else 'полная'))
        mk=len(d[(d['round']==rnd)&(d.attacker_steamid==ME)])
        eco_rows.append(dict(map=R['map'],rnd=rnd,side='CT' if team==3 else 'T',typ=typ,win=winners[rnd]==team,eq=eq,eqo=eqo,mykills=mk))
    # клатчи: остался последним живым в команде
    for rnd in sorted(set(d['round'])):
        if rnd not in myteam_by_round or rnd not in winners: continue
        team=myteam_by_round[rnd]; dr=d[d['round']==rnd].sort_values('tick')
        mates=set(eco[(eco.tick==fe[rnd-1])&(eco.team_num==team)].steamid)-{ME}; opps=set(eco[(eco.tick==fe[rnd-1])&(eco.team_num==5-team)].steamid)
        dead=set()
        for r in dr.itertuples():
            dead.add(r.user_steamid)
            if mates and mates<=dead and ME not in dead:
                clutch.append(dict(vs=len(opps-dead),win=winners[rnd]==team,side='CT' if team==3 else 'T')); break
    # гранаты
    for ev,df in R['nades'].items():
        for r in df.itertuples():
            if str(r.user_steamid)==ME: nade[ev]+=1
            else: nade_lobby[ev]+=1
    rounds_me+=len(fe); rounds_all+=len(fe)
    a=R['aim'].copy(); a['map']=R['map']; allaim.append(a)
A=pd.concat(allaim); A['att']=A.att.astype(str)
print("МАТЧИ:"); [print(' ',s) for s in summary]
print("\nКОД ПРИЦЕЛА:",dict(codes))
print("\nУБИЙСТВА ПО ОРУЖИЮ (hs):", {w:f"{kills_w[w]} ({hs_w[w]})" for w,_ in kills_w.most_common(8)})
print(f"\nСМЕРТИ БЕЗ РАЗМЕНА: ты {untr['me']}/{tot['me']} = {untr['me']/tot['me']*100:.0f}%   лобби {untr['other']/tot['other']*100:.0f}%")
print(f"ОПЕНИНГИ: ты участвуешь в {op['me']} ({op['me']/sum(len(R['freeze']) for R in res.values())*100:.0f}% раундов), выигрываешь {opw['me']/op['me']*100:.0f}%; средний игрок лобби: участие {op['other']/9/sum(len(R['freeze']) for R in res.values())*100:.0f}%, выигрыш 50%")
f=A[(A.when=='first')&A.weapon.isin(RIFLES)]; pre=A[(A.when=='pre')&A.weapon.isin(RIFLES)]
g=f.groupby('att').agg(n=('v','size'),vert=('v','median'),horiz=('h',lambda x:np.median(np.abs(x))),speed=('speed','median'),stopped=('speed',lambda x:(x<40).mean()*100))
gp=pre.groupby('att').v.median().rename('vert_pre'); g=g.join(gp); g=g[g.n>=15]
me=g.loc[ME]
print(f"\nПРИЦЕЛ (рифлы, игроки с 15+ дуэлями: {len(g)}):")
for c,lab,low in [('vert','ниже головы на 1-м выстреле, °',True),('vert_pre','ниже головы за 0.3с, °',True),('horiz','ошибка по горизонтали, °',True),('speed','скорость на 1-м выстреле, u/s',True),('stopped','доля выстрелов почти с места (<40 u/s), %',False)]:
    pct=(g[c]>me[c]).mean()*100 if low else (g[c]<me[c]).mean()*100
    print(f"  {lab:42} ты {me[c]:6.2f}   медиана игроков {g[c].median():6.2f}   ты лучше {pct:3.0f}%")
mf=f[f.att==ME]
print("  по картам:", mf.groupby('map').v.agg(['size','median']).round(2).to_dict('index'))
print("  по дистанции:", mf.assign(b=pd.cut(mf.dist,[0,500,1000,1500,5000])).groupby('b',observed=True).v.agg(['size','median']).round(2).to_dict('index'))
print("  хитгруппа первого попадания (1=голова,2=грудь,3=живот):", Counter(mf.hitgroup).most_common(5), " у лобби:", {k:round(v*100) for k,v in f[f.att!=ME].hitgroup.value_counts(normalize=True).head(4).items()})
E=pd.DataFrame(eco_rows)
print("\nЭКОНОМИКА (раунды твоей команды):")
print(E.groupby(['side','typ']).agg(n=('win','size'),winrate=('win','mean'),твои_убийства=('mykills','mean')).round(2))
C=pd.DataFrame(clutch)
print("\nКЛАТЧИ:", C.groupby('vs').win.agg(['size','sum']).to_dict('index') if len(C) else 'нет')
print("\nГРАНАТЫ на 24 раунда (ты | средний игрок лобби):")
for ev in ['flashbang_detonate','hegrenade_detonate','smokegrenade_detonate','inferno_startburn']:
    print(f"  {ev:22} {nade[ev]/rounds_me*24:5.1f} | {nade_lobby[ev]/9/rounds_all*24:5.1f}")
