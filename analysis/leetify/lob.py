import json, glob, statistics as st
from collections import Counter, defaultdict
ME='76561198105241814'
M=[json.load(open(f)) for f in glob.glob('m/*.json')]
MODE={'matchmaking':'Premier','matchmaking_competitive':'Competitive','faceit':'FACEIT'}
rows=[]; matches=[]
for m in M:
    sc=[t['score'] for t in m['team_scores']]; tot=sum(sc)
    complete = max(sc)>=13 or sorted(sc)==[12,12]
    me=[s for s in m['stats'] if s['steam64_id']==ME]
    myteam=me[0]['initial_team_number'] if me else None
    matches.append(dict(id=m['id'],mode=MODE.get(m['data_source'],m['data_source']),complete=complete,banned=m['has_banned_player'],n=len(m['stats']),t=m['finished_at'],map=m['map_name'],sc=sc,me=me[0] if me else None,myteam=myteam))
    for s in m['stats']:
        full = complete and s['rounds_count']>=0.9*tot
        rows.append(dict(mid=m['id'],mode=MODE.get(m['data_source'],m['data_source']),me=s['steam64_id']==ME,team=('me' if s['steam64_id']==ME else ('mate' if s['initial_team_number']==myteam else 'opp')),full=full,banned_match=m['has_banned_player'],complete=complete,s=s))
json.dump([dict(r, s=r['s']) for r in rows],open('rows.json','w'))
print("матчей:",len(M), Counter(x['mode'] for x in matches))
print("неполных:",sum(not x['complete'] for x in matches),"| с забаненным игроком:",sum(x['banned'] for x in matches))
print("строк игроков:",len(rows),"полных:",sum(r['full'] for r in rows),"уникальных игроков:",len({r['s']['steam64_id'] for r in rows}))
print("игроков в матче:",Counter(x['n'] for x in matches))
