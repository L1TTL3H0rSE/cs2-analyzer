# Профиль на AWP против AWP-раундов лобби: AWP-раунд = AWP в инвентаре на 10-й секунде или убийство с AWP.
#   python awp.py [папка с res_*.pkl]   (нужны кэши duels.py и conversion.py; свой кэш — awp_rounds.pkl)
import sys, glob, os, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FOLDER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'demos', 'parsed')
from extract import dec
from report import match, ME, TICK
from duels import wilson
from demoparser2 import DemoParser
cache = os.path.join(FOLDER, 'awp_rounds.pkl')
if os.path.exists(cache):
    Rd = pickle.load(open(cache, 'rb'))
else:
    rows = []
    for f in sorted(glob.glob(os.path.join(FOLDER, 'res_*.pkl'))):
        name = os.path.basename(f)[4:-4]; R = pickle.load(open(f, 'rb')); M = match(name, R)
        if M['short'] >= 0.2: continue
        fe = R['freeze']; dem = dec(os.path.join(FOLDER, '..', f'{name}.dem.zst'))
        try: inv = DemoParser(dem).parse_ticks(['inventory'], ticks=[t + 10 * TICK for t in fe])
        finally: os.remove(dem)
        has = {(fe.index(t - 10 * TICK) + 1, str(s)) for t, s, i in zip(inv.tick, inv.steamid, inv.inventory) if isinstance(i, (list, np.ndarray)) and 'AWP' in list(i)}
        d = M['d']; awpk = d[d.enemy & (d.weapon == 'awp')]
        has |= set(zip(awpk.r, awpk.att))  # подобрал AWP по ходу раунда
        for x in M['rounds']:
            x = dict(x); x['awp'] = (x['r'], x['sid']) in has; x['keep'] = x['sid'] in M['keep']
            x['awp_kills'] = int(((awpk.r == x['r']) & (awpk.att == x['sid'])).sum()); x['map'] = R['map']
            rows.append(x)
    Rd = pd.DataFrame(rows); pickle.dump(Rd, open(cache, 'wb'))
Rd = Rd[Rd.keep | (Rd.sid == ME)]
pct = lambda s: s.astype(float).mean() * 100
def prof(x):
    return (f"раундов {len(x):3}  убийств/раунд {x.kills.mean():.2f}  смертей/раунд {pct(x.died) / 100:.2f}  опенинги: участие {pct(x.op != 0):3.0f}%, "
            f"выигрыш {(x.op == 1).sum() / max((x.op != 0).sum(), 1) * 100:3.0f}%  2+ убийств {pct(x.kills >= 2):3.0f}%  раунд выигран {pct(x.won):3.0f}%  "
            f"умер первым в команде {pct(x.first_dead):3.0f}%")
me, lob = Rd[Rd.sid == ME], Rd[Rd.sid != ME]
print('ТЫ С AWP:      ', prof(me[me.awp]))
print('ТЫ БЕЗ AWP:    ', prof(me[~me.awp]))
print('ЛОББИ С AWP:   ', prof(lob[lob.awp]))
print('ЛОББИ БЕЗ AWP: ', prof(lob[~lob.awp]))
print(f"\nAWP-раундов: ты {me.awp.sum()} из {len(me)} ({pct(me.awp):.0f}%); у игроков лобби в среднем {pct(lob.awp):.0f}%; матчей, где ты брал AWP: {me[me.awp].m.nunique()} из {me.m.nunique()}")
print('по сторонам (ты с AWP):', {s: f"{len(x)} р., K/р {x.kills.mean():.2f}, D/р {pct(x.died) / 100:.2f}" for s, x in me[me.awp].groupby('side')},
      '| лобби с AWP:', {s: f"K/р {x.kills.mean():.2f}, D/р {pct(x.died) / 100:.2f}" for s, x in lob[lob.awp].groupby('side')})
print('по картам (ты с AWP):', {m[3:]: len(x) for m, x in me[me.awp].groupby('map')})

D = pd.concat([x for f in sorted(glob.glob(os.path.join(FOLDER, 'duels_*.pkl'))) if (x := pickle.load(open(f, 'rb'))) is not None], ignore_index=True)
cv = pd.concat([pickle.load(open(f, 'rb'))['duel'] for f in sorted(glob.glob(os.path.join(FOLDER, 'conv_*.pkl'))) if pickle.load(open(f, 'rb')) is not None])
D = D.merge(cv[['m', 'tick', 'K', 'V', 'dist']], on=['m', 'tick', 'K', 'V'], how='left')
P = pd.concat([D.assign(P=D.K, keep=D.keepK, g=D.k_grp, og=D.v_grp, won=True, spd=D.spd_k, ospd=D.spd_v, fs_hit=D.k_fs_hit, pre=D.k_pre),
               D.assign(P=D.V, keep=D.keepV, g=D.v_grp, og=D.k_grp, won=False, spd=D.spd_v, ospd=D.spd_k, fs_hit=D.v_fs_hit, pre=D.v_pre)])
P = P[(P.g.astype(str) == 'снайперка') & ~P.post]
a, b = P[P.P == ME], P[(P.P != ME) & P.keep]
lo, hi = wilson(a.won.sum(), len(a))
print(f"\nДУЭЛИ С СНАЙПЕРКОЙ: ты {len(a)}, винрейт {pct(a.won):.0f}% [{lo * 100:.0f}–{hi * 100:.0f}] | лобби {len(b)}, {pct(b.won):.0f}%")
for lab, f in [('ты держишь угол (стоишь, он выходит)', lambda x: (x.spd < 60) & (x.ospd > 100)), ('ты выглядываешь/двигаешься', lambda x: x.spd > 100),
               ('оба стоят', lambda x: (x.spd < 60) & (x.ospd < 60))]:
    aa, bb = a[f(a)], b[f(b)]
    print(f"  {lab:38} ты {pct(aa.won):3.0f}% (n={len(aa)})  лобби {pct(bb.won):3.0f}% (n={len(bb)})")
print(f"  первый выстрел попал: ты {pct(a[a.fs_hit.notna()].fs_hit == True):.0f}% | лобби {pct(b[b.fs_hit.notna()].fs_hit == True):.0f}%;  прицел уже на враге при появлении: ты {pct(a.pre == True):.0f}% | лобби {pct(b.pre == True):.0f}%")
for lab, lo_, hi_ in [('< 15 м', 0, 15), ('15–25 м', 15, 25), ('25+ м', 25, 999)]:
    aa, bb = a[a.dist.between(lo_, hi_, inclusive='left')], b[b.dist.between(lo_, hi_, inclusive='left')]
    print(f"  дистанция {lab:8} ты {pct(aa.won):3.0f}% (n={len(aa)})  лобби {pct(bb.won):3.0f}% (n={len(bb)})")
print('  против чего (ты):', {str(k): f"{pct(x.won):.0f}% (n={len(x)})" for k, x in a.groupby(a.og.astype(str))})
