# Прогресс по фокусмапу: целевые метрики до и после даты (по умолчанию 2026-10-01 — день фокусмапа).
#   python progress.py [YYYY-MM-DD]   (нужны таблицы extract.py и кэши duels.py)
import glob, os, pickle, sys
import numpy as np, pandas as pd
from duels import ME, MOVE, HOLD, wilson

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'demos', 'parsed')
SINCE = sys.argv[1] if len(sys.argv) > 1 else '2026-10-01'
RIFLES = {'ak47', 'm4a1_silencer', 'm4a1', 'galilar', 'famas', 'aug', 'sg556'}
WEAK = {('de_anubis', 'CT', 'Canal'), ('de_anubis', 'CT', 'Middle'), ('de_ancient', 'CT', 'Middle'),
        ('de_dust2', 'T', 'LongA'), ('de_anubis', 'T', 'OutsideLong')}

D = pd.concat([x for f in sorted(glob.glob(os.path.join(FOLDER, 'duels_*.pkl'))) if (x := pickle.load(open(f, 'rb'))) is not None], ignore_index=True)
D = D[~D.post]
P = pd.concat([D.assign(P=D.K, won=True, spd=D.spd_k, ospd=D.spd_v, pre=D.k_pre, off0=D.k_off0, side=D.side_k, place=D.place_k),
               D.assign(P=D.V, won=False, spd=D.spd_v, ospd=D.spd_k, pre=D.v_pre, off0=D.v_off0, side=D.side_v, place=D.place_v)])
me = P[P.P == ME].assign(after=lambda x: x.m.str[:10] >= SINCE)
aim = []
for f in sorted(glob.glob(os.path.join(FOLDER, 'res_*.pkl'))):
    m = os.path.basename(f)[4:-4]
    if m not in set(D.m): continue  # исключённые матчи (4v5 с ботом)
    a = pickle.load(open(f, 'rb'))['aim']
    aim.append(a[(a.att.astype(str) == ME) & (a.when == 'first') & a.weapon.isin(list(RIFLES))].assign(after=m[:10] >= SINCE))
A = pd.concat(aim)
myd = D[D.V == ME].assign(after=lambda x: x.m.str[:10] >= SINCE)


def share(x, mask):
    k, n = int(mask.sum()), len(x); lo, hi = wilson(k, n)
    return f"{k / max(n, 1) * 100:3.0f}% [{lo * 100:.0f}–{hi * 100:.0f}] (n={n})"


print(f"ФОКУСМАП: до {SINCE} — {me[~me.after].m.nunique()} матчей, после — {me[me.after].m.nunique()} матчей")
for lab, after in (('до', False), ('после', True)):
    x = me[me.after == after]; mv = x[(x.spd > MOVE) & (x.off0 < 15)]
    held, peek = (x.spd < HOLD) & (x.ospd > MOVE), (x.spd > MOVE) & (x.ospd < HOLD)
    a = A[A.after == after]; d = myd[myd.after == after]; u = d[~d.traded]
    weak = x[[(m, s, p) in WEAK for m, s, p in zip(x['map'], x.side, x.place)]]
    print(f"\n{lab.upper()}:")
    print(f"  1) в дуэлях на ходу прицел уже на враге        {share(mv, mv.pre == True)}   винрейт таких дуэлей {mv.won.mean() * 100:.0f}%  (цель 55%+)")
    print(f"     1-й выстрел ниже головы (рифлы), медиана    {a.v.median():.2f}° (n={len(a)})")
    print(f"  2) смертей без размена с 2+ гранатами          {share(u, u.nades_v >= 2)}  (цель ≤30%)")
    print(f"     слабые точки: убийств {int(weak.won.sum())} / смертей {int((~weak.won).sum())}")
    print(f"  3) дуэлей, где ты держал угол                  {share(x, held)}   где выглядывал на держащего {share(x, peek)}")
    print(f"     винрейт: держал {x[held].won.mean() * 100:.0f}%, выглядывал {x[peek].won.mean() * 100:.0f}%;  дуэлей всего {len(x)}, винрейт {x.won.mean() * 100:.0f}%")
