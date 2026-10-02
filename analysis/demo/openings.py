# Опенинги (первая дуэль раунда) за сторону — ты против лобби: кто выглядывал, флешки, тиммейт рядом, кто увидел
# первым, прицел при появлении, секунда раунда, закуп, точки; что было после проигранного опенинга.
#   python openings.py [T|CT]   (нужны таблицы extract.py и кэши duels.py, conversion.py)
import glob, os, pickle, sys
from collections import Counter
import numpy as np, pandas as pd
from report import match, ME, TICK, SIDE
from duels import annotate, MOVE, HOLD, SAW, wilson

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'demos', 'parsed')
SIDE_ = (sys.argv[1] if len(sys.argv) > 1 else 'T').upper()
NEAR = 800  # юнитов: тиммейт «рядом» — успевает разменять

D, _, _, _ = annotate(pd.concat([x for f in sorted(glob.glob(os.path.join(FOLDER, 'duels_*.pkl'))) if (x := pickle.load(open(f, 'rb'))) is not None], ignore_index=True))
CV = pd.concat([x['duel'] for f in sorted(glob.glob(os.path.join(FOLDER, 'conv_*.pkl'))) if (x := pickle.load(open(f, 'rb'))) is not None])
D = D.merge(CV[['m', 'tick', 'K', 'V', 'k_sup', 'v_sup']], on=['m', 'tick', 'K', 'V'], how='left')

rows, n_rounds = [], Counter()
for f in sorted(glob.glob(os.path.join(FOLDER, 'res_*.pkl'))):
    m = os.path.basename(f)[4:-4]; R = pickle.load(open(f, 'rb')); M = match(m, R)
    if M['short'] >= 0.2: continue
    n_rounds.update((x['side'], x['sid'] == ME) for x in M['rounds'] if x['sid'] in M['keep'] or x['sid'] == ME)
    d, pl, end, fe = M['d'], M['pl'], M['end'], R['freeze']
    ctx = {(x['r'], x['sid']): x for x in M['rounds']}
    live = d[d.enemy.to_numpy() & np.array([t <= end.get(r, 1e12) for r, t in zip(d.r, d.tick)])]
    fl = R['nades'].get('flashbang_detonate', pd.DataFrame(columns=['tick', 'user_steamid'])).assign(sid=lambda x: x.user_steamid.astype(str))
    pos = R['pos'].assign(sid=R['pos'].steamid.astype(str))
    Dm = D[D.m == m].set_index(['tick', 'K', 'V'])
    for x in live.drop_duplicates('r').itertuples():
        r, t, K, V = x.r, int(x.tick), x.att, x.vic
        if (t, K, V) not in Dm.index: continue
        du = Dm.loc[(t, K, V)]; du = du.iloc[0] if isinstance(du, pd.DataFrame) else du
        s = pos[(pos.tick <= t) & (pos.tick > fe[r - 1])]
        snap = s[s.tick == s.tick.max()] if len(s) else s
        dead = set(d[(d.r == r) & (d.tick < t)].vic)
        for P, O, won in ((K, V, True), (V, K, False)):
            side = SIDE[pl[r][P]]; c = ctx.get((r, P), {})
            me_xy = snap[snap.sid == P]
            mates = snap[snap.sid.isin([q for q, tm in pl[r].items() if tm == pl[r][P] and q != P and q not in dead])]
            near = (np.hypot(mates.X.to_numpy() - me_xy.X.iloc[0], mates.Y.to_numpy() - me_xy.Y.iloc[0]).min()
                    if len(me_xy) and len(mates) else np.nan)
            spd, ospd = (du.spd_k, du.spd_v) if won else (du.spd_v, du.spd_k)
            adv = du.saw_adv if won else -du.saw_adv
            rows.append(dict(
                m=m, map=R['map'], r=r, sec=(t - fe[r - 1]) / TICK, P=P, keep=P in M['keep'], side=side, won=won,
                place=x.attacker_last_place_name if won else x.user_last_place_name,
                op_place=x.user_last_place_name if won else x.attacker_last_place_name, weapon=x.weapon, tick=t,
                role='ты выглядывал на стоящего' if spd > MOVE and ospd < HOLD else 'выглянул он, ты стоял' if ospd > MOVE and spd < HOLD
                else 'оба двигались' if spd > MOVE and ospd > MOVE else 'оба стояли' if spd < HOLD and ospd < HOLD else 'неясно',
                saw='ты увидел раньше' if adv > SAW else 'тебя увидели раньше' if adv < -SAW else 'одновременно' if adv == adv else 'нет данных',
                pre=(du.k_pre if won else du.v_pre) == True,
                sup=bool(du.k_sup if won else du.v_sup) if (du.k_sup == du.k_sup) else False,
                own_flash=bool(((fl.sid == P) & fl.tick.between(t - 3 * TICK, t)).any()),
                near=near, buy=c.get('my_buy'), round_won=c.get('won'), traded=(r, P) in M['traded'],
                cat=None if won else du['cat']))
O = pd.DataFrame(rows)
O['near_b'] = np.where(O.near.isna(), 'нет данных', np.where(O.near <= NEAR, f'тиммейт ближе {NEAR}', f'тиммейт дальше {NEAR}'))
O['time'] = pd.cut(O.sec, [-1, 20, 40, 200], labels=['0–20 с', '20–40 с', '40+ с']).astype(str)
O['flash'] = np.where(O.sup, 'враг ослеплён флешкой твоей команды', np.where(O.own_flash, 'ты кинул флешку, но врага не ослепило', 'без флешки'))

me, lob = O[O.P == ME], O[(O.P != ME) & O.keep]
for side in ('T', 'CT'):
    a, b = me[me.side == side], lob[lob.side == side]
    print(f"{side}: опенингов ты {len(a)} ({len(a) / n_rounds[(side, True)] * 100:.0f}% твоих раундов), выиграл {a.won.mean() * 100:.0f}% [{wilson(a.won.sum(), len(a))[0] * 100:.0f}–{wilson(a.won.sum(), len(a))[1] * 100:.0f}]"
          f"  | лобби {len(b) / n_rounds[(side, False)] * 100:.0f}% раундов, выигрывает {b.won.mean() * 100:.0f}%")
a, b = me[me.side == SIDE_], lob[lob.side == SIDE_]
print(f"\nОПЕНИНГИ ЗА {SIDE_}: доля таких опенингов и винрейт (ты | лобби)")
for col, lab in (('role', 'кто выглядывал'), ('flash', 'флешка'), ('near_b', 'тиммейт для размена (по позиции ≤5 с до дуэли)'),
                 ('saw', 'кто увидел первым'), ('pre', 'прицел уже на враге при появлении'), ('time', 'секунда раунда'), ('buy', 'закуп команды')):
    print(f"  {lab}:")
    for v in sorted(set(a[col].dropna()) | set(b[col].dropna()), key=str):
        x, y = a[a[col] == v], b[b[col] == v]
        if len(x) + len(y) == 0: continue
        lo, hi = wilson(x.won.sum(), len(x))
        print(f"    {str(v):44} ты {len(x) / len(a) * 100:3.0f}% опенингов, винрейт {x.won.mean() * 100 if len(x) else float('nan'):3.0f}% [{lo * 100:.0f}–{hi * 100:.0f}] (n={len(x):2})"
              f" | лобби {len(y) / len(b) * 100:3.0f}%, винрейт {y.won.mean() * 100:3.0f}%")
print('\n  раунд выигран командой: опенинг выигран / проигран (ты | лобби):',
      f"{a[a.won].round_won.mean() * 100:.0f}% / {a[~a.won].round_won.mean() * 100:.0f}% | {b[b.won].round_won.mean() * 100:.0f}% / {b[~b.won].round_won.mean() * 100:.0f}%")
la, lb = a[~a.won], b[~b.won]
print(f"  проигранные опенинги: разменяли {la.traded.mean() * 100:.0f}% | лобби {lb.traded.mean() * 100:.0f}%;  причины:",
      {k: f"{v / len(la) * 100:.0f}%" for k, v in Counter(la.cat).most_common()}, '| лобби', {k: f"{v / len(lb) * 100:.0f}%" for k, v in Counter(lb.cat).most_common(4)})
out = os.path.join(FOLDER, 'my_openings.csv')
me.sort_values(['m', 'r'])[['m', 'r', 'sec', 'side', 'won', 'place', 'op_place', 'weapon', 'role', 'flash', 'near_b', 'saw', 'pre', 'buy', 'round_won', 'traded', 'cat', 'tick']].round(1) \
    .to_csv(out, index=False, encoding='utf-8-sig')
print(f"\n  все твои опенинги (обе стороны): {os.path.abspath(out)}")
print(f"\n  точки твоих опенингов за {SIDE_} (карта, место: выиграно–проиграно, медиана секунды):")
for (mp, pl_), x in sorted(a.groupby(['map', 'place']), key=lambda kv: -len(kv[1]))[:12]:
    print(f"    {mp[3:]:8} {pl_:16} {int(x.won.sum())}–{int((~x.won).sum())}   {x.sec.median():.0f} с   флешка {x.sup.mean() * 100:.0f}%, тиммейт рядом {(x.near <= NEAR).mean() * 100:.0f}%")
