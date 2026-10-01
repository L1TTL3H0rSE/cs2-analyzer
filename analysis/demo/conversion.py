# Конверсия преимущества — против лобби тех же матчей:
#   1) после твоего первого фрага в раунде: сохраняешь преимущество или отдаёшь его;
#   2) воронка размена твоих смертей (некому → никто не увидел убийцу → увидели, но не попали → попали, но не убили → разменяли)
#      и ты как разменивающий;
#   3) флешки: кого ослепляют, превращаются ли в убийства; дуэли под флешку своей команды;
#   4) дуэли по дистанции (берёт кэш duels.py).
#   python conversion.py [папка с res_*.pkl]   (демки — в родительской папке; кэш conv_<матч>.pkl рядом с res_*)
# ponytail: кэш не знает о версии conv() — поменял расчёт, удали conv_*.pkl
import glob, os, sys, pickle, math
from bisect import bisect_right
from collections import defaultdict
from types import SimpleNamespace
import numpy as np, pandas as pd
from demoparser2 import DemoParser
from extract import dec
from report import match, ME, TICK, SIDE
from duels import wilson

TRADE = 5 * TICK; STEP = 4
NEAR = 1000  # юнитов: тиммейт «рядом»
BLIND = 1.0  # с: «ослеплён»
STAGES = ['разменяли', 'попали по убийце, но не убили', 'видели убийцу, но не попали',
          'тиммейт рядом, но убийцу не увидел', 'тиммейты далеко, никто не увидел', 'некому: последний живой']


def conv(name, R, dem_path):
    M = match(name, R)
    if M['short'] >= 0.2: return None
    d, h, pl, end, keep, traded = M['d'], M['h'], M['pl'], M['end'], M['keep'], M['traded']
    fe = R['freeze']; rnd = lambda t: bisect_right(fe, t)
    won = {(x['r'], x['sid']): x['won'] for x in M['rounds']}
    team = lambda r, s: pl[r].get(s)
    live = d[d.enemy.to_numpy() & np.array([t <= end.get(r, 1e12) for r, t in zip(d.r, d.tick)])]
    first = live.drop_duplicates(['r', 'att'])  # d отсортирован по тику
    ticks = {tt for t in first.tick for tt in (t, t + 3 * TICK)} | {tt for t in live.tick for tt in range(t, t + TRADE + 1, STEP)}
    dem = dec(dem_path)
    try:
        tk = DemoParser(dem).parse_ticks(['X', 'Y', 'approximate_spotted_by'], ticks=sorted(ticks))
    finally:
        if dem != dem_path: os.remove(dem)
    tk['spot'] = [frozenset(map(str, l)) if isinstance(l, (list, np.ndarray)) else frozenset() for l in tk.approximate_spotted_by]
    P = {s: g.drop_duplicates('tick').set_index('tick') for s, g in tk.assign(sid=tk.steamid.astype(str)).groupby('sid')}
    at = lambda s, t: P[s].loc[t] if s in P and t in P[s].index else None
    dist = lambda a, b: math.hypot(a.X - b.X, a.Y - b.Y) if a is not None and b is not None else np.nan
    death_t = {(r, v): t for r, v, t in zip(d.r, d.vic, d.tick)}
    dead_by = lambda r, t: {v for (rr, v), tt in death_t.items() if rr == r and tt <= t}

    # 1) после первого фрага игрока в раунде
    frag = []
    for x in first.itertuples():
        r, A, t = x.r, x.att, int(x.tick); tm = team(r, A)
        dt = (death_t.get((r, A), np.inf) - t) / TICK
        k2 = live[(live.r == r) & (live.att == A) & (live.tick > t)]
        dt2 = (k2.tick.iloc[0] - t) / TICK if len(k2) else np.inf
        a0, a3 = at(A, t), (at(A, t + 3 * TICK) if dt > 3 else None)
        vic = SimpleNamespace(X=x.user_X, Y=x.user_Y)
        moved, toward = dist(a0, a3), dist(a0, vic) - dist(a3, vic)
        dead = dead_by(r, t)
        n_my = sum(q == tm and s not in dead for s, q in pl[r].items()); n_en = sum(q != tm and s not in dead for s, q in pl[r].items())
        beh = ('умер в первые 3 с' if dt <= 3 else 'нет данных' if np.isnan(moved) else
               'пошёл вперёд' if toward > 150 else 'остался на месте' if moved < 150 else 'сменил позицию/отошёл')
        frag.append(dict(m=name, map=R['map'], r=r, P=A, keep=A in keep, side=SIDE[tm], sec=(t - fe[r - 1]) / TICK,
                         opening=t == live[live.r == r].tick.min(), adv=n_my - n_en, beh=beh, died10=dt <= 10,
                         traded=(r, A) in traded, k2=dt2 <= 10, gave=dt <= 10 and (r, A) not in traded and dt2 > dt,
                         won=won.get((r, A)), place=x.attacker_last_place_name))

    # 2) воронка размена и ты как разменивающий
    funnel, mates_rows = [], []
    for x in live.itertuples():
        r, K, V, t = x.r, x.att, x.vic, int(x.tick); tv = team(r, V)
        dead = dead_by(r, t); mates = [s for s, q in pl[r].items() if q == tv and s not in dead]
        seen = {s for tt in range(t, t + TRADE + 1, STEP) if (k := at(K, tt)) is not None for s in k.spot if s in mates}
        hk = h[(h.vic == K) & (h.tick > t) & (h.tick <= t + TRADE)]
        killers = set(d[(d.r == r) & (d.vic == K) & (d.tick > t) & (d.tick <= t + TRADE)].att)
        pv = at(V, t); near = {s: dist(at(s, t), pv) for s in mates}
        nd = [v for v in near.values() if not np.isnan(v)]
        st = ('разменяли' if (r, V) in traded else 'некому: последний живой' if not mates
              else 'попали по убийце, но не убили' if hk.att.isin(mates).any() else 'видели убийцу, но не попали' if seen
              else 'тиммейт рядом, но убийцу не увидел' if nd and min(nd) <= NEAR else 'тиммейты далеко, никто не увидел')
        funnel.append(dict(m=name, map=R['map'], r=r, V=V, K=K, keep=V in keep, side=SIDE[tv], st=st,
                           sec=(t - fe[r - 1]) / TICK, place=x.user_last_place_name, weapon=x.weapon))
        for s in mates:
            mates_rows.append(dict(m=name, P=s, keep=s in keep, near=near[s], saw=s in seen,
                                   hit=bool((hk.att == s).any()), kill=s in killers))

    # 3) флешки и дуэли под флешку
    bl = R['blind'].assign(att=R['blind'].attacker_steamid.astype(str), vic=R['blind'].user_steamid.astype(str))
    bl = bl[bl.blind_duration >= BLIND]
    fl = R['nades'].get('flashbang_detonate', pd.DataFrame(columns=['tick', 'user_steamid']))
    flashes = []
    for tf, F in zip(fl.tick, fl.user_steamid.astype(str)):
        r = rnd(tf)
        if r == 0 or F not in pl[r]: continue
        tm = pl[r][F]; b = bl[(bl.att == F) & bl.tick.between(tf - 2, tf + 2)]
        en = b[[team(r, v) not in (None, tm) for v in b.vic]]
        conv_by = [k.att for e in en.itertuples() for k in live[(live.r == r) & (live.vic == e.vic)].itertuples()
                   if e.tick <= k.tick <= e.tick + e.blind_duration * TICK and team(r, k.att) == tm]
        flashes.append(dict(m=name, P=F, keep=F in keep, en=len(en), mt=int(sum(team(r, v) == tm and v != F for v in b.vic)),
                            self_=bool((b.vic == F).any()), conv=len(conv_by) > 0, conv_self=F in conv_by))
    blinded = lambda r, s, by_team, t: bool(len(bl[(bl.vic == s) & (bl.tick <= t) & (bl.tick + bl.blind_duration * TICK >= t)
                                                  & np.array([team(r, a) == by_team for a in bl.att], dtype=bool)]))
    duel = [dict(m=name, tick=int(x.tick), K=x.att, V=x.vic, dist=x.distance,
                 k_sup=blinded(x.r, x.vic, team(x.r, x.att), x.tick), v_sup=blinded(x.r, x.att, team(x.r, x.vic), x.tick))
            for x in live.itertuples()]
    return dict(frag=pd.DataFrame(frag), funnel=pd.DataFrame(funnel), mates=pd.DataFrame(mates_rows),
                flashes=pd.DataFrame(flashes), duel=pd.DataFrame(duel))


pct = lambda s: s.astype(float).mean() * 100


def main(folder):
    C = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(folder, 'res_*.pkl'))):
        name = os.path.basename(f)[4:-4]; cache = os.path.join(folder, f'conv_{name}.pkl')
        if not os.path.exists(cache):
            x = conv(name, pickle.load(open(f, 'rb')), os.path.join(folder, '..', f'{name}.dem.zst'))
            pickle.dump(x, open(cache, 'wb')); print('разобран', name, flush=True)
        x = pickle.load(open(cache, 'rb'))
        if x is not None:
            for k, v in x.items(): C[k].append(v)
    T = {k: pd.concat(v, ignore_index=True) for k, v in C.items()}

    F = T['frag']; fm, fo = F[F.P == ME], F[(F.P != ME) & F.keep]
    print(f"ПОСЛЕ ТВОЕГО ПЕРВОГО ФРАГА В РАУНДЕ (ты: {len(fm)}, лобби: {len(fo)})")
    outs = [('раунд выигран', 'won'), ('ещё убийство за 10 с', 'k2'), ('умер в течение 10 с', 'died10'),
            ('отдал преимущество: умер за 10 с без размена и без 2-го фрага', 'gave')]
    for lab, c in outs:
        print(f"  {lab:62} ты {pct(fm[c] == True):3.0f}%   лобби {pct(fo[c] == True):3.0f}%")
    print('  что делал первые 3 с после фрага → доля; потом: отдал преимущество / 2-й фраг / раунд выигран  (ты | лобби)')
    for beh in ['остался на месте', 'сменил позицию/отошёл', 'пошёл вперёд', 'умер в первые 3 с']:
        a, b = fm[fm.beh == beh], fo[fo.beh == beh]
        print(f"    {beh:24} {len(a) / len(fm) * 100:3.0f}% | {len(b) / len(fo) * 100:3.0f}%    отдал {pct(a.gave):3.0f}% | {pct(b.gave):3.0f}%"
              f"   2-й фраг {pct(a.k2):3.0f}% | {pct(b.k2):3.0f}%   раунд {pct(a.won == True):3.0f}% | {pct(b.won == True):3.0f}%   (n={len(a)})")
    for lab, f in [('опенинг', lambda x: x.opening), ('не опенинг', lambda x: ~x.opening)]:
        a, b = fm[f(fm)], fo[f(fo)]
        print(f"    {lab:24} отдал преимущество {pct(a.gave):3.0f}% | {pct(b.gave):3.0f}%   пошёл вперёд {pct(a.beh == 'пошёл вперёд'):3.0f}% | {pct(b.beh == 'пошёл вперёд'):3.0f}%   (n={len(a)})")
    g = fm[fm.gave].assign(spot=lambda x: x['map'].str[3:] + ' ' + x.side + ' ' + x.place.astype(str)).spot.value_counts().head(5)
    print('    где чаще отдаёшь преимущество:', g.to_dict())

    U = T['funnel']; um, uo = U[U.V == ME], U[(U.V != ME) & U.keep]
    print(f"\nВОРОНКА РАЗМЕНА ТВОИХ СМЕРТЕЙ (ты: {len(um)}, лобби: {len(uo)})")
    for st in STAGES:
        print(f"  {st:40} ты {pct(um.st == st):3.0f}% ({(um.st == st).sum():3})   лобби {pct(uo.st == st):3.0f}%")
    far = um[um.st == 'тиммейты далеко, никто не увидел'].assign(spot=lambda x: x['map'].str[3:] + ' ' + x.side + ' ' + x.place.astype(str))
    print('  где ты умираешь далеко от команды:', far.spot.value_counts().head(6).to_dict())
    Mt = T['mates']; mm, mo = Mt[Mt.P == ME], Mt[(Mt.P != ME) & Mt.keep]
    print(f"\nТЫ КАК РАЗМЕНИВАЮЩИЙ (тиммейт погиб, ты жив; ты: {len(mm)}, лобби: {len(mo)})")
    for lab, f in [('все', lambda x: x.near == x.near), (f'ты был ближе {NEAR}', lambda x: x.near <= NEAR), (f'дальше {NEAR}', lambda x: x.near > NEAR)]:
        a, b = mm[f(mm)], mo[f(mo)]
        sa, sb = a[a.saw], b[b.saw]
        print(f"  {lab:18} увидел убийцу за 5 с {pct(a.saw):3.0f}% | {pct(b.saw):3.0f}%   из увиденных: попал {pct(sa.hit):3.0f}% | {pct(sb.hit):3.0f}%"
              f", разменял {pct(sa.kill):3.0f}% | {pct(sb.kill):3.0f}%   разменял всего {pct(a.kill):3.0f}% | {pct(b.kill):3.0f}%  (n={len(a)})")

    Fl = T['flashes']; lm, lo = Fl[Fl.P == ME], Fl[(Fl.P != ME) & Fl.keep]
    print(f"\nФЛЕШКИ (ты: {len(lm)}, лобби: {len(lo)}; ослеплён = {BLIND}+ с)")
    for lab, f in [('врагов ослеплено на флешку', lambda x: x.en.mean()), ('флешек, ослепивших хотя бы одного врага, %', lambda x: pct(x.en > 0)),
                   ('своих ослеплено на флешку', lambda x: x.mt.mean()), ('ослепил сам себя, %', lambda x: pct(x.self_)),
                   ('ослеплённого врага убила твоя команда, % флешек', lambda x: pct(x.conv)), ('…убил ты сам, % флешек', lambda x: pct(x.conv_self)),
                   ('из ослепивших врага — убийство, %', lambda x: pct(x[x.en > 0].conv))]:
        print(f"  {lab:50} ты {f(lm):5.2f}   лобби {f(lo):5.2f}")
    Dd = T['duel']; persp = pd.concat([Dd.assign(P=Dd.K, won=True, sup=Dd.k_sup, against=Dd.v_sup), Dd.assign(P=Dd.V, won=False, sup=Dd.v_sup, against=Dd.k_sup)])
    keep = set(zip(F.m[F.keep], F.P[F.keep])) | set(zip(U.m[U.keep], U.V[U.keep]))
    pm, po = persp[persp.P == ME], persp[(persp.P != ME) & np.array([(m, p) in keep for m, p in zip(persp.m, persp.P)], dtype=bool)]
    print('  дуэли: соперник ослеплён флешкой твоей команды / ты ослеплён их флешкой — доля дуэлей и винрейт (ты | лобби)')
    for lab, c in [('соперник ослеплён', 'sup'), ('ты ослеплён', 'against')]:
        a, b = pm[pm[c]], po[po[c]]
        lo_, hi_ = wilson(a.won.sum(), len(a))
        print(f"    {lab:20} {len(a) / len(pm) * 100:3.0f}% | {len(b) / len(po) * 100:3.0f}% дуэлей   винрейт {pct(a.won):3.0f}% [{lo_ * 100:.0f}–{hi_ * 100:.0f}] | {pct(b.won):3.0f}%  (n={len(a)})")

    dfiles = sorted(glob.glob(os.path.join(folder, 'duels_*.pkl')))
    if dfiles:
        D = pd.concat([x for f in dfiles if (x := pickle.load(open(f, 'rb'))) is not None], ignore_index=True)[['m', 'tick', 'K', 'V', 'k_grp', 'v_grp', 'keepK', 'keepV']]
        D = D.merge(Dd[['m', 'tick', 'K', 'V', 'dist']], on=['m', 'tick', 'K', 'V'])
        per = pd.concat([D.assign(P=D.K, keep=D.keepK, g=D.k_grp, og=D.v_grp, won=True), D.assign(P=D.V, keep=D.keepV, g=D.v_grp, og=D.k_grp, won=False)])
        per['bin'] = pd.cut(per.dist, [0, 5, 15, 25, 200], labels=['<5 м', '5–15 м', '15–25 м', '25+ м'])
        print('\nДУЭЛИ ПО ДИСТАНЦИИ (винрейт; ты | лобби в том же матчапе и дистанции)')
        for g_, og in [('винтовка', 'винтовка'), ('винтовка', 'снайперка'), ('винтовка', 'SMG/дробовик'), ('пистолет', 'пистолет'), ('дигл', 'винтовка')]:
            x = per[(per.g.astype(str) == g_) & (per.og.astype(str) == og)]
            a, b = x[x.P == ME], x[(x.P != ME) & x.keep]
            cells = []
            for bn in per.bin.cat.categories:
                aa, bb = a[a.bin == bn], b[b.bin == bn]
                cells.append(f"{bn}: {pct(aa.won):3.0f}% | {pct(bb.won):3.0f}% (n={len(aa)})" if len(aa) >= 5 else f"{bn}: — ")
            print(f"  {g_:9} против {og:13} " + '   '.join(cells))

    out = os.path.join(folder, 'my_conversion.csv')
    pd.concat([fm.assign(что='первый фраг'), um.assign(что='смерть')]).sort_values(['m', 'r'])[
        ['m', 'r', 'что', 'side', 'sec', 'place', 'beh', 'adv', 'k2', 'died10', 'gave', 'traded', 'won', 'st', 'weapon']].round(1).to_csv(out, index=False, encoding='utf-8-sig')
    print('\nтвои фраги и смерти с разметкой:', os.path.abspath(out))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', '..', 'demos', 'parsed'))
