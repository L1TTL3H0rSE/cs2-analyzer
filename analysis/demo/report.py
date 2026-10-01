# Подробный разбор своей игры по таблицам extract.py.
#   python report.py [папка с res_*.pkl]   (по умолчанию ../../demos/parsed)
# Лобби — все остальные игроки этих матчей, отыгравшие >=90% раундов. CI — bootstrap по матчам.
import pickle, glob, os, sys
from bisect import bisect_right
from collections import Counter, defaultdict
import numpy as np, pandas as pd

ME = '76561198105241814'
TICK = 64; TRADE = 5 * TICK
RIFLES = {'ak47', 'm4a1_silencer', 'm4a1', 'galilar', 'famas', 'aug', 'sg556'}
SIDE = {2: 'T', 3: 'CT'}
rng = np.random.default_rng(0)


def buy(v, r):
    if r in (1, 13): return 'пистолетка'
    return 'эко' if v < 1500 else 'форс' if v < 3500 else 'полная'


def match(name, R):
    fe = R['freeze']; n = len(fe)
    rnd = lambda t: bisect_right(fe, t)
    eco = R['eco']; ri = {t: i + 1 for i, t in enumerate(fe)}
    pl = defaultdict(dict); eq = {}
    for t, s, tm, v in zip(eco.tick, eco.steamid.astype(str), eco.team_num, eco.current_equip_value):
        if tm in (2, 3): pl[ri[t]][s] = int(tm); eq[(ri[t], s)] = v
    rr = R['rounds'].assign(r=R['rounds'].tick.map(rnd)); rr = rr[rr.r > 0].drop_duplicates('r', keep='last')
    win = {r: 3 if str(w) in ('CT', '3') else 2 for r, w in zip(rr.r, rr.winner)}
    end = dict(zip(rr.r, rr.tick))
    team = lambda r, s: pl[r].get(s)
    present = Counter(s for r in pl for s in pl[r])
    keep = {s for s, c in present.items() if c >= 0.9 * n}

    d = R['deaths'].copy(); d['r'] = d.tick.map(rnd); d = d[d.r > 0].sort_values('tick')
    d['att'] = d.attacker_steamid.astype(str); d['vic'] = d.user_steamid.astype(str); d['ast'] = d.assister_steamid.astype(str)
    # после своей смерти игрок может управлять ботом, и события бота пишутся на его steamid — отбрасываем их
    # (кроме урона от гранаты/молотова, брошенных при жизни)
    fd = {}
    for r, v, t in zip(d.r, d.vic, d.tick): fd.setdefault((r, v), t)
    alive_then = lambda r, s, t, w: t <= fd.get((r, s), 1e12) or w in ('inferno', 'hegrenade')
    d = d[[t <= fd[(r, v)] and alive_then(r, a, t, w) for r, v, a, t, w in zip(d.r, d.vic, d.att, d.tick, d.weapon)]]
    assert d.groupby(['r', 'vic']).size().max() == 1, f'{name}: две смерти игрока за раунд'
    d['enemy'] = [team(r, a) is not None and team(r, v) is not None and team(r, a) != team(r, v) for r, a, v in zip(d.r, d.att, d.vic)]
    h = R['hurt'].copy(); h['r'] = h.tick.map(rnd); h = h[h.r > 0].sort_values('tick')
    h['att'] = h.attacker_steamid.astype(str); h['vic'] = h.user_steamid.astype(str)
    h['real'] = (h.groupby(['r', 'vic']).health.shift(1).fillna(100) - h.health).clip(lower=0)  # урон с учётом остатка HP
    h = h[[t <= fd.get((r, v), 1e12) and alive_then(r, a, t, w) for r, v, a, t, w in zip(h.r, h.vic, h.att, h.tick, h.weapon)]]
    h['enemy'] = [team(r, a) is not None and team(r, v) is not None and team(r, a) != team(r, v) for r, a, v in zip(h.r, h.att, h.vic)]
    he = h[h.enemy]
    bl = R['blind'].copy(); bl['att'] = bl.attacker_steamid.astype(str); bl['vic'] = bl.user_steamid.astype(str); bl['r'] = bl.tick.map(rnd)

    S = defaultdict(Counter)  # статистика игрока за матч
    rounds = []; clutches = []; deaths_me = []; traded_all = set()
    for r in range(1, n + 1):
        for s, tm in pl[r].items():
            S[s]['rounds'] += 1; S[s]['ct_rounds' if tm == 3 else 't_rounds'] += 1
        dr = d[d.r == r]; dre = dr[dr.enemy]; live = dre[dre.tick <= end.get(r, 1e12)]
        kills = Counter(dre.att); died = set(dr.vic); assist = set(dre.ast)
        traded = set(); tradek = Counter()
        for x in dre.itertuples():
            after = dre[(dre.tick > x.tick) & (dre.tick <= x.tick + TRADE) & (dre.vic == x.att)]
            if len(after): traded.add(x.vic)
            before = dre[(dre.tick < x.tick) & (dre.tick >= x.tick - TRADE) & (dre.att == x.vic)]
            if any(team(r, v) == team(r, x.att) for v in before.vic): tradek[x.att] += 1
        traded_all |= {(r, v) for v in traded}
        op = live.iloc[0] if len(live) else None
        hr = he[(he.r == r) & (he.tick <= end.get(r, 1e12))]
        first = {}
        if len(hr): f = hr.iloc[0]; first = {team(r, f.att): f.att, team(r, f.vic): f.vic}
        first_dead = {}
        for x in live.itertuples(): first_dead.setdefault(team(r, x.vic), x.vic)
        # клатчи: последний живой в команде против >=1 врага
        alive = {2: set(), 3: set()}
        for s, tm in pl[r].items(): alive[tm].add(s)
        done = set()
        for x in dr[dr.tick <= end.get(r, 1e12)].itertuples():
            if team(r, x.vic): alive[team(r, x.vic)].discard(x.vic)
            for tm in (2, 3):
                if tm not in done and len(alive[tm]) == 1 and alive[5 - tm]:
                    s = next(iter(alive[tm])); done.add(tm)
                    clutches.append(dict(sid=s, vs=len(alive[5 - tm]), won=win.get(r) == tm))
        tb = {tm: np.mean([eq[(r, s)] for s, t in pl[r].items() if t == tm]) for tm in (2, 3)}
        for s, tm in pl[r].items():
            c = S[s]; k = kills[s]
            c['kills'] += k; c['deaths'] += s in died; c['assists'] += s in assist and k == 0
            c['kast'] += bool(k or s in assist or s not in died or s in traded)
            c['surv'] += s not in died; c['tradek'] += tradek[s]; c['traded'] += s in traded
            c['multi2'] += k >= 2; c['first_contact'] += first.get(tm) == s
            if op is not None and s in (op.att, op.vic): c['op_att'] += 1; c['op_win'] += op.att == s
            rounds.append(dict(m=name, r=r, sid=s, side=SIDE[tm], won=win.get(r) == tm, kills=k, died=s in died,
                               op=0 if op is None or s not in (op.att, op.vic) else (1 if op.att == s else -1),
                               first_dead=first_dead.get(tm) == s, traded=s in traded,
                               my_buy=buy(tb[tm], r), en_buy=buy(tb[5 - tm], r)))
    # убийства, урон, гранаты, ослепления
    for x in d[d.enemy].itertuples():
        S[x.att]['gun_kills'] += x.weapon in RIFLES or x.weapon not in {'hegrenade', 'inferno', 'knife', 'world'}
        S[x.att]['hs'] += bool(x.headshot)
        S[x.vic]['awp_deaths'] += x.weapon == 'awp'
        S[x.vic]['early_deaths'] += x.tick - fe[x.r - 1] < 20 * TICK
        blind = bl[(bl.vic == x.vic) & (bl.tick <= x.tick) & (bl.tick + bl.blind_duration * TICK >= x.tick)]
        S[x.vic]['died_blind'] += len(blind) > 0
        if x.vic == ME:
            deaths_me.append(dict(m=name, map=R['map'], side=SIDE[team(x.r, ME)], place=x.user_last_place_name,
                                  t=(x.tick - fe[x.r - 1]) / TICK, traded=None, weapon=x.weapon, r=x.r, tick=x.tick))
    for s, g in he.groupby('att'):
        S[s]['dmg'] += g.real.sum(); S[s]['he_dmg'] += g[g.weapon == 'hegrenade'].real.sum()
        S[s]['molly_dmg'] += g[g.weapon == 'inferno'].real.sum()
        S[s]['rifle_hits'] += g.weapon.isin(RIFLES).sum()
    sh = R['shots']; sh = sh[sh.weapon.isin(RIFLES)]
    for s, n_ in zip(sh.user_steamid.astype(str), sh.n): S[s]['rifle_shots'] += n_
    for ev, key in [('flashbang_detonate', 'flash'), ('hegrenade_detonate', 'he'), ('smokegrenade_detonate', 'smoke'), ('inferno_startburn', 'molly')]:
        for s in R['nades'].get(ev, pd.DataFrame(columns=['user_steamid'])).user_steamid.astype(str): S[s][key] += 1
    for x in bl[bl.blind_duration >= 1].itertuples():
        ta, tv = team(x.r, x.att), team(x.r, x.vic)
        if ta and tv: S[x.att]['flash_en' if ta != tv else 'flash_tm'] += x.vic != x.att
    for s in R['bomb'].get('bomb_planted', pd.DataFrame(columns=['user_steamid'])).user_steamid.astype(str): S[s]['plants'] += 1
    # размен своих смертей
    for dm in deaths_me:
        dm['traded'] = any(x['sid'] == ME and x['r'] == dm['r'] and x['traded'] for x in rounds if x['m'] == name)
    # позиции: место на 20 с и расстояние до ближайшего живого тиммейта
    pos = R['pos'].copy(); key = {f + s * TICK: (i + 1, s) for i, f in enumerate(fe) for s in range(5, 65, 5)}
    pos['r'] = pos.tick.map(lambda t: key[t][0]); pos['s'] = pos.tick.map(lambda t: key[t][1]); pos['sid'] = pos.steamid.astype(str)
    pos = pos[(pos.health > 0) & (pos.tick < pos.r.map(lambda r: end.get(r, 1e12))) & pos.team_num.isin([2, 3])]
    near = []
    for (t, tm), g in pos.groupby(['tick', 'team_num']):
        xy = g[['X', 'Y']].to_numpy()
        if len(xy) < 2: continue
        dist = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)); np.fill_diagonal(dist, np.inf)
        for s, dd, r, sec, place in zip(g.sid, dist.min(1), g.r, g.s, g.last_place_name):
            near.append(dict(m=name, map=R['map'], sid=s, side=SIDE[tm], s=sec, near=dd, place=place))
    pm = [dict(m=name, map=R['map'], sid=s, keep=s in keep, **c) for s, c in S.items() if c['rounds']]
    for c in clutches: c['m'] = name
    myteam = pl[n].get(ME) or pl[1].get(ME)
    score = (int((R['final'].team_num == myteam).sum() and R['final'][R['final'].team_num == myteam].team_rounds_total.iloc[0]),
             int(R['final'][R['final'].team_num == 5 - myteam].team_rounds_total.iloc[0]))
    short = np.mean([sorted(Counter(pl[r].values()).get(tm, 0) for tm in (2, 3)) != [5, 5] for r in pl])
    return dict(pm=pm, rounds=rounds, clutches=clutches, deaths_me=deaths_me, near=near, score=score, short=short,
                d=d, h=h, pl=pl, end=end, keep=keep, traded=traded_all,
                team_of={s: pl[n].get(s) for s in pl[n]}, aim=R['aim'].assign(m=name, map=R['map']))


def ratio_ci(num, den, B=2000):
    num = np.asarray(num, float); den = np.asarray(den, float)
    i = rng.integers(0, len(num), (B, len(num)))
    return np.percentile(num[i].sum(1) / np.maximum(den[i].sum(1), 1e-9), [2.5, 97.5])


def main(folder):
    M = {os.path.basename(f)[4:-4]: match(os.path.basename(f)[4:-4], pickle.load(open(f, 'rb')))
         for f in sorted(glob.glob(os.path.join(folder, 'res_*.pkl')))}
    for m in [m for m, v in M.items() if v['short'] >= 0.2]:
        print(f"ИСКЛЮЧЁН {m[:10]} {m[11:].split('_1-')[0]} {M[m]['score'][0]}:{M[m]['score'][1]} — неполный состав в {M[m]['short'] * 100:.0f}% раундов")
        del M[m]
    P = pd.DataFrame([x for m in M.values() for x in m['pm']]).fillna(0)
    me = P[P.sid == ME].set_index('m'); lob = P[(P.sid != ME) & P.keep]
    won = {m: v['score'][0] > v['score'][1] for m, v in M.items()}
    print(f"МАТЧЕЙ {len(M)}: побед {sum(won.values())}, поражений {len(M) - sum(won.values())}; раундов {int(me.rounds.sum())}; игроков лобби (>=90% раундов): {len(lob)} игроко-матчей\n")

    print('МАТЧИ (ADR и KAST — место в лобби из 10):')
    for m, v in M.items():
        x = P[P.m == m]; adr = x.dmg / x.rounds; kast = x.kast / x.rounds
        mi = x.index[x.sid == ME][0]
        print(f"  {m[:10]} {m[11:].split('_1-')[0]:12} {v['score'][0]:>2}:{v['score'][1]:<2} K/D {int(me.loc[m, 'kills']):>2}/{int(me.loc[m, 'deaths']):<2} "
              f"ADR {adr[mi]:5.1f} (#{int((adr > adr[mi]).sum()) + 1})  KAST {kast[mi] * 100:3.0f}% (#{int((kast > kast[mi]).sum()) + 1})")

    METRICS = [
        ('Убийства за раунд', 'kills', 'rounds', 1, 2), ('Смерти за раунд', 'deaths', 'rounds', 1, 2), ('ADR', 'dmg', 'rounds', 1, 1),
        ('KAST, %', 'kast', 'rounds', 100, 0), ('HS% убийств с огнестрела', 'hs', 'gun_kills', 100, 0),
        ('Точность рифлов (попадания/выстрелы), %', 'rifle_hits', 'rifle_shots', 100, 1),
        ('Участие в опенингах, % раундов', 'op_att', 'rounds', 100, 0), ('Выигранные опенинги, %', 'op_win', 'op_att', 100, 0),
        ('Первый контакт команды (урон), % раундов', 'first_contact', 'rounds', 100, 0),
        ('Размены (убил убийцу тиммейта) на 100 раундов', 'tradek', 'rounds', 100, 1),
        ('Твои смерти, которые разменяли, %', 'traded', 'deaths', 100, 0), ('Выживание, % раундов', 'surv', 'rounds', 100, 0),
        ('Раунды с 2+ убийствами, %', 'multi2', 'rounds', 100, 0), ('Смерти в первые 20 с раунда, %', 'early_deaths', 'deaths', 100, 0),
        ('Умер ослеплённым, % смертей', 'died_blind', 'deaths', 100, 0), ('Смерти от AWP, %', 'awp_deaths', 'deaths', 100, 0),
        ('Флешек на 24 раунда', 'flash', 'rounds', 24, 1), ('Врагов ослеплено (>=1 с) на флешку', 'flash_en', 'flash', 1, 2),
        ('Своих ослеплено на флешку', 'flash_tm', 'flash', 1, 2), ('HE на 24 раунда', 'he', 'rounds', 24, 1),
        ('Урон на одну HE', 'he_dmg', 'he', 1, 1), ('Молотовов на 24 раунда', 'molly', 'rounds', 24, 1),
        ('Урон на один молотов', 'molly_dmg', 'molly', 1, 1), ('Смоков на 24 раунда', 'smoke', 'rounds', 24, 1),
        ('Установки бомбы, % T-раундов', 'plants', 't_rounds', 100, 0)]
    print('\nТЫ vs ЛОББИ (95% CI по матчам; лобби — сумма по игрокам):')
    for lab, a, b, k, dec in METRICS:
        v = me[a].sum() / max(me[b].sum(), 1e-9) * k; lo, hi = ratio_ci(me[a], me[b]) * k; l = lob[a].sum() / lob[b].sum() * k
        print(f"  {lab:47} ты {v:6.{dec}f} [{lo:.{dec}f}–{hi:.{dec}f}]   лобби {l:6.{dec}f}")

    print('\nПОБЕДЫ vs ПОРАЖЕНИЯ (ты; рядом — средний тиммейт и соперник по ADR):')
    for lab, ms in [('победы', [m for m in M if won[m]]), ('поражения', [m for m in M if not won[m]])]:
        x = me.loc[ms]; tm_adr = []; op_adr = []
        for m in ms:
            t = M[m]['team_of']; y = P[(P.m == m) & (P.sid != ME)]
            tm_adr += list((y.dmg / y.rounds)[[t.get(s) == t.get(ME) for s in y.sid]])
            op_adr += list((y.dmg / y.rounds)[[t.get(s) not in (None, t.get(ME)) for s in y.sid]])
        print(f"  {lab:9} n={len(ms):2}  ADR {x.dmg.sum() / x.rounds.sum():5.1f} (тиммейты {np.mean(tm_adr):5.1f}, соперники {np.mean(op_adr):5.1f})  "
              f"KAST {x.kast.sum() / x.rounds.sum() * 100:3.0f}%  K/D {x.kills.sum() / x.deaths.sum():.2f}  опенинги {x.op_win.sum():.0f}/{x.op_att.sum():.0f}  "
              f"разменяно смертей {x.traded.sum() / x.deaths.sum() * 100:3.0f}%")

    Rd = pd.DataFrame([x for m in M.values() for x in m['rounds']])
    keepset = set(zip(lob.m, lob.sid)) | {(m, ME) for m in M}
    Rd = Rd[[(m, s) in keepset for m, s in zip(Rd.m, Rd.sid)]]
    rm, rl = Rd[Rd.sid == ME], Rd[Rd.sid != ME]
    print('\nВЛИЯНИЕ НА РАУНД (винрейт твоей команды в раундах, где…; для лобби — то же по каждому игроку):')
    for lab, f in [('все раунды', lambda x: x.won == x.won), ('ты выиграл опенинг', lambda x: x.op == 1), ('ты проиграл опенинг', lambda x: x.op == -1),
                   ('ты умер первым в команде', lambda x: x.first_dead), ('ты умер без размена', lambda x: x.died & ~x.traded),
                   ('ты сделал 2+ убийства', lambda x: x.kills >= 2), ('ты выжил', lambda x: ~x.died)]:
        a, b = rm[f(rm)], rl[f(rl)]
        print(f"  {lab:26} ты {a.won.mean() * 100:3.0f}% (n={len(a):3})   лобби {b.won.mean() * 100:3.0f}%")

    print('\nСТОРОНЫ (ты vs лобби):')
    for side in ('CT', 'T'):
        a, b = rm[rm.side == side], rl[rl.side == side]
        print(f"  {side:2}  винрейт раундов твоей команды {a.won.mean() * 100:3.0f}% (n={len(a)})   убийств/раунд {a.kills.mean():.2f} | лобби {b.kills.mean():.2f}   "
              f"смертей/раунд {a.died.mean():.2f} | {b.died.mean():.2f}   опенинги {(a.op == 1).sum()}/{(a.op != 0).sum()} ({(a.op != 0).mean() * 100:.0f}% раундов) | лобби {(b.op != 0).mean() * 100:.0f}%, выигрыш {(b.op == 1).sum() / max((b.op != 0).sum(), 1) * 100:.0f}%")
    print('  по картам (раунды твоей команды: n и винрейт; твои убийства и смерти за раунд):')
    for mp, g in rm.assign(map=rm.m.str.split('_').str[1:3].str.join('_')).groupby('map'):
        parts = []
        for side in ('CT', 'T'):
            x = g[g.side == side]
            parts.append(f"{side} n={len(x):3} win {x.won.mean() * 100:3.0f}% K {x.kills.mean():.2f} D {x.died.mean():.2f}")
        print(f"    {mp:11} матчей {g.m.nunique():2}   " + '   '.join(parts))

    print('\nЭКОНОМИКА (средняя стоимость снаряжения команды на конце фризтайма; винрейт твоей команды):')
    t = rm.groupby(['my_buy', 'en_buy']).won.agg(['size', 'mean'])
    for (a, b), row in t.iterrows():
        if row['size'] >= 5: print(f"  мы {a:10} против {b:10} n={int(row['size']):3}  win {row['mean'] * 100:3.0f}%")

    Dm = pd.DataFrame([x for m in M.values() for x in m['deaths_me']])
    print(f"\nТВОИ СМЕРТИ: {len(Dm)}, без размена {(~Dm.traded).sum()} ({(~Dm.traded).mean() * 100:.0f}%); CT без размена {(~Dm[Dm.side == 'CT'].traded).mean() * 100:.0f}%, T {(~Dm[Dm.side == 'T'].traded).mean() * 100:.0f}%")
    print(f"  время смерти от начала раунда: медиана {Dm.t.median():.0f} с; до 20 с {(Dm.t < 20).mean() * 100:.0f}%; по оружию убийцы: {dict(Counter(Dm.weapon).most_common(5))}")
    print('  чаще всего умираешь (карта, сторона, место: смертей / без размена):')
    for (mp, side, pl_), g in sorted(Dm.groupby(['map', 'side', 'place']), key=lambda kv: -len(kv[1]))[:10]:
        print(f"    {mp[3:]:8} {side:2} {pl_:18} {len(g):2} / {(~g.traded).sum():2}")

    C = pd.DataFrame([x for m in M.values() for x in m['clutches']])
    C = C[[(m, s) in keepset for m, s in zip(C.m, C.sid)]]
    print('\nКЛАТЧИ (последний живой против N): ты выиграно/всего | лобби винрейт')
    for vs in range(1, 6):
        a, b = C[(C.sid == ME) & (C.vs == vs)], C[(C.sid != ME) & (C.vs == vs)]
        if len(a) or len(b): print(f"  1v{vs}: {int(a.won.sum())}/{len(a)}   лобби {b.won.mean() * 100:3.0f}% (n={len(b)})")

    A = pd.concat([m['aim'] for m in M.values()]); A['att'] = A.att.astype(str)
    A = A[A.weapon.isin(RIFLES)]
    f, pre = A[A.when == 'first'], A[A.when == 'pre']
    g = f.groupby('att').agg(n=('v', 'size'), vert=('v', 'median'), horiz=('h', lambda x: np.median(np.abs(x))),
                              speed=('speed', 'median'), accurate=('speed', lambda x: (x <= 75).mean() * 100),
                              head=('hitgroup', lambda x: (x == 'head').mean() * 100))
    g = g.join(pre.groupby('att').v.median().rename('vert_pre')); g = g[g.n >= 15]; mrow = g.loc[ME]
    print(f"\nПРИЦЕЛ, РИФЛЫ (дуэли с попаданием; игроков с 15+ дуэлями: {len(g)}; у тебя {int(mrow.n)}):")
    for c, lab, low in [('vert', 'ниже головы на 1-м выстреле, °', True), ('vert_pre', 'ниже головы за 0.3 с до выстрела, °', True),
                        ('horiz', 'ошибка по горизонтали на 1-м выстреле, °', True), ('speed', 'скорость на 1-м выстреле, u/s', True),
                        ('accurate', '1-й выстрел на скорости <=75 u/s, %', False), ('head', '1-е попадание в голову, %', False)]:
        better = (g[c] > mrow[c]).mean() * 100 if low else (g[c] < mrow[c]).mean() * 100
        print(f"  {lab:42} ты {mrow[c]:6.2f}   медиана игроков {g[c].median():6.2f}   ты лучше {better:3.0f}% игроков")
    mf = f[f.att == ME]
    print('  ты по оружию:', {w: f"n={len(x)} верт {x.v.median():.2f}° голова {(x.hitgroup == 'head').mean() * 100:.0f}%" for w, x in mf.groupby('weapon') if len(x) >= 10})
    print('  ты по дистанции:', {str(b): f"n={len(x)} верт {x.v.median():.2f}° голова {(x.hitgroup == 'head').mean() * 100:.0f}%"
                              for b, x in mf.groupby(pd.cut(mf.dist, [0, 500, 1000, 1500, 5000]), observed=True)})
    print('  ты по скорости:', {str(b): f"n={len(x)} голова {(x.hitgroup == 'head').mean() * 100:.0f}%"
                             for b, x in mf.groupby(pd.cut(mf.speed, [-1, 75, 150, 400]), observed=True)},
          ' лобби:', {str(b): f"голова {(x.hitgroup == 'head').mean() * 100:.0f}%" for b, x in f[f.att != ME].groupby(pd.cut(f[f.att != ME].speed, [-1, 75, 150, 400]), observed=True)})

    N = pd.DataFrame([x for m in M.values() for x in m['near']])
    print('\nПОЗИЦИИ (расстояние до ближайшего живого тиммейта, медиана, юниты):')
    for side in ('CT', 'T'):
        for sec in (20, 40):
            a, b = N[(N.sid == ME) & (N.side == side) & (N.s == sec)], N[(N.sid != ME) & (N.side == side) & (N.s == sec)]
            print(f"  {side:2} на {sec} с: ты {a.near.median():5.0f} (дальше 1500 — {(a.near > 1500).mean() * 100:3.0f}%)   лобби {b.near.median():5.0f} ({(b.near > 1500).mean() * 100:3.0f}%)")
    print('  где ты на 20-й секунде (карта, сторона: топ-3 места, % раундов):')
    a = N[(N.sid == ME) & (N.s == 20)]
    for (mp, side), x in a.groupby(['map', 'side']):
        if len(x) < 8: continue
        top = x.place.value_counts(normalize=True).head(3)
        print(f"    {mp[3:]:8} {side:2} (n={len(x):2}): " + ', '.join(f"{p} {v * 100:.0f}%" for p, v in top.items()))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', '..', 'demos', 'parsed'))
