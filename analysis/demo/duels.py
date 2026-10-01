# Разбор дуэлей по тикам демки (3 с до каждого убийства): в чём ты проиграл каждую дуэль — аим, позиционка,
# решение, оружие/HP; аим в дуэлях и оружейные матчапы; винрейт по ситуациям; сильные/слабые позиции — против лобби.
#   python duels.py [папка с res_*.pkl]   (демки *.dem.zst — в родительской папке; кэш duels_<матч>.pkl рядом с res_*)
# ponytail: кэш не знает о версии duels() — поменял расчёт признаков, удали duels_*.pkl
# Дуэль = убийство: дуэли, где оба выжили, в демке не отличить от простого обмена уроном — их здесь нет.
import glob, os, sys, pickle, math, re
import numpy as np, pandas as pd
from demoparser2 import DemoParser
from extract import dec
from report import match, ratio_ci, ME, TICK, SIDE

WIN = 3 * TICK
FOV = 45              # враг дальше 45° от прицела по горизонтали — сбоку/сзади (край экрана 4:3; на 16:9 ~53°)
MOVE, HOLD = 100, 60  # u/s за 0.25 с: «двигался» / «стоял» (кто выглядывал)
SHOT_MOVE = 80        # u/s в момент выстрела: выше — стрелял в движении (порог точности рифлов/пистолетов ~75–80)
SAW = 0.25            # с: «увидел раньше», если разница больше
SLOW = 100            # мс: «выстрелил позже», если отстал больше
BAD, LOW_HP = 0.35, 40  # невыгодный матчап — ожидаемый винрейт ниже 35%; мало HP — меньше 40
GRP = {}
for g, names in [('пистолет', 'glock usp_silencer hkp2000 p250 fiveseven tec9 cz75a elite glock18 usps p2000 cz75auto dualberettas'),
                 ('дигл', 'deagle revolver deserteagle r8revolver'),
                 ('SMG/дробовик', 'mac10 mp9 mp7 mp5sd ump45 p90 bizon ppbizon nova xm1014 mag7 sawedoff'),
                 ('винтовка', 'ak47 m4a1 m4a1_silencer m4a1s m4a4 galilar famas aug sg556 sg553 m249 negev'),
                 ('снайперка', 'awp ssg08 g3sg1 scar20')]:
    GRP.update(dict.fromkeys(names.split(), g))
grp = lambda w: GRP.get(re.sub(r'[^a-z0-9_]', '', str(w).lower()))  # None — не огнестрел
NADE = ('flash', 'smoke', 'molotov', 'incendiary', 'highexplosive')


def streak(ticks):
    """Начало последнего непрерывного (разрывы <= 8 тиков) отрезка видимости."""
    if not ticks: return None
    s = ticks[-1]
    for t in reversed(ticks[:-1]):
        if s - t > 8: break
        s = t
    return s


assert streak([2, 4, 6, 20, 22]) == 20 and streak([2, 4, 8]) == 2 and streak([]) is None


def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 1.0
    p = k / n; dd = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / dd, (c + r) / dd


def duels(name, R, dem_path):
    M = match(name, R)
    if M['short'] >= 0.2: return None
    d, h, pl, end = M['d'], M['h'], M['pl'], M['end']
    fe = R['freeze']; ctx = {(x['r'], x['sid']): x for x in M['rounds']}
    kills = d[d.enemy]
    bl = R['blind'].assign(vic=R['blind'].user_steamid.astype(str))
    plants = R['bomb'].get('bomb_planted', pd.DataFrame(columns=['tick'])).tick.tolist()
    dem = dec(dem_path)
    try:
        p = DemoParser(dem)
        tk = p.parse_ticks(['X', 'Y', 'Z', 'pitch', 'yaw', 'duck_amount', 'approximate_spotted_by'],
                           ticks=sorted({tt for t in kills.tick for tt in range(t - WIN - 32, t + 1)}))
        inv = p.parse_ticks(['inventory', 'active_weapon_name'], ticks=sorted({t - 1 for t in kills.tick}))
        wf = p.parse_event('weapon_fire')
    finally:
        if dem != dem_path: os.remove(dem)
    tk['spot'] = [frozenset(map(str, l)) if isinstance(l, (list, np.ndarray)) else frozenset() for l in tk.approximate_spotted_by]
    P = {s: g.drop_duplicates('tick').set_index('tick')[['X', 'Y', 'Z', 'pitch', 'yaw', 'duck_amount', 'spot']] for s, g in tk.assign(sid=tk.steamid.astype(str)).groupby('sid')}
    I = inv.assign(sid=inv.steamid.astype(str)).set_index(['tick', 'sid'])
    shots_of = {s: np.sort(g.tick.to_numpy()) for s, g in wf.assign(sid=wf.user_steamid.astype(str)).groupby('sid')}

    def at(s, tt):
        df = P.get(s)
        return df.loc[tt] if df is not None and tt in df.index else None

    def speed(s, tt, dt=16):
        a, b = at(s, tt), at(s, tt - dt)
        return math.hypot(a.X - b.X, a.Y - b.Y) * TICK / dt if a is not None and b is not None else np.nan

    def off(a_sid, b_sid, tt):  # насколько b в стороне от прицела a по горизонтали, °
        a, b = at(a_sid, tt), at(b_sid, tt)
        return abs((a.yaw - math.degrees(math.atan2(b.Y - a.Y, b.X - a.X)) + 180) % 360 - 180) if a is not None and b is not None else np.nan

    def hp_before(s, r, tt):
        x = h[(h.vic == s) & (h.r == r) & (h.tick < tt)]
        return int(x.health.iloc[-1]) if len(x) else 100

    def track(a, b, ticks):
        """Прицел a относительно головы b по тикам: h — по горизонтали, v — по вертикали (>0 — ниже головы), °."""
        if a not in P or b not in P: return None
        A, B = P[a].reindex(ticks), P[b].reindex(ticks)
        dx, dy = B.X - A.X, B.Y - A.Y
        dz = (B.Z + 62 - 18 * B.duck_amount) - (A.Z + 64 - 18 * A.duck_amount)
        hz = np.hypot(dx, dy)
        hh = (A.yaw - np.degrees(np.arctan2(dy, dx)) + 180) % 360 - 180
        v = A.pitch + np.degrees(np.arctan2(dz, hz))
        head, wid, low = (np.degrees(np.arctan(u / hz)) for u in (5, 12, 40))  # голова ~5 юнитов, корпус ±12 и до 40 ниже головы
        return pd.DataFrame(dict(h=hh, v=v, on=(hh.abs() < wid) & (v > -head) & (v < low)), index=ticks)

    def aim(a, b, saw, t0, t):
        """Аим a по b с момента, когда a мог его увидеть, до смерти одного из них."""
        s0 = saw if saw is not None else t0
        T = track(a, b, list(range(s0, t + 1)))
        if T is None or T.h.isna().iloc[0]: return {}
        r0 = T.iloc[0]; off0 = math.hypot(r0.h, r0.v)
        on = T.index[T.on.to_numpy(dtype=bool)]
        # флаг видимости запаздывает (~1/4 первых выстрелов раньше него) — выстрелы берём с запасом 0.5 с
        st = shots_of.get(a, np.array([])); st = st[(st >= s0 - 32) & (st <= t)]
        hh = h[(h.att == a) & (h.vic == b) & (h.tick >= s0 - 32) & (h.tick <= t)]
        out = dict(off0=off0, v0=r0.v, pre=bool(r0.on) or off0 < 2, ttt=(on[0] - s0) / TICK * 1000 if len(on) else np.nan,
                   shots=len(st), hits=len(hh), dmg=hh.real.sum(), fs=np.nan, react=np.nan, fs_v=np.nan,
                   fs_on=False, fs_hit=False, fs_head=False, spd_fs=np.nan)
        if len(st):
            fs = int(st[0]); f = T.loc[fs] if fs in T.index else r0; first = hh[hh.tick.between(fs, fs + 1)]
            out.update(fs=fs, react=(fs - s0) / TICK * 1000, fs_v=f.v, fs_on=bool(f.on), fs_hit=len(first) > 0,
                       fs_head=bool((first.hitgroup == 'head').any()), spd_fs=speed(a, fs, 2))
        return out

    rows = []
    for x in kills.itertuples():
        t, r, K, V = int(x.tick), x.r, x.att, x.vic
        w = list(range(t - WIN, t + 1)); snap = lambda tt: max([u for u in w if u <= tt] or [w[0]])
        ks = streak([tt for tt in w if (a := at(V, tt)) is not None and K in a.spot])  # K видит V
        vs = streak([tt for tt in w if (a := at(K, tt)) is not None and V in a.spot])  # V видит K
        hv = h[(h.vic == V) & (h.r == r) & (h.tick <= t) & (h.tick >= t - WIN)]
        hk = hv[hv.att == K]; k_hit = int(hk.tick.min()) if len(hk) else t
        start = min([s for s in (ks, vs) if s is not None], default=None)
        if start is None:  # видимости нет — начало перестрелки по первому выстрелу
            fs_any = [st[(st >= t - WIN) & (st <= t)] for st in (shots_of.get(K, np.array([])), shots_of.get(V, np.array([])))]
            start = min([int(s[0]) for s in fs_any if len(s)] + [k_hit])
        t0 = snap(start)
        dead_before = set(d[(d.r == r) & (d.tick < t)].vic)
        alive = lambda tm: [s for s, tt in pl[r].items() if tt == tm and s not in dead_before]
        tv = pl[r][V]; mates = [s for s in alive(tv) if s != V]
        pv = at(V, t - 1)
        near = min([math.hypot(q.X - pv.X, q.Y - pv.Y) for s in mates if (q := at(s, t - 1)) is not None], default=np.nan) if pv is not None else np.nan
        try: iv = I.loc[(t - 1, V)]; nades = sum(any(k in re.sub(r'[^a-z]', '', str(i).lower()) for k in NADE) for i in iv.inventory); v_grp = grp(iv.active_weapon_name)
        except KeyError: nades, v_grp = np.nan, None
        blind = lambda s: bool(len(bl[(bl.vic == s) & (bl.tick <= t) & (bl.tick + bl.blind_duration * TICK >= t)]))
        c = ctx.get((r, V), {})
        row = dict(
            m=name, map=R['map'], r=r, tick=t, sec=(t - fe[r - 1]) / TICK, K=K, V=V, side_k=SIDE[pl[r][K]], side_v=SIDE[tv],
            place_k=x.attacker_last_place_name, place_v=x.user_last_place_name, weapon=x.weapon, k_grp=grp(x.weapon), v_grp=v_grp,
            wallbang=x.penetrated > 0, smoke=bool(x.thrusmoke), post=t > end.get(r, 1e12), pistol=r in (1, 13),
            saw_adv=(np.inf if vs is None else (vs - ks) / TICK) if ks is not None else (-np.inf if vs is not None else np.nan),
            off_v=off(V, K, snap(k_hit)), spd_k=speed(K, t0), spd_v=speed(V, t0), multi=hv[hv.enemy].att.nunique(),
            hp_k=hp_before(K, r, t0), hp_v=hp_before(V, r, min(t0, k_hit)), blind_v=blind(V), blind_k=blind(K),
            nades_v=nades, near_v=near, alive_v=len(alive(tv)), alive_k=len(alive(5 - tv)),
            planted=any(fe[r - 1] <= pt < t for pt in plants), traded=(r, V) in M['traded'],
            v_kills=len(d[(d.r == r) & (d.att == V) & d.enemy & (d.tick < t)]),
            v_round_dmg=h[(h.r == r) & (h.att == V) & h.enemy & (h.tick < t)].real.sum(),
            v_buy=c.get('my_buy'), k_buy=c.get('en_buy'), keepK=K in M['keep'], keepV=V in M['keep'])
        row.update({f'k_{k}': v for k, v in aim(K, V, ks, t0, t).items()})
        row.update({f'v_{k}': v for k, v in aim(V, K, vs, t0, t).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def why(x):
    """Как K убил V (для «как ты убиваешь»)."""
    if x.post: return 'после конца раунда'
    if x.k_grp is None: return 'граната/молотов/нож'
    if x.wallbang: return 'прострел'
    if x.smoke: return 'через смок'
    if x.off_v > FOV: return 'взял сбоку/сзади'
    if x.multi >= 2: return 'вместе с тиммейтом'
    if x.spd_k > MOVE and x.spd_v < HOLD: return 'выглянул на держащего'
    if x.spd_v > MOVE and x.spd_k < HOLD: return 'держал угол — пришёл пикер'
    if x.saw_adv > SAW: return 'увидел раньше'
    return 'дуэль на равных'


def blame(x):
    """В чём V проиграл дуэль: (категория, подпричина). Порядок важен — первое подходящее."""
    if x.post or x.k_grp is None or x.wallbang or x.smoke: return 'прочее', 'прострел/смок/граната/после раунда'
    if x.v_grp is None: return 'прочее', 'в руках не оружие (нож/граната/бомба)'
    if x.off_v > FOV: return 'позиционка', 'взяли сбоку/сзади'
    if x.multi >= 2: return 'позиционка', 'двое и больше'
    if x.saw_adv > SAW: return 'позиционка', 'тебя увидели раньше'
    if x.exp_v < BAD: return 'оружие/HP', f'невыгодное оружие: {x.v_grp} против {x.k_grp}'
    if x.hp_v < LOW_HP: return 'оружие/HP', 'мало HP до дуэли'
    k_pre, v_pre = x.k_pre == True, x.v_pre == True  # noqa: E712 — numpy bool / NaN
    if x.spd_v > MOVE and x.spd_k < HOLD and k_pre:
        return 'решение', 'выглянул в готовый угол' + (' с гранатой в кармане' if x.nades_v >= 1 else '')
    # честная дуэль — смотрим аим
    if np.isnan(x.v_fs):
        return 'аим', 'прицел был не на нём' if k_pre and not v_pre else 'не успел выстрелить'
    if x.v_spd_fs > SHOT_MOVE: return 'аим', 'стрелял в движении'
    if x.v_fs - x.k_fs > SLOW * TICK / 1000:  # кто выстрелил раньше — по тикам выстрелов, без флага видимости
        return 'аим', 'прицел был не на нём' if k_pre and not v_pre else 'выстрелил позже'
    if x.v_dmg == 0: return 'аим', 'успел выстрелить, но промахнулся'
    if x.v_dmg >= 70: return 'на равных', 'почти убил (70+ урона)'
    return 'аим', 'попадал, но меньше/не в голову'


def main(folder):
    D = []
    for f in sorted(glob.glob(os.path.join(folder, 'res_*.pkl'))):
        name = os.path.basename(f)[4:-4]; cache = os.path.join(folder, f'duels_{name}.pkl')
        if not os.path.exists(cache):
            x = duels(name, pickle.load(open(f, 'rb')), os.path.join(folder, '..', f'{name}.dem.zst'))
            pickle.dump(x, open(cache, 'wb')); print('разобран', name, flush=True)
        x = pickle.load(open(cache, 'rb'))
        if x is not None: D.append(x)
    D = pd.concat(D, ignore_index=True)
    D['k_grp'] = D.k_grp.astype(object).where(D.k_grp.notna(), None); D['v_grp'] = D.v_grp.astype(object).where(D.v_grp.notna(), None)

    # ожидаемый винрейт по оружию: как этот матчап играет лобби
    ok = ~D.post & D.k_grp.notna() & D.v_grp.notna()
    per = pd.concat([D[ok].assign(P=D.K, keep=D.keepK, me_g=D.k_grp, op_g=D.v_grp, won=True),
                     D[ok].assign(P=D.V, keep=D.keepV, me_g=D.v_grp, op_g=D.k_grp, won=False)])
    L = per[(per.P != ME) & per.keep].groupby(['me_g', 'op_g']).won.agg(['mean', 'size'])
    exp = {k: v['mean'] for k, v in L.iterrows() if v['size'] >= 20}
    D['exp_v'] = [exp.get((a, b), np.nan) for a, b in zip(D.v_grp, D.k_grp)]
    D['exp_k'] = [exp.get((a, b), np.nan) for a, b in zip(D.k_grp, D.v_grp)]
    D['cat'], D['sub'] = zip(*[blame(x) for x in D.itertuples()])
    D['kill'] = [why(x) for x in D.itertuples()]
    my_d, my_k = D[D.V == ME], D[D.K == ME]
    lob_d, lob_k = D[(D.V != ME) & D.keepV], D[(D.K != ME) & D.keepK]
    print(f"матчей {D.m.nunique()}, дуэлей {len(D)}; твоих убийств {len(my_k)}, смертей {len(my_d)}")

    print(f"\nВ ЧЁМ ТЫ ПРОИГРАЛ ДУЭЛЬ (все твои смерти; ты: n={len(my_d)}, лобби: n={len(lob_d)}; в скобках — сколько из них без размена)")
    tot = my_d.groupby('m').size()
    for cat, g in sorted(my_d.groupby('cat'), key=lambda kv: -len(kv[1])):
        per_m = my_d.groupby('m').cat.apply(lambda s: (s == cat).sum())
        lo, hi = ratio_ci(per_m.values, tot.reindex(per_m.index).values) * 100
        print(f"  {cat:11} ты {len(g) / len(my_d) * 100:3.0f}% [{lo:.0f}–{hi:.0f}] ({len(g):3}, без размена {(~g.traded).sum():3})   лобби {(lob_d.cat == cat).mean() * 100:3.0f}%")
        for sub, s in sorted(g.groupby('sub'), key=lambda kv: -len(kv[1])):
            print(f"      {sub:44} ты {len(s) / len(my_d) * 100:3.0f}% ({len(s):3})   лобби {(lob_d['sub'] == sub).mean() * 100:3.0f}%")

    # аим в честных дуэлях: без флангов, перекрёстков, прострелов/смоков, с примерно равным оружием и HP
    fair = D[ok & ~D.wallbang & ~D.smoke & (D.off_v <= FOV) & (D.multi < 2) & ~(D.saw_adv.abs() > SAW)
             & D.exp_v.between(BAD, 1 - BAD) & (D.hp_v >= LOW_HP) & (D.hp_k >= LOW_HP)]
    A = pd.concat([fair.filter(regex='^k_').rename(columns=lambda c: c[2:]).assign(P=fair.K, keep=fair.keepK, won=True),
                   fair.filter(regex='^v_').rename(columns=lambda c: c[2:]).assign(P=fair.V, keep=fair.keepV, won=False)])
    am, al = A[A.P == ME], A[(A.P != ME) & A.keep]
    print(f"\nАИМ В ЧЕСТНЫХ ДУЭЛЯХ (ты: {len(am)}, винрейт {am.won.mean() * 100:.0f}%; лобби: {len(al)}). Медианы; «в момент, когда увидел» — начало видимости")
    stats = [('прицел уже на враге, когда увидел его, %', lambda x: x.pre.astype(float).mean() * 100),
             ('отклонение прицела в этот момент, °', lambda x: x.off0.median()),
             ('  из них по вертикали (+ ниже головы), °', lambda x: x.v0.median()),
             ('время до наведения на корпус*, мс', lambda x: x.ttt.median()),
             ('время до первого выстрела*, мс', lambda x: x.react.median()),
             ('первая пуля попала, %', lambda x: x[x.shots > 0].fs_hit.astype(float).mean() * 100),
             ('первая пуля в голову, %', lambda x: x[x.shots > 0].fs_head.astype(float).mean() * 100),
             (f'первый выстрел в движении (>{SHOT_MOVE} u/s), %', lambda x: (x[x.shots > 0].spd_fs > SHOT_MOVE).mean() * 100),
             ('попаданий / выстрелов в дуэли, %', lambda x: x.hits.sum() / max(x.shots.sum(), 1) * 100)]
    print(f"  {'':46} {'ты: все':>8} {'выигр.':>7} {'проигр.':>7} | {'лобби: все':>10} {'выигр.':>7} {'проигр.':>7}")
    for lab, f in stats:
        vals = [f(am), f(am[am.won]), f(am[~am.won]), f(al), f(al[al.won]), f(al[~al.won])]
        print(f"  {lab:46} {vals[0]:8.1f} {vals[1]:7.1f} {vals[2]:7.1f} | {vals[3]:10.1f} {vals[4]:7.1f} {vals[5]:7.1f}")
    print('  * от флага видимости, а он запаздывает — абсолютные мс не реакция, сравнивать только с лобби')

    print('\nОРУЖИЕ (твой винрейт в матчапе vs лобби в том же матчапе; матчапы с 8+ твоими дуэлями)')
    pm = per[per.P == ME]
    for (a, b), g in sorted(pm.groupby(['me_g', 'op_g']), key=lambda kv: -len(kv[1])):
        if len(g) < 8: continue
        lo, hi = wilson(g.won.sum(), len(g))
        print(f"  ты {a:13} против {b:13} ты {g.won.mean() * 100:3.0f}% [{lo * 100:.0f}–{hi * 100:.0f}] (n={len(g):3})   лобби {exp.get((a, b), np.nan) * 100:3.0f}%")
    pm = pm.assign(e=[exp.get((a, b), np.nan) for a, b in zip(pm.me_g, pm.op_g)])
    pl_ = per[(per.P != ME) & per.keep].assign(e=lambda x: [exp.get((a, b), np.nan) for a, b in zip(x.me_g, x.op_g)])
    print(f"  дуэлей в невыгодном матчапе (у лобби <{BAD * 100:.0f}% побед): ты {(pm.e < BAD).mean() * 100:.0f}% | лобби {(pl_.e < BAD).mean() * 100:.0f}%")

    print(f"\nКАК ТЫ УБИВАЕШЬ  (ты: n={len(my_k)}; лобби: n={len(lob_k)})")
    for lab, cnt in my_k.kill.value_counts().items():
        print(f"  {lab:32} ты {cnt / len(my_k) * 100:4.0f}% ({cnt:3})   лобби {(lob_k.kill == lab).mean() * 100:4.0f}%")

    # винрейт дуэлей по ситуациям — с точки зрения каждого участника
    rows = []
    for x in D[ok].itertuples():
        for side, Pl, won in (('k', x.K, True), ('v', x.V, False)):
            mv, op = (x.spd_k, x.spd_v) if side == 'k' else (x.spd_v, x.spd_k)
            adv = x.saw_adv if side == 'k' else -x.saw_adv
            hp = x.hp_k if side == 'k' else x.hp_v
            rows.append(dict(P=Pl, keep=x.keepK if side == 'k' else x.keepV, won=won,
                             peek='ты выглядывал на стоящего' if mv > MOVE and op < HOLD else 'ты стоял, выглянул он' if op > MOVE and mv < HOLD
                             else 'оба двигались' if mv > MOVE and op > MOVE else 'оба стояли' if mv < HOLD and op < HOLD else None,
                             saw='ты увидел раньше' if adv > SAW else 'тебя увидели раньше' if adv < -SAW else 'одновременно' if not np.isnan(adv) else None,
                             hp='у тебя <40 HP' if hp < LOW_HP else None))
    S = pd.DataFrame(rows); sm, sl = S[S.P == ME], S[(S.P != ME) & S.keep]
    print(f"\nВИНРЕЙТ ДУЭЛЕЙ ПО СИТУАЦИЯМ (ты: {len(sm)}, винрейт {sm.won.mean() * 100:.0f}%; лобби 50%)")
    for col in ('peek', 'saw', 'hp'):
        for lab in S[col].dropna().unique():
            a, b = sm[sm[col] == lab], sl[sl[col] == lab]
            lo, hi = wilson(a.won.sum(), len(a))
            print(f"  {lab:28} ты {a.won.mean() * 100:4.0f}% [{lo * 100:.0f}–{hi * 100:.0f}] (n={len(a):3})   лобби {b.won.mean() * 100:4.0f}%   доля таких дуэлей: ты {len(a) / len(sm) * 100:3.0f}% | лобби {len(b) / len(sl) * 100:3.0f}%")

    # позиции: винрейт дуэлей на точке против лобби на той же точке
    pos = pd.concat([D[~D.post].assign(P=D.K, place=D.place_k, side=D.side_k, won=True, keep=D.keepK, untr=False),
                     D[~D.post].assign(P=D.V, place=D.place_v, side=D.side_v, won=False, keep=D.keepV, untr=~D.traded)])
    pos['spot'] = pos['map'].str[3:] + ' ' + pos.side + ' ' + pos.place.astype(str)
    lobw = pos[(pos.P != ME) & pos.keep].groupby('spot').won.agg(['mean', 'size'])
    mine = pos[pos.P == ME].groupby('spot').agg(n=('won', 'size'), k=('won', 'sum'), untr=('untr', 'sum'))
    mine = mine[mine.n >= 6].join(lobw)
    mine['lo'], mine['hi'] = zip(*[wilson(k, n) for k, n in zip(mine.k, mine.n)])
    cats = my_d.assign(spot=my_d['map'].str[3:] + ' ' + my_d.side_v + ' ' + my_d.place_v.astype(str)).groupby('spot').cat.agg(lambda s: {k: int(v) for k, v in s.value_counts().items()})
    fmt = lambda s, x: (f"  {s:28} убийств {int(x.k):2} / смертей {int(x.n - x.k):2} (без размена {int(x.untr):2})  винрейт {x.k / x.n * 100:3.0f}% [{x.lo * 100:.0f}–{x.hi * 100:.0f}]"
                        f"   лобби там {x['mean'] * 100:3.0f}% (n={int(x['size'])})   смерти: {cats.get(s, {})}")
    print(f"\nСИЛЬНЫЕ ПОЗИЦИИ (точки с 6+ твоими дуэлями: {len(mine)}; сортировка по нижней границе винрейта минус лобби):")
    for s, x in mine.assign(score=mine.lo - mine['mean']).sort_values('score', ascending=False).head(6).iterrows(): print(fmt(s, x))
    print('СЛАБЫЕ ПОЗИЦИИ (сортировка по верхней границе винрейта минус лобби):')
    for s, x in mine.assign(score=mine.hi - mine['mean']).sort_values('score').head(6).iterrows(): print(fmt(s, x))

    out = os.path.join(folder, 'my_duels.csv')
    k = my_k.assign(ты='убил', причина=my_k.kill, подпричина='', моё_место=my_k.place_k, моё_оружие=my_k.k_grp, его_оружие=my_k.v_grp, ожидаемо=my_k.exp_k,
                    **{f'мой_{c}': my_k[f'k_{c}'] for c in ('pre', 'off0', 'v0', 'ttt', 'react', 'fs_hit', 'fs_head', 'spd_fs', 'shots', 'hits', 'dmg')},
                    его_react=my_k.v_react, его_pre=my_k.v_pre)
    dd = my_d.assign(ты='умер', причина=my_d.cat, подпричина=my_d['sub'], моё_место=my_d.place_v, моё_оружие=my_d.v_grp, его_оружие=my_d.k_grp, ожидаемо=my_d.exp_v,
                     **{f'мой_{c}': my_d[f'v_{c}'] for c in ('pre', 'off0', 'v0', 'ttt', 'react', 'fs_hit', 'fs_head', 'spd_fs', 'shots', 'hits', 'dmg')},
                     его_react=my_d.k_react, его_pre=my_d.k_pre)
    cols = ['m', 'r', 'sec', 'ты', 'причина', 'подпричина', 'traded', 'side_v', 'моё_место', 'моё_оружие', 'его_оружие', 'ожидаемо', 'saw_adv', 'off_v',
            'spd_k', 'spd_v', 'multi', 'hp_v', 'nades_v', 'near_v'] + [c for c in k.columns if c.startswith('мой_')] + ['его_react', 'его_pre']
    pd.concat([k, dd]).sort_values(['m', 'tick'])[cols].round(2).to_csv(out, index=False, encoding='utf-8-sig')
    print('\nвсе твои дуэли с разметкой:', os.path.abspath(out))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', '..', 'demos', 'parsed'))
