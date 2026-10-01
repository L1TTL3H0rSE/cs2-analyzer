# Вклад в раунды и моменты для просмотра — против лобби тех же матчей:
#   1) модель вероятности победы в раунде (кто жив, снаряжение, бомба, время) → цена каждого фрага, смерти,
#      установки и разминирования; твой вклад за раунд сверх K/D, самые дорогие смерти и ценные фраги;
#   2) предсказуемость: та же точка на 20-й секунде раунд за раундом — растёт ли смертность на ней;
#   3) шаги: было ли тебя слышно сопернику за 2 с до дуэли — и винрейт таких дуэлей;
#   4) моменты для просмотра: тик и команда demo_gototick (демку сначала распаковать из .zst).
#   python impact.py [папка с res_*.pkl]   (демки — в родительской папке; кэш steps_<матч>.pkl; нужны кэши duels.py)
# ponytail: модель — логистическая регрессия на ~3 тыс. состояний из этих же матчей; при сотнях матчей — отдельная
# модель по сторонам/картам и перекрёстная проверка по матчам.
import glob, os, pickle, sys
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from demoparser2 import DemoParser
from extract import dec
from report import match, ME, TICK, SIDE, ratio_ci
from duels import wilson, HOLD

ROUND, BOMB = 115, 40   # с: время раунда и таймер бомбы
AUDIBLE = 1100          # юнитов: примерная дальность слышимости бега (стены не учитываем)
PRE = 5 * TICK          # моменты для просмотра — с запасом 5 с


def features(a_ct, a_t, eq_ct, eq_t, planted, elapsed, since_plant):
    return [1.0, a_ct - a_t, a_ct + a_t, (eq_ct - eq_t) / 1000, planted, planted * (a_ct - a_t),
            (1 - planted) * elapsed / ROUND, planted * since_plant / BOMB]


def fit_logit(X, y, lam=1.0, it=30):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); W = p * (1 - p)
        w -= np.linalg.solve(X.T @ (X * W[:, None]) + lam * np.eye(len(w)), X.T @ (p - y) + lam * w)
    return w


def rounds_of(name, R):
    """Состояния раундов (для модели) и события с состояниями до/после."""
    M = match(name, R)
    if M['short'] >= 0.2: return None
    d, pl, end, fe = M['d'], M['pl'], M['end'], R['freeze']
    eco = R['eco']; ri = {t: i + 1 for i, t in enumerate(fe)}
    eq = {(ri[t], str(s)): v for t, s, v in zip(eco.tick, eco.steamid, eco.current_equip_value)}
    ct_won = {x['r']: x['won'] == (x['side'] == 'CT') for x in M['rounds']}
    bomb = {k: v.assign(sid=v.user_steamid.astype(str)) for k, v in R['bomb'].items()}
    rnd = lambda t: int(np.searchsorted(fe, t, side='right'))
    states, events = [], []
    for r in range(1, len(fe) + 1):
        if r not in ct_won: continue
        dead, plant_t = set(), None
        ev = [(t, 'kill', a, v, place_a, place_v) for t, a, v, e, place_a, place_v in
              zip(d.tick, d.att, d.vic, d.enemy, d.attacker_last_place_name, d.user_last_place_name) if e and rnd(t) == r and t <= end.get(r, 1e12)]
        for k, typ in (('bomb_planted', 'plant'), ('bomb_defused', 'defuse')):
            b = bomb.get(k)
            if b is not None:
                ev += [(int(t), typ, s, None, None, None) for t, s in zip(b.tick, b.sid) if rnd(t) == r]
        ev.sort(key=lambda e: e[0])

        def state(t):
            alive = [(s, tm) for s, tm in pl[r].items() if s not in dead]
            a = {tm: [s for s, q in alive if q == tm] for tm in (2, 3)}
            return features(len(a[3]), len(a[2]), sum(eq.get((r, s), 0) for s in a[3]), sum(eq.get((r, s), 0) for s in a[2]),
                            plant_t is not None, (t - fe[r - 1]) / TICK, (t - plant_t) / TICK if plant_t else 0)

        states.append((state(fe[r - 1]), ct_won[r]))
        for t, typ, actor, victim, pa, pv in ev:
            if actor not in pl[r]: continue
            before = state(t)
            if typ == 'kill': dead.add(victim)
            elif typ == 'plant': plant_t = t
            after = None if typ == 'defuse' else state(t)
            if after: states.append((after, ct_won[r]))
            events.append(dict(m=name, map=R['map'], r=r, tick=t, typ=typ, actor=actor, victim=victim, team=pl[r][actor],
                               place_a=pa, place_v=pv, before=before, after=after, keepA=actor in M['keep'],
                               keepV=victim in M['keep'] if victim else False, sec=(t - fe[r - 1]) / TICK))
    return dict(states=states, events=events, score=M['score'], keep=M['keep'], pl=pl, d=d, end=end, fe=fe, rounds=M['rounds'])


def footsteps(name, folder):
    cache = os.path.join(folder, f'steps_{name}.pkl')
    if not os.path.exists(cache):
        dem = dec(os.path.join(folder, '..', f'{name}.dem.zst'))
        try: s = DemoParser(dem).parse_event('player_footstep', player=['X', 'Y', 'Z'])
        finally: os.remove(dem)
        pickle.dump(s.assign(sid=s.user_steamid.astype(str))[['tick', 'sid', 'user_X', 'user_Y', 'user_Z']], open(cache, 'wb'))
    return pickle.load(open(cache, 'rb'))


def main(folder):
    G = {}
    for f in sorted(glob.glob(os.path.join(folder, 'res_*.pkl'))):
        name = os.path.basename(f)[4:-4]; x = rounds_of(name, pickle.load(open(f, 'rb')))
        if x is not None: G[name] = (x, pickle.load(open(f, 'rb')))
    label = lambda m: f"{m[:10]} {m.split('_1-')[0][11:]} {G[m][0]['score'][0]}:{G[m][0]['score'][1]}"
    moments = []

    # 1) модель вероятности победы
    S = [s for g, _ in G.values() for s in g['states']]
    X = np.array([s for s, _ in S]); y = np.array([float(w) for _, w in S])
    w = fit_logit(X, y)
    p = 1 / (1 + np.exp(-X @ w))
    base = fit_logit(X[:, :3], y); pb = 1 / (1 + np.exp(-X[:, :3] @ base))
    ll = lambda q: -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    print(f"МОДЕЛЬ ВЕРОЯТНОСТИ ПОБЕДЫ В РАУНДЕ: {len(S)} состояний из {len(G)} матчей; log-loss {ll(p):.3f} (только «кто жив» {ll(pb):.3f}, монетка 0.693)")
    cal = pd.DataFrame(dict(p=p, y=y)).assign(b=lambda x: pd.cut(x.p, [0, .2, .4, .6, .8, 1]))
    print('  калибровка (предсказано → факт):', ', '.join(f"{g.p.mean():.2f}→{g.y.mean():.2f} (n={len(g)})" for _, g in cal.groupby('b', observed=True)))
    wp = lambda f: float(1 / (1 + np.exp(-np.dot(f, w))))
    for a, b in ((5, 5), (5, 4), (4, 5), (4, 3), (2, 1), (1, 2)):
        print(f"    {a}v{b} при равном снаряжении, без бомбы, середина раунда: CT {wp(features(a, b, 4000 * a, 4000 * b, 0, 40, 0)) * 100:.0f}%", end=';')
    print(f"  4v4 после установки, 10 с: CT {wp(features(4, 4, 16000, 16000, 1, 60, 10)) * 100:.0f}%")

    E = pd.DataFrame([e for g, _ in G.values() for e in g['events']])
    E['ct_before'] = [wp(b) for b in E.before]
    E['ct_after'] = [1.0 if t == 'defuse' else wp(a) for t, a in zip(E.typ, E.after)]
    E['gain'] = np.where(E.team == 3, 1, -1) * (E.ct_after - E.ct_before)   # для команды действующего игрока
    E['team_before'] = np.where(E.team == 3, E.ct_before, 1 - E.ct_before)
    credit = pd.concat([E[['m', 'r', 'actor', 'keepA', 'gain', 'typ']].rename(columns={'actor': 'P', 'keepA': 'keep'}),
                        E[E.typ == 'kill'][['m', 'r', 'victim', 'keepV', 'gain']].rename(columns={'victim': 'P', 'keepV': 'keep'}).assign(gain=lambda x: -x.gain, typ='death')])
    rounds_n = Counter((m, s) for g, _ in G.values() for x in g['rounds'] for m, s in [(x['m'], x['sid'])])
    per = credit.groupby(['m', 'P']).gain.sum().reset_index()
    per['n'] = [rounds_n[(m, s)] for m, s in zip(per.m, per.P)]
    keep_pm = {(m, s) for m, (g, _) in G.items() for s in g['keep']}
    per['keep'] = [(m, s) in keep_pm for m, s in zip(per.m, per.P)]; per['wpr'] = per.gain / per.n
    me_pm, lob_pm = per[per.P == ME], per[(per.P != ME) & per.keep]
    lo, hi = ratio_ci(me_pm.gain.values, me_pm.n.values) * 100
    ranks = [g[g.keep | (g.P == ME)].wpr.rank(ascending=False)[g.P == ME].iloc[0] for _, g in per.groupby('m') if (g.P == ME).any()]
    print(f"\nТВОЙ ВКЛАД В РАУНДЫ (сумма изменений шанса команды выиграть раунд от твоих фрагов, смертей, установок и разминирований)")
    print(f"  за раунд: ты {me_pm.gain.sum() / me_pm.n.sum() * 100:+.1f} п.п. [{lo:+.1f}…{hi:+.1f}]   лобби {lob_pm.gain.sum() / lob_pm.n.sum() * 100:+.1f} п.п.   "
          f"место в лобби матча (медиана): {np.median(ranks):.0f}; топ-3 в {np.mean(np.array(ranks) <= 3) * 100:.0f}% матчей")
    cm, cl = credit[credit.P == ME], credit[(credit.P != ME) & credit.keep]
    for typ, lab in (('kill', 'фраг'), ('death', 'смерть'), ('plant', 'установка'), ('defuse', 'разминирование')):
        a, b = cm[cm.typ == typ], cl[cl.typ == typ]
        if len(a): print(f"  {lab:15} в среднем {a.gain.mean() * 100:+5.1f} п.п. (n={len(a):3}; сумма {a.gain.sum() * 100:+6.0f})   лобби {b.gain.mean() * 100:+5.1f} п.п.")
    kills_me = E[(E.typ == 'kill') & (E.actor == ME)]
    print(f"  фрагов, сделанных при шансе команды ≤30% (камбэк-фраги): ты {(kills_me.team_before <= .3).mean() * 100:.0f}% | лобби {(E[(E.typ == 'kill') & (E.actor != ME) & E.keepA].team_before <= .3).mean() * 100:.0f}%;"
          f"  когда раунд уже почти выигран (≥85%): ты {(kills_me.team_before >= .85).mean() * 100:.0f}% | лобби {(E[(E.typ == 'kill') & (E.actor != ME) & E.keepA].team_before >= .85).mean() * 100:.0f}%")
    dm = E[(E.typ == 'kill') & (E.victim == ME)].assign(cost=lambda x: x.gain, my_before=lambda x: 1 - x.team_before)
    print(f"  смертей в выгодном раунде (шанс команды ≥60%), стоивших ≥10 п.п.: ты {((dm.my_before >= .6) & (dm.cost >= .10)).mean() * 100:.0f}% смертей | лобби "
          f"{(((1 - E[(E.typ == 'kill') & (E.victim != ME) & E.keepV].team_before) >= .6) & (E[(E.typ == 'kill') & (E.victim != ME) & E.keepV].gain >= .10)).mean() * 100:.0f}%")
    print('  самые дорогие твои смерти:')
    for x in dm.sort_values('cost', ascending=False).head(8).itertuples():
        print(f"    {label(x.m):24} р{x.r:2} {x.sec:4.0f}с {x.place_v:14} шанс команды {x.my_before * 100:3.0f}% → {(x.my_before - x.cost) * 100:3.0f}%")
        moments.append(dict(категория='дорогая смерть', m=x.m, r=x.r, sec=x.sec, tick=x.tick, что=f"{x.place_v}: шанс {x.my_before * 100:.0f}%→{(x.my_before - x.cost) * 100:.0f}%"))
    print('  самые ценные твои фраги:')
    for x in kills_me.sort_values('gain', ascending=False).head(5).itertuples():
        print(f"    {label(x.m):24} р{x.r:2} {x.sec:4.0f}с {x.place_a:14} шанс команды {x.team_before * 100:3.0f}% → {(x.team_before + x.gain) * 100:3.0f}%")
        moments.append(dict(категория='ценный фраг', m=x.m, r=x.r, sec=x.sec, tick=x.tick, что=f"{x.place_a}: шанс {x.team_before * 100:.0f}%→{(x.team_before + x.gain) * 100:.0f}%"))

    # 2) предсказуемость
    rows = []
    for m, (g, R) in G.items():
        pos = R['pos']; fe = g['fe']; t20 = {f + 20 * TICK: i + 1 for i, f in enumerate(fe)}
        pos = pos[pos.tick.isin(list(t20)) & (pos.health > 0)].assign(r=lambda x: x.tick.map(t20), sid=lambda x: x.steamid.astype(str))
        pos = pos[pos.tick < pos.r.map(lambda r: g['end'].get(r, 1e12))]
        d = g['d']; dd = d[d.enemy]
        for s, ps in pos.groupby('sid'):
            seen = Counter()
            for x in ps.sort_values('r').itertuples():
                side = SIDE[g['pl'][x.r].get(s, 0)] if g['pl'][x.r].get(s) else None
                if side is None: continue
                key = (side, x.last_place_name); rep = seen[key]; seen[key] += 1
                died = dd[(dd.r == x.r) & (dd.vic == s) & (dd.tick > x.tick)]
                rows.append(dict(m=m, P=s, keep=s in g['keep'], side=side, place=x.last_place_name, rep=min(rep, 3), r=x.r,
                                 died_here=bool(len(died)) and died.user_last_place_name.iloc[0] == x.last_place_name,
                                 died=bool(len(died)), tick=int(died.tick.iloc[0]) if len(died) else None,
                                 sec=(died.tick.iloc[0] - fe[x.r - 1]) / TICK if len(died) else None,
                                 kill_here=bool(((dd.r == x.r) & (dd.att == s) & (dd.attacker_last_place_name == x.last_place_name)).any())))
    Pz = pd.DataFrame(rows); pm, po = Pz[Pz.P == ME], Pz[(Pz.P != ME) & Pz.keep]
    print('\nПРЕДСКАЗУЕМОСТЬ (где ты на 20-й секунде; повтор — сколько раз до этого в матче стоял там же за эту сторону)')
    for side in ('CT', 'T'):
        a, b = pm[pm.side == side], po[po.side == side]
        top = a.groupby('m').place.agg(lambda s: s.value_counts(normalize=True).iloc[0]).mean()
        topl = b.groupby(['m', 'P']).place.agg(lambda s: s.value_counts(normalize=True).iloc[0]).mean()
        print(f"  {side}: доля раундов на своей самой частой точке в матче — ты {top * 100:.0f}% | лобби {topl * 100:.0f}%")
        for rep in range(4):
            aa, bb = a[a.rep == rep], b[b.rep == rep]
            lo, hi = wilson(aa.died_here.sum(), len(aa))
            print(f"    {['впервые', '2-й раз', '3-й раз', '4-й и дальше'][rep]:13} умер на этой точке {aa.died_here.mean() * 100:3.0f}% [{lo * 100:.0f}–{hi * 100:.0f}] (n={len(aa):3})"
                  f" | лобби {bb.died_here.mean() * 100:3.0f}%   убил с неё {aa.kill_here.mean() * 100:3.0f}% | {bb.kill_here.mean() * 100:3.0f}%")
    rep_deaths = pm[(pm.rep >= 3) & pm.died_here]
    for x in rep_deaths.itertuples():
        moments.append(dict(категория='смерть на повторной точке', m=x.m, r=x.r, sec=x.sec, tick=x.tick, что=f"{x.side} {x.place}: {x.rep + 1}-й+ раз"))

    # 3) шаги перед дуэлью
    D = pd.concat([x for f in sorted(glob.glob(os.path.join(folder, 'duels_*.pkl'))) if (x := pickle.load(open(f, 'rb'))) is not None], ignore_index=True)
    D = D[~D.post & D.m.isin(list(G))]
    rows = []
    for m, g in D.groupby('m'):
        st = footsteps(m, folder); dth = G[m][0]['d'].set_index('tick')
        for x in g.itertuples():
            if x.tick not in dth.index: continue
            dr = dth.loc[[x.tick]]; dr = dr[dr.vic == x.V].iloc[0]
            pos = {x.K: (dr.attacker_X, dr.attacker_Y, dr.attacker_Z), x.V: (dr.user_X, dr.user_Y, dr.user_Z)}
            for Pl, O, won, o_spd in ((x.K, x.V, True, x.spd_v), (x.V, x.K, False, x.spd_k)):
                s = st[(st.sid == Pl) & st.tick.between(x.tick - 2 * TICK, x.tick)]
                ox, oy, oz = pos[O]
                loud = bool((np.sqrt((s.user_X - ox) ** 2 + (s.user_Y - oy) ** 2 + (s.user_Z - oz) ** 2) <= AUDIBLE).any())
                rows.append(dict(m=m, P=Pl, keep=x.keepK if won else x.keepV, won=won, loud=loud, holder=o_spd < HOLD, tick=x.tick, r=x.r, sec=x.sec,
                                 place=x.place_k if won else x.place_v))
    Sx = pd.DataFrame(rows); sm, so = Sx[Sx.P == ME], Sx[(Sx.P != ME) & Sx.keep]
    print(f"\nШАГИ ЗА 2 С ДО ДУЭЛИ В РАДИУСЕ ~{AUDIBLE} ЮНИТОВ ОТ СОПЕРНИКА (бег слышно, шаг с Shift — нет; стены не учтены)")
    print(f"  тебя было слышно в {sm.loud.mean() * 100:.0f}% дуэлей | лобби {so.loud.mean() * 100:.0f}%")
    for lab, f in (('тебя было слышно', lambda x: x.loud), ('ты был тихим', lambda x: ~x.loud),
                   ('слышно, а он стоял и ждал', lambda x: x.loud & x.holder), ('тихо, а он стоял и ждал', lambda x: ~x.loud & x.holder)):
        a, b = sm[f(sm)], so[f(so)]
        lo, hi = wilson(a.won.sum(), len(a))
        print(f"    {lab:28} винрейт {a.won.mean() * 100:3.0f}% [{lo * 100:.0f}–{hi * 100:.0f}] (n={len(a):3}) | лобби {b.won.mean() * 100:3.0f}%")
    for x in sm[sm.loud & sm.holder & ~sm.won].itertuples():
        moments.append(dict(категория='умер после своих шагов, его ждали', m=x.m, r=x.r, sec=x.sec, tick=x.tick, что=str(x.place)))

    # 4) моменты для просмотра
    Mo = pd.DataFrame(moments)
    Mo['матч'] = Mo.m.map(label); Mo['демка'] = Mo.m + '.dem'
    Mo['команда'] = [f"demo_gototick {max(int(t) - PRE, 0)}" for t in Mo.tick]
    out = os.path.join(folder, 'my_moments.csv')
    Mo.sort_values(['категория', 'm', 'r'])[['категория', 'матч', 'r', 'sec', 'что', 'демка', 'команда']].round(0).to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\nмоменты для просмотра ({len(Mo)}: {dict(Counter(Mo['категория']))}): {os.path.abspath(out)}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', '..', 'demos', 'parsed'))
