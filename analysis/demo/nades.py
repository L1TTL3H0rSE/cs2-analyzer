# Гранаты и победа в раунде (все 10 игроков): утилити до первого убийства раунда в раундах полного закупа команды.
# Сравнение внутри одной команды в одной половине матча (сильная команда и кидает больше, и выигрывает чаще — это убираем)
# с поправкой на разницу закупа. Это связь, не доказанная причина: раскидка идёт вместе с выбором плента/тактики.
#   python nades.py [карта]   (нужны таблицы extract.py; все точки → demos/parsed/nades_spots.csv)
import glob, os, pickle, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from report import match, ME
from impact import fit_logit

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'demos', 'parsed')
ONLY = sys.argv[1] if len(sys.argv) > 1 else None
TYPES = {'smokegrenade_detonate': 'смок', 'flashbang_detonate': 'флешка', 'inferno_startburn': 'молотов', 'hegrenade_detonate': 'HE'}
MIN_N = 10  # командо-раундов с раскидкой и без неё, чтобы показать точку
MIN_M = 5   # и матчей с ней: бутстрап по 3 матчам даёт ложно узкие интервалы
B = 2000
rng = np.random.default_rng(0)

TR, NA, POS = [], [], []
for f in sorted(glob.glob(os.path.join(FOLDER, 'res_*.pkl'))):
    m = os.path.basename(f)[4:-4]; R = pickle.load(open(f, 'rb')); M = match(m, R)
    if M['short'] >= 0.2: continue
    fe, pl, d, end = R['freeze'], M['pl'], M['d'], M['end']
    POS.append(R['pos'].assign(map=R['map'])[['map', 'X', 'Y', 'Z', 'last_place_name']])
    ri = {t: i + 1 for i, t in enumerate(fe)}
    e = R['eco'][R['eco'].team_num.isin([2, 3])]
    teq = e.assign(r=e.tick.map(ri)).groupby(['r', 'team_num']).current_equip_value.mean()
    live = d[d.enemy & (d.tick <= d.r.map(end).fillna(1e12))].drop_duplicates('r').set_index('r')
    for x in pd.DataFrame(M['rounds']).drop_duplicates(['r', 'side']).itertuples():
        tm = 2 if x.side == 'T' else 3
        fk = live.loc[x.r] if x.r in live.index else None
        TR.append(dict(m=m, map=R['map'], r=x.r, side=x.side, won=float(x.won), buy=x.my_buy,
                       eq=(teq.get((x.r, tm), 0) - teq.get((x.r, 5 - tm), 0)) * 5 / 1000,
                       op=np.nan if fk is None else float(pl[x.r].get(fk.att) == tm),
                       half=0 if x.r <= 12 else 1 if x.r <= 24 else 2 + (x.r - 25) // 3, me=pl[x.r].get(ME) == tm,
                       contact=int(fk.tick) if fk is not None else end.get(x.r, 10 ** 9), start=fe[x.r - 1]))
    cut = {(t['r'], t['side']): (t['start'], t['contact']) for t in TR if t['m'] == m}
    for ev, g in R['nades'].items():
        for t, s, nx, ny, nz in zip(g.tick, g.user_steamid.astype(str), g.x, g.y, g.z):
            r = int(np.searchsorted(fe, t, side='right')); tm = pl.get(r, {}).get(s)
            if tm is None: continue
            a, b = cut.get((r, 'T' if tm == 2 else 'CT'), (0, -1))
            if a < t < b: NA.append(dict(m=m, map=R['map'], r=r, side='T' if tm == 2 else 'CT', type=TYPES[ev], sid=s, x=nx, y=ny, z=nz))
T, N = pd.DataFrame(TR), pd.DataFrame(NA)

# место падения — ближайшая точка позиций игроков на той же карте (названия зон из игры)
POS = pd.concat(POS).dropna()
for mp, g in N.groupby('map'):
    p = POS[POS['map'] == mp]
    N.loc[g.index, 'place'] = p.last_place_name.to_numpy()[cKDTree(p[['X', 'Y', 'Z']].to_numpy()).query(g[['x', 'y', 'z']].to_numpy())[1]]

T = T[T.buy == 'полная'].reset_index(drop=True)
grp = [T.m, T.side, T.half]
for col in ('won', 'op'):
    ok = T[col].notna()
    X = np.c_[np.ones(ok.sum()), (T.side[ok] == 'CT'), T['eq'][ok]]
    p = 1 / (1 + np.exp(-X @ fit_logit(X, T[col][ok].to_numpy())))
    T.loc[ok, col + '_res'] = T[col][ok] - p
    T[col + '_dm'] = T[col + '_res'] - T.groupby(grp)[col + '_res'].transform('mean')
key = ['m', 'r', 'side']
cnt = N.groupby(key + ['type']).size().unstack(fill_value=0).reindex(columns=list(TYPES.values()), fill_value=0)
T = T.merge(cnt, left_on=key, right_index=True, how='left').fillna({c: 0 for c in TYPES.values()})


def boot_diff(g, y, flag):
    """Разница средних (с раскидкой − без), 95% интервал бутстрапом по матчам."""
    ok = ~np.isnan(y); g, y, flag = g[ok], y[ok], flag[ok].astype(float)
    codes, inv = np.unique(g, return_inverse=True); k = len(codes)
    s1, n1 = np.bincount(inv, y * flag, k), np.bincount(inv, flag, k)
    s0, n0 = np.bincount(inv, y * (1 - flag), k), np.bincount(inv, 1 - flag, k)
    i = rng.integers(0, k, (B, k))
    with np.errstate(invalid='ignore', divide='ignore'):
        b = s1[i].sum(1) / n1[i].sum(1) - s0[i].sum(1) / n0[i].sum(1)
    return s1.sum() / n1.sum() - s0.sum() / n0.sum(), *np.nanpercentile(b, [2.5, 97.5])


def boot_ols(g, y, X):
    """Наклоны МНК (X уже без константы, демин внутри команды-половины), 95% интервал бутстрапом по матчам."""
    ok = ~np.isnan(y); g, y, X = g[ok], y[ok], X[ok]
    codes, inv = np.unique(g, return_inverse=True); k = len(codes)
    XX = np.zeros((k, X.shape[1], X.shape[1])); Xy = np.zeros((k, X.shape[1]))
    np.add.at(XX, inv, X[:, :, None] * X[:, None, :]); np.add.at(Xy, inv, X * y[:, None])
    i = rng.integers(0, k, (B, k)); ridge = 1e-6 * np.eye(X.shape[1])
    b = np.linalg.solve(XX[i].sum(1) + ridge, Xy[i].sum(1)[..., None])[..., 0]
    return np.linalg.solve(XX.sum(0) + ridge, Xy.sum(0)), np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)


S = T if ONLY is None else T[T['map'] == ONLY]
print(f"НАДО ПОНИМАТЬ: {S.m.nunique()} матчей, {len(S)} командо-раундов полного закупа; утилити — только до первого убийства раунда.")
print("Эффект = насколько чаще команда выигрывала раунд / опенинг, чем в своих же раундах той же половины без этого (п.п.).")
types = list(TYPES.values())
print("\nКАЖДАЯ ДОПОЛНИТЕЛЬНАЯ ГРАНАТА ДО КОНТАКТА (в среднем по картам):")
for side in ('T', 'CT'):
    x = S[S.side == side]; Xd = (x[types] - x.groupby(['m', 'side', 'half'])[types].transform('mean')).to_numpy(float)
    avg = x[types].mean()
    for col, lab in (('won_dm', 'раунд'), ('op_dm', 'опенинг')):
        b, lo, hi = boot_ols(x.m.to_numpy(), x[col].to_numpy(float), Xd)
        print(f"  {side:2} {lab:8} " + "   ".join(f"{t} {b[j] * 100:+4.1f} [{lo[j] * 100:+.0f}…{hi[j] * 100:+.0f}]" for j, t in enumerate(types)))
    print(f"     в среднем за раунд кидают: " + ", ".join(f"{t} {avg[t]:.1f}" for t in types))

rows = []
for (mp, side, typ, place), g in N[N.m.isin(set(S.m))].groupby(['map', 'side', 'type', 'place']):
    x = S[(S['map'] == mp) & (S.side == side)]
    used = x.set_index(key).index.isin(g.set_index(key).index)
    if used.sum() < MIN_N or (~used).sum() < MIN_N or x[used].m.nunique() < MIN_M: continue
    me_threw = g[g.sid == ME].set_index(key).index.unique()
    rw = boot_diff(x.m.to_numpy(), x.won_dm.to_numpy(float), used)
    ro = boot_diff(x.m.to_numpy(), x.op_dm.to_numpy(float), used)
    rows.append(dict(map=mp, side=side, type=typ, place=place, rounds=int(used.sum()), share=used.mean(),
                     win=rw[0], win_lo=rw[1], win_hi=rw[2], op=ro[0], op_lo=ro[1], op_hi=ro[2],
                     me_rounds=int(x[x.me].shape[0]), me_threw=len(me_threw), team_used_when_me=int(used[x.me.to_numpy()].sum())))
P = pd.DataFrame(rows).sort_values('win', ascending=False)
P.round(3).to_csv(os.path.join(FOLDER, 'nades_spots.csv'), index=False, encoding='utf-8-sig')


def show(x):
    for r in x.itertuples():
        sure = ' *' if r.win_lo > 0 or r.win_hi < 0 else ''
        print(f"  {r.map[3:]:8} {r.side:2} {r.type:7} {r.place:18} раунд {r.win * 100:+4.0f} [{r.win_lo * 100:+.0f}…{r.win_hi * 100:+.0f}]{sure:2}"
              f"  опенинг {r.op * 100:+4.0f} [{r.op_lo * 100:+.0f}…{r.op_hi * 100:+.0f}]   в {r.share * 100:2.0f}% раундов (n={r.rounds:3})"
              f"   ты кинул в {r.me_threw} из {r.me_rounds} своих")


print(f"\nТОЧКИ (карта, сторона, тип, куда упала) — {len(P)} точек с {MIN_N}+ раундами с раскидкой и без и {MIN_M}+ матчами; * — интервал не задевает 0"
      f" (при стольких сравнениях часть звёздочек случайна):")
print(" лучше всего связаны с победой:"); show(P.head(12))
print(" хуже всего:"); show(P.tail(6).iloc[::-1])

mine = N[N.m.isin(set(S.m))].merge(S[key + ['me']], on=key)
for side in ('T', 'CT'):
    x = mine[mine.side == side]; n_me = (S.me & (S.side == side)).sum(); n_all = (S.side == side).sum()
    you = x[x.sid == ME].groupby('type').size() / max(n_me, 1); team = x.groupby('type').size() / n_all / 5
    print(f"\nТВОЯ УТИЛИТИ ДО КОНТАКТА за {side} (на раунд полного закупа, ты | средний игрок): "
          + ", ".join(f"{t} {you.get(t, 0):.2f} | {team.get(t, 0):.2f}" for t in types))
