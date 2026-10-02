# Прогноз плато по ELO FACEIT: как твои ADR, K/D и винрейт зависят от среднего ELO лобби и где винрейт упадёт до ~50%.
#   python elo.py   (данные — collector: npm run history → data/faceit/)
import json, os
import numpy as np, pandas as pd
GUID = '08a23418-e244-4a05-bc94-680a06fecc1a'  # твой FACEIT id
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'faceit')
S = pd.DataFrame(json.load(open(os.path.join(F, 'stats.json'), encoding='utf-8')))
rows = []
for x in S.itertuples():
    f = os.path.join(F, 'matches', f'{x.matchId}.json')
    if not os.path.exists(f): continue
    p = json.load(open(f, encoding='utf-8')); t = p['teams']
    mine = 'faction1' if any(r['id'] == GUID for r in t['faction1']['roster']) else 'faction2'
    other = 'faction2' if mine == 'faction1' else 'faction1'
    me = [r for r in t[mine]['roster'] if r['id'] == GUID][0]
    elos = [r['elo'] for fct in ('faction1', 'faction2') for r in t[fct]['roster'] if r.get('elo')]
    rows.append(dict(date=pd.Timestamp(int(x.date), unit='ms'), win=int(x.i10), adr=float(x.c10), kd=float(x.c2), premade=bool(x.premade),
                     elo=me.get('elo'), stats_after=x.elo, stats_delta=x.elo_delta, lobby=np.mean(elos), opp=np.mean([r['elo'] for r in t[other]['roster'] if r.get('elo')]),
                     p_win=t[mine].get('stats', {}).get('winProbability')))
E = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
chk = E.assign(a=pd.to_numeric(E.stats_after, errors='coerce'), d=pd.to_numeric(E.stats_delta, errors='coerce')).dropna(subset=['a', 'd', 'elo'])
print(f"сверка ELO из комнаты с историей (до матча = после − изменение): совпало {(chk.elo == chk.a - chk.d).mean() * 100:.0f}% из {len(chk)}")
E = E[E.lobby.notna()].reset_index(drop=True); Ee = E[E.elo.notna() & (E.elo > 0)]
print(f"матчей: {len(E)}, с {E.date.min():%Y-%m-%d} по {E.date.max():%Y-%m-%d}")
for k in (24, 50, len(E)):
    x = E.tail(k)
    print(f"  последние {k:3}: винрейт {x.win.mean() * 100:.0f}%, ELO {Ee[Ee.date >= x.date.iloc[0]].elo.iloc[0]:.0f} → {int(S.dropna(subset=['elo']).sort_values('date').elo.iloc[-1])} после последнего матча с ELO, "
          f"ELO лобби {x.lobby.mean():.0f}, прогноз FACEIT {x.p_win.mean() * 100:.0f}%, ADR {x.adr.mean():.0f}, K/D {x.kd.mean():.2f}, пати {x.premade.mean() * 100:.0f}%")
dd = E.set_index('date').resample('W').size(); print(f"  матчей в неделю: последние 8 недель с игрой — {dd[dd > 0].tail(8).mean():.1f}; всего за 8 последних календарных недель {dd.tail(8).sum()}")
print(f"  ELO лобби: от {E.lobby.min():.0f} до {E.lobby.max():.0f}")
rng = np.random.default_rng(0)


def fit(x, y, logit=False):
    X = np.c_[np.ones(len(x)), x]
    if not logit: return np.linalg.lstsq(X, y, rcond=None)[0]
    w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w)); W = p * (1 - p) + 1e-9
        w -= np.linalg.solve(X.T @ (X * W[:, None]) + 1e-3 * np.eye(2), X.T @ (p - y) + 1e-3 * w)
    return w


z = (E.lobby.to_numpy() - 2200) / 100; adr = E.adr.to_numpy(); win = E.win.to_numpy().astype(float)
for col in ('adr', 'kd'):
    y = E[col].to_numpy(); b = [fit(z[i], y[i])[1] for i in (rng.integers(0, len(E), len(E)) for _ in range(2000))]
    print(f"  {col}: на +100 ELO лобби {fit(z, y)[1]:+.2f} [95% {np.percentile(b, 2.5):+.2f}…{np.percentile(b, 97.5):+.2f}]")
ww = fit(z, win, logit=True); bw = []
for _ in range(2000):
    i = rng.integers(0, len(E), len(E)); bw.append(fit(z[i], win[i], logit=True)[1])
print(f"  винрейт: наклон логита на +100 ELO лобби {ww[1]:+.2f} [95% {np.percentile(bw, 2.5):+.2f}…{np.percentile(bw, 97.5):+.2f}]; 50% при ELO лобби ≈ {2200 - 100 * ww[0] / ww[1]:.0f}")
wa, la = fit(adr, win, logit=True), fit(z, adr)
plat = lambda w_, l_: 2200 + 100 * ((-w_[0] / w_[1]) - l_[0]) / l_[1]
bo = []
for _ in range(3000):
    i = rng.integers(0, len(E), len(E)); w_, l_ = fit(adr[i], win[i], logit=True), fit(z[i], adr[i])
    if l_[1] < 0 and w_[1] > 0: bo.append(plat(w_, l_))
print(f"  ADR, при котором винрейт ~50%: {-wa[0] / wa[1]:.0f}; ELO лобби, где твой ADR до него опустится: {plat(wa, la):.0f}"
      f" [50%: {np.percentile(bo, 25):.0f}…{np.percentile(bo, 75):.0f}; 80%: {np.percentile(bo, 10):.0f}…{np.percentile(bo, 90):.0f}; бутстрапов с падающим ADR {len(bo) / 30:.0f}%]")
E['bin'] = pd.cut(E.lobby, [0, 2000, 2100, 2200, 2300, 3000])
print(E.groupby('bin', observed=True).agg(n=('win', 'size'), win=('win', 'mean'), adr=('adr', 'mean'), kd=('kd', 'mean')).round(2).to_string())
r = E.tail(24); print(f"  последние 24: ELO лобби {r.lobby.mean():.0f}, ADR {r.adr.mean():.0f} — линия по всем матчам даёт тут {la[0] + la[1] * (r.lobby.mean() - 2200) / 100:.0f}")
