# Сетапы (мышь, настройки, монитор): ты против лобби в тех же матчах, каждая группа — против твоей нормы по остальным матчам.
#   python setups.py A=d5d8bf4e,91fc7c80 B=616fbff7,cc8820b9   (кусок FACEIT match id; нужны кэши duels.py)
# Сравнение «ты − лобби» убирает силу соперников внутри матча, но не усталость, карту и порядок в сессии.
import glob, os, pickle, sys
import numpy as np, pandas as pd
from duels import ME, MOVE
from report import lobby_elo

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'demos', 'parsed')
G = {a.split('=')[0]: a.split('=')[1].split(',') for a in sys.argv[1:]}
gof = lambda m: next((g for g, ids in G.items() if any(i in m for i in ids)), 'норма')
D = pd.concat([x for f in sorted(glob.glob(os.path.join(FOLDER, 'duels_*.pkl'))) if (x := pickle.load(open(f, 'rb'))) is not None], ignore_index=True)
D = D[~D.post]
P = pd.concat([D.assign(P=D.K, keep=D.keepK, won=True, fs_hit=D.k_fs_hit, fs_head=D.k_fs_head, hits=D.k_hits, shots=D.k_shots, pre=D.k_pre, spd=D.spd_k, off0=D.k_off0),
               D.assign(P=D.V, keep=D.keepV, won=False, fs_hit=D.v_fs_hit, fs_head=D.v_fs_head, hits=D.v_hits, shots=D.v_shots, pre=D.v_pre, spd=D.spd_v, off0=D.v_off0)])
P = P[(P.P == ME) | P.keep].assign(G=lambda x: x.m.map(gof), me=lambda x: x.P == ME)
P['elo'] = P.m.map({m: lobby_elo(m) for m in P.m.unique()})
share = lambda s: (s == True).sum() / max(s.notna().sum(), 1) * 100
F = {'винрейт дуэлей': lambda x: x.won.mean() * 100,
     '1-й выстрел попал': lambda x: share(x.fs_hit),
     '1-й выстрел в голову': lambda x: share(x.fs_head),
     'точность в дуэлях': lambda x: x.hits.sum() / max(x.shots.sum(), 1) * 100,
     'проиграл, попав по врагу (не добил)': lambda x: (x[~x.won].hits > 0).mean() * 100,
     'на ходу прицел уже на враге': lambda x: share(x[(x.spd > MOVE) & (x.off0 < 15)].pre)}
gap = lambda x, f: f(x[x.me]) - f(x[~x.me])
byM = {m: x for m, x in P.groupby('m')}
ms = {g: P[P.G == g].m.unique() for g in P.G.unique()}
rng = np.random.default_rng(0)

N = P[P.G == 'норма']; med = N.elo.median()
print(f"матчей: " + ", ".join(f"{g} {len(v)} (ELO лобби {P[P.G == g].elo.mean():.0f})" for g, v in ms.items()))
print(f"«ты − лобби», п.п.; у нормы — и как он меняется с силой лобби (ниже/выше медианы {med:.0f}); Δ — группа минус норма [95% бутстрап по матчам]")
for k, f in F.items():
    cells = []
    for g in G:
        if g not in ms: continue
        d = gap(P[P.G == g], f) - gap(N, f)
        if len(ms[g]) < 3: cells.append(f"{g} {gap(P[P.G == g], f):+5.1f} (Δ {d:+.0f}, мало матчей)"); continue
        bs = [gap(pd.concat([byM[m] for m in rng.choice(ms[g], len(ms[g]))]), f) - gap(pd.concat([byM[m] for m in rng.choice(ms['норма'], len(ms['норма']))]), f)
              for _ in range(500)]
        cells.append(f"{g} {gap(P[P.G == g], f):+5.1f} (Δ {d:+.0f} [{np.percentile(bs, 2.5):+.0f}…{np.percentile(bs, 97.5):+.0f}])")
    print(f"  {k:36} норма {gap(N, f):+5.1f} ({gap(N[N.elo < med], f):+.1f} / {gap(N[N.elo >= med], f):+.1f})   " + "   ".join(cells))
