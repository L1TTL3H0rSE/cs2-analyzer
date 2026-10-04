# Сравнение твоих матчей на карте: раньше даты («до») и с даты («после») — общая игра, стороны, дуэли и аим,
# причины смертей, конверсия, вклад в раунды, позиции, AWP.
#   python compare.py de_dust2 2026-10-01   (нужны таблицы extract.py и кэши duels.py, conversion.py, awp.py)
#   python compare.py all 2026-10-03T09:00 2026-10-03T14:00 2026-10-04   — два окна [1-я, 2-я) и [2-я, 3-я), напр. сетапы
#     внутри одного дня; время — UTC, начало матча (finishedAt в collector/state.json), all — все карты
import glob, json, os, pickle, sys
from collections import Counter
import numpy as np, pandas as pd
from report import match, ME, ratio_ci, lobby_elo
from duels import annotate, MOVE, HOLD
from impact import rounds_of, model

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'demos', 'parsed')
MAP, CUTS = sys.argv[1], sys.argv[2:]
RIFLES = {'ak47', 'm4a1_silencer', 'm4a1', 'galilar', 'famas', 'aug', 'sg556'}
STATE = json.load(open(os.path.join(FOLDER, '..', '..', 'collector', 'state.json'), encoding='utf-8'))['done']
when = lambda m: (STATE.get('1-' + m.split('_1-')[1], {}).get('finishedAt') or m[:10])
lo_, mid, hi_ = (CUTS + [''])[:3] if len(CUTS) == 3 else ('', CUTS[0], '￿')
grp = lambda m: None if not lo_ <= when(m) < hi_ else 'после' if when(m) >= mid else 'до'
GR = ('до', 'после')

res = {os.path.basename(f)[4:-4]: pickle.load(open(f, 'rb')) for f in sorted(glob.glob(os.path.join(FOLDER, 'res_*.pkl')))}
M = {m: match(m, R) for m, R in res.items()}
M = {m: x for m, x in M.items() if x['short'] < 0.2}
sel = [m for m in M if (MAP == 'all' or f'_{MAP}_' in m) and grp(m)]
by = {g: [m for m in sel if grp(m) == g] for g in GR}


def row(lab, f, fmt='{:.0f}', ci=None):
    cells = []
    for g in GR:
        v = f(g)
        cells.append((fmt.format(v) if v == v else '—') + (f" [{ci(g)[0]:.0f}–{ci(g)[1]:.0f}]" if ci else ''))
    print(f"  {lab:46} {cells[0]:>16} | {cells[1]:<16}")


def wl(g):
    s = [M[m]['score'] for m in by[g]]
    return f"{len(s)} матчей, {sum(a > b for a, b in s)}–{sum(a < b for a, b in s)}"


print(f"{MAP}: до {mid} — {wl('до')}; после — {wl('после')}")
el = {g: np.nanmean([lobby_elo(m) for m in by[g]]) for g in GR}
# ponytail: −3.4 ADR на +100 ELO лобби — оценка analysis/faceit/elo.py по 100 матчам; пересчитать, когда матчей станет больше
print(f"  средний ELO лобби: до {el['до']:.0f} | после {el['после']:.0f} → только из-за лобби ожидаемо ADR {-3.4 * (el['после'] - el['до']) / 100:+.1f}")
print(f"  {'':46} {'до':>16} | после")

# общая игра
pm = {g: pd.DataFrame([x for m in by[g] for x in M[m]['pm'] if x['sid'] == ME]).fillna(0) for g in GR}
r_ = lambda a, b, k=1: (lambda g: pm[g][a].sum() / max(pm[g][b].sum(), 1e-9) * k)
print('ОБЩАЯ ИГРА')
row('раундов', lambda g: pm[g].rounds.sum())
row('убийства за раунд', r_('kills', 'rounds'), '{:.2f}')
row('смерти за раунд', r_('deaths', 'rounds'), '{:.2f}')
row('ADR', r_('dmg', 'rounds'), '{:.0f}', ci=lambda g: ratio_ci(pm[g].dmg, pm[g].rounds) if len(pm[g]) > 1 else (np.nan, np.nan))
row('KAST, %', r_('kast', 'rounds', 100))
row('HS% убийств с огнестрела', r_('hs', 'gun_kills', 100))
row('точность рифлов, %', r_('rifle_hits', 'rifle_shots', 100), '{:.1f}')
row('участие в опенингах, % раундов', r_('op_att', 'rounds', 100))
row('выигранные опенинги, %', r_('op_win', 'op_att', 100))
row('твои смерти разменяли, %', r_('traded', 'deaths', 100))
row('раунды с 2+ убийствами, %', r_('multi2', 'rounds', 100))
row('флешек на 24 раунда / врагов на флешку', lambda g: pm[g].flash.sum() / pm[g].rounds.sum() * 24, '{:.1f}')
row('  врагов ослеплено на флешку', r_('flash_en', 'flash'), '{:.2f}')
row('HE на 24 раунда', r_('he', 'rounds', 24), '{:.1f}')

# стороны
Rd = {g: pd.DataFrame([x for m in by[g] for x in M[m]['rounds'] if x['sid'] == ME]) for g in GR}
print('СТОРОНЫ')
for side in ('CT', 'T'):
    s = lambda g: Rd[g][Rd[g].side == side]
    row(f'{side}: раундов / винрейт команды, %', lambda g: s(g).won.mean() * 100, '{:.0f}')
    row(f'{side}: убийства / смерти за раунд', lambda g: s(g).kills.mean(), '{:.2f}')
    row(f'{side}:   смерти за раунд', lambda g: s(g).died.mean(), '{:.2f}')
    row(f'{side}: опенинги — участие, %', lambda g: (s(g).op != 0).mean() * 100)
    row(f'{side}:   выигрыш, %', lambda g: (s(g).op == 1).sum() / max((s(g).op != 0).sum(), 1) * 100)
    row(f'{side}: умер первым в команде, %', lambda g: s(g).first_dead.mean() * 100)

# дуэли и аим
D, per, exp, ok = annotate(pd.concat([x for f in sorted(glob.glob(os.path.join(FOLDER, 'duels_*.pkl'))) if (x := pickle.load(open(f, 'rb'))) is not None], ignore_index=True))
D = D[D.m.isin(sel) & ~D.post]
P = pd.concat([D.assign(P=D.K, won=True, spd=D.spd_k, ospd=D.spd_v, pre=D.k_pre, off0=D.k_off0, g_=D.k_grp, og=D.v_grp, react=D.k_react, fs_hit=D.k_fs_hit),
               D.assign(P=D.V, won=False, spd=D.spd_v, ospd=D.spd_k, pre=D.v_pre, off0=D.v_off0, g_=D.v_grp, og=D.k_grp, react=D.v_react, fs_hit=D.v_fs_hit)])
P = P[P.P == ME].assign(G=lambda x: x.m.map(grp))
q = lambda g: P[P.G == g]
print('ДУЭЛИ И АИМ')
row('дуэлей / винрейт, %', lambda g: q(g).won.mean() * 100)
row('держал угол: доля дуэлей, %', lambda g: ((q(g).spd < HOLD) & (q(g).ospd > MOVE)).mean() * 100)
row('  винрейт, %', lambda g: q(g)[(q(g).spd < HOLD) & (q(g).ospd > MOVE)].won.mean() * 100)
row('выглядывал на держащего: доля, %', lambda g: ((q(g).spd > MOVE) & (q(g).ospd < HOLD)).mean() * 100)
row('  винрейт, %', lambda g: q(g)[(q(g).spd > MOVE) & (q(g).ospd < HOLD)].won.mean() * 100)
mv = lambda g: q(g)[(q(g).spd > MOVE) & (q(g).off0 < 15)]
row('на ходу прицел уже на враге, %', lambda g: (mv(g).pre == True).mean() * 100)
row('  винрейт дуэлей на ходу, %', lambda g: mv(g).won.mean() * 100)
row('винтовка против винтовки, винрейт %', lambda g: q(g)[(q(g).g_ == 'винтовка') & (q(g).og == 'винтовка')].won.mean() * 100)
row('против AWP/скаута, винрейт %', lambda g: q(g)[q(g).og == 'снайперка'].won.mean() * 100)
aim = pd.concat([res[m]['aim'].assign(G=grp(m)) for m in sel])
aim = aim[(aim.att.astype(str) == ME) & (aim.when == 'first') & aim.weapon.isin(list(RIFLES))]
row('1-й выстрел ниже головы (рифлы), °', lambda g: aim[aim.G == g].v.median(), '{:.2f}')
row('время до 1-го выстрела (медиана), мс', lambda g: q(g).react.median())
row('1-й выстрел попал, %', lambda g: (q(g).fs_hit == True).sum() / max(q(g).fs_hit.notna().sum(), 1) * 100)
dd = D[D.V == ME].assign(G=lambda x: x.m.map(grp))
row('проиграл дуэль, попав по врагу (не добил), %', lambda g: (dd[dd.G == g].v_hits > 0).mean() * 100)
print('  причины проигранных дуэлей (доля твоих смертей):')
for cat in ['аим', 'позиционка', 'оружие/HP', 'решение', 'прочее', 'на равных']:
    row(f'    {cat}, %', lambda g: (dd[dd.G == g].cat == cat).mean() * 100)

# конверсия
C = {}
for m in sel:
    f = os.path.join(FOLDER, f'conv_{m}.pkl')
    if os.path.exists(f) and (x := pickle.load(open(f, 'rb'))) is not None:
        for k, v in x.items(): C.setdefault(k, []).append(v.assign(G=grp(m)))
C = {k: pd.concat(v) for k, v in C.items()}
fr, fu, mt = C['frag'][C['frag'].P == ME], C['funnel'][C['funnel'].V == ME], C['mates'][C['mates'].P == ME]
print('КОНВЕРСИЯ')
row('после твоего 1-го фрага: раунд выигран, %', lambda g: (fr[fr.G == g].won == True).mean() * 100)
row('  отдал преимущество, %', lambda g: fr[fr.G == g].gave.mean() * 100)
row('твои смерти: тиммейты далеко, никто не увидел, %', lambda g: (fu[fu.G == g].st == 'тиммейты далеко, никто не увидел').mean() * 100)
row('  тиммейт рядом, но убийцу не увидел, %', lambda g: (fu[fu.G == g].st == 'тиммейт рядом, но убийцу не увидел').mean() * 100)
row('ты разменивающий: увидел убийцу тиммейта, %', lambda g: mt[mt.G == g].saw.mean() * 100)
row('  разменял, %', lambda g: mt[mt.G == g].kill.mean() * 100)

# вклад в раунды (модель — по всем матчам)
G = {m: (rounds_of(m, res[m]), res[m]) for m in M}
_, _, _, E, credit = model(G)
cr = credit[(credit.P == ME) & credit.m.isin(sel)].assign(G=lambda x: x.m.map(grp))
n_r = {g: sum(1 for m in by[g] for x in M[m]['rounds'] if x['sid'] == ME) for g in GR}
print('ВКЛАД В РАУНДЫ')
row('изменение шанса команды за раунд, п.п.', lambda g: cr[cr.G == g].gain.sum() / n_r[g] * 100, '{:+.1f}')
row('  средний фраг, п.п.', lambda g: cr[(cr.G == g) & (cr.typ == 'kill')].gain.mean() * 100, '{:+.1f}')
row('  средняя смерть, п.п.', lambda g: cr[(cr.G == g) & (cr.typ == 'death')].gain.mean() * 100, '{:+.1f}')

# позиции
print('ПОЗИЦИИ (твои дуэли на точке: убийств–смертей)')
pos = pd.concat([D[D.K == ME].assign(side=D.side_k, place=D.place_k, won=True), D[D.V == ME].assign(side=D.side_v, place=D.place_v, won=False)])
pos = pos.assign(G=pos.m.map(grp), spot=pos.side + ' ' + pos.place.astype(str))
for spot, x in sorted(pos.groupby('spot'), key=lambda kv: -len(kv[1]))[:10]:
    cell = lambda g: f"{int(x[x.G == g].won.sum())}–{int((~x[x.G == g].won).sum())}"
    print(f"  {spot:46} {cell('до'):>16} | {cell('после')}")
for side in ('CT', 'T'):
    for g in GR:
        top = []
        for m in by[g]:
            R = res[m]; fe = R['freeze']; t20 = {f + 20 * 64: i + 1 for i, f in enumerate(fe)}
            p_ = R['pos']; p_ = p_[p_.tick.isin(list(t20)) & (p_.steamid.astype(str) == ME) & (p_.health > 0)]
            top += [pl for t, pl in zip(p_.tick, p_.last_place_name) if M[m]['pl'][t20[t]].get(ME) == (3 if side == 'CT' else 2)]
        c = Counter(top)
        print(f"  {side} на 20-й секунде, {g:6} " + ', '.join(f"{p} {n / max(len(top), 1) * 100:.0f}%" for p, n in c.most_common(4)) + f"  (n={len(top)})")

# AWP
f = os.path.join(FOLDER, 'awp_rounds.pkl')
if os.path.exists(f):
    A = pickle.load(open(f, 'rb')); A = A[(A.sid == ME) & A.m.isin(sel)].assign(G=lambda x: x.m.map(grp))
    print('AWP')
    row('доля раундов с AWP, %', lambda g: A[A.G == g].awp.mean() * 100)
    row('  убийства / смерти за AWP-раунд', lambda g: A[(A.G == g) & A.awp].kills.mean(), '{:.2f}')
    row('    смерти за AWP-раунд', lambda g: A[(A.G == g) & A.awp].died.mean(), '{:.2f}')
