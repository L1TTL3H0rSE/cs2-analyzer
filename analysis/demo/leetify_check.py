# Сверка наших аим-метрик (duels.py) с Leetify по тем же матчам и игрокам: насколько наш «момент появления»
# и время до урона согласуются с их reaction_time и preaim. Метрики Leetify — под их именами и в их единицах.
#   python leetify_check.py [папка с res_*.pkl]
# Ответы Leetify кэшируются в data/leetify/ (в .gitignore): только для локальных тестов, не в продукт и не в git.
import glob, json, os, pickle, sys, time, urllib.request
import numpy as np, pandas as pd
from report import TICK

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
CACHE = os.path.join(ROOT, 'data', 'leetify')
API = 'https://api-public.cs-prod.leetify.com/v2/matches/faceit/'


def leetify_match(faceit_id):
    path = os.path.join(CACHE, f'{faceit_id}.json')
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        req = urllib.request.Request(API + faceit_id, headers={'Accept': 'application/json'})
        if os.environ.get('LEETIFY_API_KEY'): req.add_header('_leetify_key', os.environ['LEETIFY_API_KEY'])
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=30) as r: body = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 3:
                    print(f'  Leetify: {faceit_id} — HTTP {e.code}, пропускаю'); return {}
                time.sleep(10 * (attempt + 1))
        open(path, 'wb').write(body); time.sleep(1.5)  # без ключа лимиты жёстче
    return json.load(open(path, encoding='utf-8'))


def main(folder):
    D = pd.concat([x for f in sorted(glob.glob(os.path.join(folder, 'duels_*.pkl'))) if (x := pickle.load(open(f, 'rb'))) is not None], ignore_index=True)
    rows = []
    for m, g in D.groupby('m'):
        R = pickle.load(open(os.path.join(folder, f'res_{m}.pkl'), 'rb')); h = R['hurt']
        h = h.assign(att=h.attacker_steamid.astype(str), vic=h.user_steamid.astype(str))
        for side, other in (('k', 'V'), ('v', 'K')):
            for x in g.itertuples():
                P, O, fs, react = getattr(x, 'K' if side == 'k' else 'V'), getattr(x, other), getattr(x, f'{side}_fs'), getattr(x, f'{side}_react')
                s0 = fs - react * TICK / 1000 if not (np.isnan(fs) or np.isnan(react)) else np.nan
                fd = h[(h.att == P) & (h.vic == O) & (h.tick >= s0 - 32) & (h.tick <= x.tick)].tick.min() if not np.isnan(s0) else np.nan
                rows.append(dict(m=m, P=P, off0=getattr(x, f'{side}_off0'), react=react, ttd=(fd - s0) / TICK * 1000 if not np.isnan(fd) else np.nan))
    ours = pd.DataFrame(rows).groupby(['m', 'P']).agg(n=('off0', 'count'), off0=('off0', 'median'), react=('react', 'median'), ttd=('ttd', 'median')).reset_index()

    lee = []
    for m in ours.m.unique():
        j = leetify_match('1-' + m.split('_1-')[1])
        for s in j.get('stats', []):
            lee.append(dict(m=m, P=str(s['steam64_id']), reaction_time=s.get('reaction_time'), preaim=s.get('preaim'),
                            accuracy_enemy_spotted=s.get('accuracy_enemy_spotted'), after_0811=m[:10] >= '2026-08-11'))
    J = ours.merge(pd.DataFrame(lee), on=['m', 'P'])
    J = J[(J.n >= 5) & J.reaction_time.notna()]
    print(f"игроко-матчей с 5+ нашими дуэлями и данными Leetify: {len(J)} (матчей {J.m.nunique()})")
    for lab, x in (('все', J), ('до 11.08 (старые хитбоксы Leetify)', J[~J.after_0811]), ('после 11.08', J[J.after_0811])):
        if len(x) < 10: print(f"  {lab}: мало данных (n={len(x)})"); continue
        c = lambda a, b: x[[a, b]].corr(method='spearman').iloc[0, 1]
        print(f"  {lab:34} n={len(x):3}  Spearman: наше время до урона ~ reaction_time {c('ttd', 'reaction_time'):+.2f};"
              f"  наше время до 1-го выстрела ~ reaction_time {c('react', 'reaction_time'):+.2f};  наше отклонение при появлении ~ preaim {c('off0', 'preaim'):+.2f}")
    me = J[J.P == '76561198105241814']
    print(f"\nты по матчам (медианы): наше время до урона {me.ttd.median():.0f} мс, Leetify reaction_time {me.reaction_time.median():.3f} с;"
          f"  наше отклонение {me.off0.median():.1f}°, Leetify preaim {me.preaim.median():.1f}°")
    print(f"лобби: наше время до урона {J.ttd.median():.0f} мс, Leetify reaction_time {J.reaction_time.median():.3f} с;  наше отклонение {J.off0.median():.1f}°, Leetify preaim {J.preaim.median():.1f}°")
    rk = lambda col: J.groupby('m')[col].rank(pct=True)
    J = J.assign(r_lee=rk('reaction_time'), r_our=rk('ttd'), p_lee=rk('preaim'), p_our=rk('off0'))
    m2 = J[J.P == '76561198105241814']
    print(f"твой ранг в лобби матча (0 — лучший, 1 — худший), медиана: время до урона — наш {m2.r_our.median():.2f}, Leetify {m2.r_lee.median():.2f};"
          f"  прицел при появлении — наш {m2.p_our.median():.2f}, Leetify {m2.p_lee.median():.2f}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', '..', 'demos', 'parsed'))
