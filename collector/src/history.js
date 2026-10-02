// История матчей FACEIT (ELO, K/D, ADR) и комнаты матчей (ELO всех игроков) → data/faceit/ (локальный кэш, в .gitignore).
//   npm run history   — без Turnstile; уже скачанные комнаты пропускает
import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { launchContext } from './browser.js';
import { getMyGuidBrowser } from './faceit-browser.js';
import { ROOT } from './config.js';
const OUT = join(ROOT, '..', 'data', 'faceit');
mkdirSync(OUT + '/matches', { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const { context, page } = await launchContext({ visible: false });
try {
  const me = await getMyGuidBrowser(page);
  const get = (u) => page.evaluate(async (u) => { const r = await fetch(u, { credentials: 'include' }); return { s: r.status, j: r.ok ? await r.json() : null }; }, u);
  const st = await get(`https://www.faceit.com/api/stats/v1/stats/time/users/${me}/games/cs2?size=100`);
  writeFileSync(OUT + '/stats.json', JSON.stringify(st.j));
  let n = 0;
  for (const x of st.j) {
    const f = `${OUT}/matches/${x.matchId}.json`;
    if (existsSync(f)) continue;
    let r = await get(`https://www.faceit.com/api/match/v2/match/${x.matchId}`);
    for (let a = 0; r.s === 429 && a < 4; a++) { await sleep(15000 * (a + 1)); r = await get(`https://www.faceit.com/api/match/v2/match/${x.matchId}`); }
    if (r.j) { writeFileSync(f, JSON.stringify(r.j.payload)); n++; } else console.log(x.matchId, 'HTTP', r.s);
    await sleep(1500);
  }
  console.log(`матчей в истории: ${st.j.length}; скачано комнат: ${n}`);
} finally {
  await context.close();
}
