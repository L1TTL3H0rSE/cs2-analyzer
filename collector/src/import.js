// Перенести вручную скачанные демки FACEIT (1-<matchId>-1-1.dem.zst) в demos/ с именами как у сборщика и отметить в state.json.
//   npm run import -- <папка>   (по умолчанию ~/Downloads/demos)
import { readFileSync, writeFileSync, readdirSync, renameSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';
import { launchContext } from './browser.js';
import { listMatchesBrowser, getMyGuidBrowser, getDemoResourceBrowser } from './faceit-browser.js';
import { config } from './config.js';

const IN = process.argv[2] || join(homedir(), 'Downloads', 'demos');
const files = readdirSync(IN).filter((f) => /^1-[0-9a-f-]{36}-1-1\.dem\.zst$/.test(f));
if (!files.length) { console.log('нет демок FACEIT в', IN); process.exit(0); }
const state = JSON.parse(readFileSync(config.stateFile, 'utf8'));
const { context, page } = await launchContext({ visible: false });
try {
  const me = await getMyGuidBrowser(page);
  const times = Object.fromEntries((await listMatchesBrowser(page, 100)).map((m) => [m.matchId, m.finishedAt]));
  for (const f of files) {
    const matchId = f.replace(/-1-1\.dem\.zst$/, '');
    if (state.done[matchId]) { console.log(matchId, 'уже есть в state.json — пропуск'); continue; }
    const info = await getDemoResourceBrowser(page, matchId, me);
    const fin = times[matchId];
    const date = fin ? new Date(fin * 1000).toISOString().slice(0, 10) : 'unknown';
    const map = (info.map || 'map').replace(/[^a-z0-9_]/gi, '');
    const file = join(config.demoDir, `${date}_${map}_${matchId}.dem.zst`);
    if (existsSync(file)) { console.log(matchId, 'файл уже есть:', file); continue; }
    renameSync(join(IN, f), file);
    state.done[matchId] = { file, map: info.map, won: info.won, score: info.score, finishedAt: fin ? new Date(fin * 1000).toISOString() : null,
                            at: new Date().toISOString(), source: 'manual' };
    console.log(matchId, fin ? new Date(fin * 1000).toISOString() : 'нет в истории', info.map, info.score, info.won ? 'победа' : 'поражение', '→', file);
  }
  writeFileSync(config.stateFile, JSON.stringify(state, null, 2));
} finally {
  await context.close();
}
