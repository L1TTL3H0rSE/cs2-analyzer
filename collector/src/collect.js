// Скачивает демки новых FACEIT-матчей в config.demoDir.
//   node src/collect.js              — последние HISTORY_SIZE матчей, пропуская уже скачанные
//   node src/collect.js --match 1-…  — один конкретный матч (для теста)
//   node src/collect.js --visible    — не сворачивать окно браузера (отладка)
import { mkdirSync, existsSync, readFileSync, writeFileSync, createWriteStream, renameSync } from 'node:fs';
import { join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { config } from './config.js';
import { launchContext } from './browser.js';
import { listMatchesBrowser, getMyGuidBrowser, getDemoResourceBrowser } from './faceit-browser.js';
import { getPresignedUrl } from './faceit-demo.js';

const args = process.argv.slice(2);
const onlyMatch = args.includes('--match') ? args[args.indexOf('--match') + 1] : null;
const visible = args.includes('--visible');

mkdirSync(config.demoDir, { recursive: true });
const state = existsSync(config.stateFile) ? JSON.parse(readFileSync(config.stateFile, 'utf8')) : { done: {} };
const save = () => writeFileSync(config.stateFile, JSON.stringify(state, null, 2));
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

const { context, page } = await launchContext({ visible });
let failures = 0;
try {
  const me = await getMyGuidBrowser(page);
  if (!me) throw new Error('Нет сессии FACEIT в профиле сборщика. Запусти: npm run login');

  const matches = onlyMatch
    ? [{ matchId: onlyMatch, finishedAt: 0 }]
    : await listMatchesBrowser(page, config.historySize);
  log(`матчей в истории: ${matches.length}`);

  for (const { matchId, finishedAt } of matches) {
    if (state.done[matchId] && !onlyMatch) continue;
    try {
      const info = await getDemoResourceBrowser(page, matchId, me);
      if (!info.resourceUrl) { log(matchId, 'нет demo_url:', info.status ?? info.error); continue; }

      const p = await getPresignedUrl(page, matchId, info.resourceUrl);
      if (!p.downloadUrl) { failures++; log(matchId, 'presign не удался', p.status, p.error ?? p.raw ?? ''); continue; }

      const ext = new URL(p.downloadUrl).pathname.endsWith('.zst') ? '.dem.zst' : '.dem.gz';
      const date = finishedAt ? new Date(finishedAt * 1000).toISOString().slice(0, 10) : 'unknown';
      const map = (info.map || 'map').replace(/[^a-z0-9_]/gi, '');
      const file = join(config.demoDir, `${date}_${map}_${matchId}${ext}`);

      const res = await fetch(p.downloadUrl);
      if (!res.ok) { failures++; log(matchId, 'скачивание HTTP', res.status); continue; }
      // .part → rename: недокачанный файл не выглядит готовой демкой
      await pipeline(Readable.fromWeb(res.body), createWriteStream(file + '.part'));
      renameSync(file + '.part', file);
      state.done[matchId] = { file, map: info.map, won: info.won, score: info.score, at: new Date().toISOString() };
      save();
      log(matchId, '→', file);
    } catch (e) {
      failures++;
      log(matchId, 'ошибка:', e.message);
    }
  }
} finally {
  await context.close();
}
process.exit(failures ? 1 : 0);
