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

  const iso = (sec) => (sec ? new Date(sec * 1000).toISOString() : null);
  for (const { matchId, finishedAt } of matches) {
    if (state.done[matchId] && !onlyMatch) {
      state.done[matchId].finishedAt ??= iso(finishedAt); // время матча для старых записей
      continue;
    }
    try {
      const info = await getDemoResourceBrowser(page, matchId, me);
      if (!info.resourceUrl) { log(matchId, 'нет demo_url:', info.status ?? info.error); continue; }

      const date = finishedAt ? new Date(finishedAt * 1000).toISOString().slice(0, 10) : 'unknown';
      const map = (info.map || 'map').replace(/[^a-z0-9_]/gi, '');
      let file;

      // связь бывает рвётся молча: 60 с без данных — обрыв и до 3 попыток заново;
      // подписанная ссылка живёт недолго (повтор по старой после обрыва — HTTP 401), поэтому на каждую попытку новая
      let ok = false;
      for (let attempt = 1; attempt <= 3 && !ok; attempt++) {
        const p = await getPresignedUrl(page, matchId, info.resourceUrl);
        if (!p.downloadUrl) { log(matchId, 'presign не удался', p.status, p.error ?? p.raw ?? ''); break; }
        file = join(config.demoDir, `${date}_${map}_${matchId}${new URL(p.downloadUrl).pathname.endsWith('.zst') ? '.dem.zst' : '.dem.gz'}`);
        const ac = new AbortController(); let timer;
        const kick = () => { clearTimeout(timer); timer = setTimeout(() => ac.abort(new Error('нет данных 60 с')), 60_000); };
        try {
          kick();
          const res = await fetch(p.downloadUrl, { signal: ac.signal });
          if (!res.ok) { log(matchId, 'скачивание HTTP', res.status); break; }
          const body = Readable.fromWeb(res.body).on('data', kick);
          // .part → rename: недокачанный файл не выглядит готовой демкой
          await pipeline(body, createWriteStream(file + '.part'));
          ok = true;
        } catch (e) { log(matchId, `попытка ${attempt}:`, e.message); } finally { clearTimeout(timer); }
      }
      if (!ok) { failures++; continue; }
      renameSync(file + '.part', file);
      state.done[matchId] = { file, map: info.map, won: info.won, score: info.score, finishedAt: iso(finishedAt), at: new Date().toISOString() };
      save();
      log(matchId, '→', file);
    } catch (e) {
      failures++;
      log(matchId, 'ошибка:', e.message);
    }
  }
  save();
} finally {
  await context.close();
}
process.exit(failures ? 1 : 0);
