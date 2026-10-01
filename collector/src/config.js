import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

export const config = {
  // 'msedge' | 'chrome' — реальный браузер, не встроенный Chromium Playwright
  browserChannel: process.env.BROWSER_CHANNEL || 'msedge',
  // Отдельный профиль сборщика (НЕ твой основной профиль браузера)
  profileDir: process.env.PROFILE_DIR || join(ROOT, '.browser-profile'),
  demoDir: process.env.DEMO_DIR || join(ROOT, '..', 'demos'),
  stateFile: join(ROOT, 'state.json'),
  historySize: Number(process.env.HISTORY_SIZE || 20),
};
