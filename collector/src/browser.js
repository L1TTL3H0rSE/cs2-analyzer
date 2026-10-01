import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { config } from './config.js';

/**
 * Kill any browser process still holding OUR profile (e.g. a lingering login
 * instance). Matches on our profile dir in the command line, so it never
 * touches the user's normal Edge.
 */
function killProfileBrowsers() {
  const dir = config.profileDir;
  const ps =
    `Get-CimInstance Win32_Process -Filter "Name='msedge.exe' OR Name='chrome.exe'" ` +
    `| Where-Object { $_.CommandLine -like '*${dir}*' } ` +
    `| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }`;
  try {
    execFileSync(
      'powershell',
      ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', ps],
      { stdio: 'ignore', timeout: 15000 },
    );
  } catch {
    /* best effort */
  }
}

/**
 * Launch a persistent, logged-in browser context using the real Edge/Chrome
 * binary (best Cloudflare/Turnstile reputation — far better than headless).
 * The profile dir keeps you logged into FACEIT between runs.
 *
 * @param {{ visible?: boolean }} opts  visible=true for the one-time login;
 *   otherwise the window is minimized so scheduled runs stay out of your way.
 */
export async function launchContext({ visible = false } = {}) {
  mkdirSync(config.profileDir, { recursive: true });
  killProfileBrowsers();

  // 'chromium' (or empty) => Playwright's bundled browser (no channel).
  const useChannel =
    config.browserChannel && config.browserChannel !== 'chromium'
      ? { channel: config.browserChannel }
      : {};

  const context = await chromium.launchPersistentContext(config.profileDir, {
    ...useChannel,
    headless: false,
    viewport: null,
    args: ['--disable-blink-features=AutomationControlled'],
  });

  const page = context.pages()[0] ?? (await context.newPage());

  if (!visible) {
    await minimizeWindow(context, page).catch(() => {});
  }
  return { context, page };
}

/** Minimize the OS window via CDP so unattended runs don't grab the screen. */
async function minimizeWindow(context, page) {
  const session = await context.newCDPSession(page);
  const { windowId } = await session.send('Browser.getWindowForTarget');
  await session.send('Browser.setWindowBounds', {
    windowId,
    bounds: { windowState: 'minimized' },
  });
}
