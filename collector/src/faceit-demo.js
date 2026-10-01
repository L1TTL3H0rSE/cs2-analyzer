// Turns a FACEIT demo resource_url into a downloadable presigned URL, exactly
// the way FACEIT's own website does it: from within a logged-in faceit.com
// page, mint a Cloudflare Turnstile token, then POST to the download-url API.
// (This is the same flow the open-source CSNADES extension uses.)

const DOWNLOAD_URL_API = 'https://www.faceit.com/api/download/v2/demos/download-url';

/**
 * @param {import('playwright').Page} page  a page in a logged-in FACEIT profile
 * @param {string} matchId
 * @param {string} resourceUrl  demo_url[0] from the Data API
 * @returns {Promise<{downloadUrl?: string, error?: string, status?: number, raw?: string, siteKey?: string}>}
 */
export async function getPresignedUrl(page, matchId, resourceUrl) {
  // Load the match room so FACEIT's demo-download code + Turnstile are present.
  await page.goto(`https://www.faceit.com/en/cs2/room/${matchId}`, {
    waitUntil: 'domcontentloaded',
  });
  await page.waitForTimeout(3500);

  return page.evaluate(
    async ({ resourceUrl, apiUrl }) => {
      const SITEKEY_RE = /0x4AAA[A-Za-z0-9_-]{6,}/;

      async function ensureTurnstile() {
        if (window.turnstile) return true;
        await new Promise((resolve, reject) => {
          const s = document.createElement('script');
          s.src =
            'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
          s.onload = resolve;
          s.onerror = reject;
          document.head.appendChild(s);
        }).catch(() => {});
        for (let i = 0; i < 60 && !window.turnstile; i++) {
          await new Promise((r) => setTimeout(r, 100));
        }
        return Boolean(window.turnstile);
      }

      async function findSiteKey() {
        const inHtml = document.documentElement.innerHTML.match(SITEKEY_RE);
        if (inHtml) return inHtml[0];
        const scripts = performance
          .getEntriesByType('resource')
          .map((e) => e.name)
          .filter((n) => n.startsWith(location.origin) && n.includes('.js'));
        for (const url of scripts) {
          try {
            const txt = await (await fetch(url)).text();
            const m = txt.match(SITEKEY_RE);
            if (m) return m[0];
          } catch {
            /* ignore */
          }
        }
        return null;
      }

      if (!(await ensureTurnstile())) return { error: 'turnstile-script-failed' };
      const siteKey = await findSiteKey();
      if (!siteKey) return { error: 'sitekey-not-found' };

      const container = document.createElement('div');
      container.style.position = 'fixed';
      container.style.left = '-9999px';
      container.style.top = '0';
      document.body.appendChild(container);

      const token = await new Promise((resolve) => {
        let settled = false;
        const finish = (v) => {
          if (!settled) {
            settled = true;
            resolve(v);
          }
        };
        try {
          window.turnstile.render(container, {
            sitekey: siteKey,
            action: 'matchroomFinished_downloadDemos',
            callback: (t) => finish(t),
            'error-callback': () => finish(null),
          });
        } catch {
          finish(null);
        }
        setTimeout(() => finish(null), 25000);
      });

      if (!token) return { error: 'turnstile-no-token', siteKey };

      const res = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ resource_url: resourceUrl, captcha_token: token }),
      });
      const raw = await res.text();
      let json;
      try {
        json = JSON.parse(raw);
      } catch {
        /* ignore */
      }
      return {
        status: res.status,
        downloadUrl: json?.payload?.download_url,
        siteKey,
        raw: json?.payload?.download_url ? undefined : raw.slice(0, 300),
      };
    },
    { resourceUrl, apiUrl: DOWNLOAD_URL_API },
  );
}
