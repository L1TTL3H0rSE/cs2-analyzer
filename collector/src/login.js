// One-time (occasional) login helper.
// Opens a real browser window using the persistent profile and navigates to
// FACEIT. YOU log in manually (I never handle your credentials). Once you're
// logged in, press Enter in this terminal to save the session and close.
//
// Run: npm run login
import readline from 'node:readline/promises';
import { stdin, stdout } from 'node:process';
import { launchContext } from './browser.js';

const { context, page } = await launchContext({ visible: true });

await page.goto('https://www.faceit.com/en', { waitUntil: 'domcontentloaded' });

console.log('\n=== FACEIT login ===');
console.log('A browser window is open. Log into FACEIT there (if not already).');
console.log('When your profile shows you are logged in, come back here.\n');

const rl = readline.createInterface({ input: stdin, output: stdout });
await rl.question('Press Enter once you are logged in to save the session... ');
rl.close();

await context.close();
console.log('Session saved to the profile. You can close this window.');
