import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { preannotateBlock } = await import('../../../src/activelabel/web/static/js/cv/preannotate.js');

setMessages({ 'pre.none_supported': 'No model can pre-annotate this project yet' }, 'en');

/** Runs fn with the API answering `body`; returns the URLs asked. */
async function withApi(body, fn) {
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(url);
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try {
    await fn();
  } finally {
    globalThis.fetch = real;
  }
  return asked;
}

test('the tab reads the pre-annotations of the project and says when no model fits', async () => {
  let changed = 0;
  const pre = preannotateBlock({ projectId: 4 }, { changed: () => { changed += 1; }, reviewCvat: () => {}, reviewFiftyone: () => {} });
  const asked = await withApi({ models: [{ number: 1, supported: false }], runs: [] }, () => pre.load());
  assert.deepEqual(asked, ['/api/projects/4/preannotations']);
  assert.match(pre.node.textContent, /No model can pre-annotate this project yet/);
  assert.equal(changed, 0);
  assert.equal(typeof pre.choose, 'function');
});
