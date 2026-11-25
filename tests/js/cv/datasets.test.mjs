import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
for (const [name, value] of Object.entries({
  window, document: window.document, Node: window.Node, localStorage: window.localStorage,
})) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { datasetsPanel, staleCount } = await import('../../../src/activelabel/web/static/js/cv/datasets.js');

setMessages({ 'step.datasets': 'Data split' }, 'en');

test('staleCount adds the images changed, added and removed since the version was made', () => {
  assert.equal(staleCount({}), 0);
  assert.equal(staleCount({ stale: { changed_images: 2, added: 1, removed: 3 } }), 6);
});

test('the Data split step counts the versions for the step bar', async () => {
  let stepper = 0;
  const ctx = { project: { id: 3, task: 'detection' }, jobs: [], panels: {}, watchers: [] };
  const panel = datasetsPanel({ ctx, projectId: 3, renderStepper: () => { stepper += 1; }, showTab: () => {} });
  assert.equal(panel.node.id, 'panel-datasets');
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(url);
    return { ok: true, status: 200, text: async () => JSON.stringify({ versions: [{ version: 1 }] }) };
  };
  try {
    await panel.count();
  } finally {
    globalThis.fetch = real;
  }
  assert.deepEqual(asked, ['/api/projects/3/datasets']);
  assert.equal(ctx.datasetCount, 1);
  assert.equal(stepper, 1);
});
