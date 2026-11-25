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
const { modelsPanel } = await import('../../../src/activelabel/web/static/js/cv/models.js');

setMessages({ 'step.models': 'Models' }, 'en');

test('the Models step counts the saved models for the step bar', async () => {
  let stepper = 0;
  const ctx = { project: { id: 8, task: 'detection' }, jobs: [], panels: {}, watchers: [], weightless: new Set() };
  const panel = modelsPanel({ ctx, projectId: 8, renderStepper: () => { stepper += 1; }, showTab: () => {} });
  assert.equal(panel.node.id, 'panel-models');
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(url);
    return { ok: true, status: 200, text: async () => JSON.stringify({ models: [{ number: 1 }, { number: 2 }], current: 2 }) };
  };
  try {
    await panel.count();
  } finally {
    globalThis.fetch = real;
  }
  assert.deepEqual(asked, ['/api/projects/8/models']);
  assert.equal(ctx.modelCount, 2);
  assert.equal(ctx.currentModel, 2);
  assert.equal(stepper, 1);
});
