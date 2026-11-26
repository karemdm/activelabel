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

test('the public-model dialog lets several categories name one class', async () => {
  const proto = window.HTMLDialogElement.prototype;
  if (!proto.showModal) proto.showModal = function showModal() { this.setAttribute('open', ''); };
  if (!proto.close) proto.close = function close() { this.removeAttribute('open'); this.dispatchEvent(new window.Event('close')); };
  const ctx = { project: { id: 8, task: 'detection' }, jobs: [], panels: {}, watchers: [], weightless: new Set() };
  const panel = modelsPanel({ ctx, projectId: 8, renderStepper: () => {}, showTab: () => {} });
  const catalog = { models: [{ id: 'm', label: 'M', note: '', box_map: 0.4, size_mb: 10, classes: ['car', 'truck', 'bus'], suggested: { car: 'car' } }], project_classes: ['car', 'bike'] };
  const posted = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    if (init && init.method === 'POST') posted.push(JSON.parse(init.body));
    const body = String(url).endsWith('public-catalog') ? catalog : { models: [], number: 1 };
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try {
    document.body.appendChild(panel.node);
    const button = [...panel.node.querySelectorAll('button')].find((b) => b.className === 'btn');
    button.click();
    await new Promise((resolve) => setTimeout(resolve, 20));
    const dlg = document.querySelector('dialog');
    const select = dlg.querySelector('#md-map-car');
    assert.equal(select.multiple, true);
    for (const o of select.options) o.selected = o.value !== 'bus';  // car (suggested) and truck
    select.dispatchEvent(new window.Event('change'));
    dlg.querySelector('form').dispatchEvent(new window.Event('submit', { cancelable: true }));
    await new Promise((resolve) => setTimeout(resolve, 20));
  } finally {
    globalThis.fetch = real;
  }
  assert.deepEqual(posted[0].mapping, { car: 'car', truck: 'car' });
});
