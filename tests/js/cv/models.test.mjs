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
    /* #650: car comes suggested as a chip; truck is added from the list, without Ctrl */
    const add = dlg.querySelector('#md-map-0');
    assert.equal(add.multiple, false);
    assert.deepEqual([...dlg.querySelectorAll('.md-map-row')[0].querySelectorAll('.chip')].map((c) => c.firstChild.textContent), ['car']);
    add.value = 'truck';
    add.dispatchEvent(new window.Event('change'));
    assert.equal(dlg.querySelectorAll('.md-map-row')[0].querySelectorAll('.chip').length, 1, 'choosing in the list adds nothing by itself');
    dlg.querySelector('.md-map-addbtn').click();
    assert.equal(document.activeElement.id, 'md-map-0', 'the focus goes back to the list of the class');
    /* bike: car is taken by car, so it is listed disabled with that class */
    const taken = [...dlg.querySelector('#md-map-1').options].find((o) => o.value === 'car');
    assert.equal(taken.disabled, true);
    dlg.querySelector('form').dispatchEvent(new window.Event('submit', { cancelable: true }));
    await new Promise((resolve) => setTimeout(resolve, 20));
  } finally {
    globalThis.fetch = real;
  }
  assert.deepEqual(posted[0].mapping, { car: 'car', truck: 'car' });
});

test('a category is taken out with its own button and a class is cleared at once (#650)', async () => {
  setMessages({ 'md.mapping_remove': 'Remove {category} from {cls}' }, 'en');
  const ctx = { project: { id: 9, task: 'detection' }, jobs: [], panels: {}, watchers: [], weightless: new Set() };
  const panel = modelsPanel({ ctx, projectId: 9, renderStepper: () => {}, showTab: () => {} });
  const catalog = { models: [{ id: 'm', label: 'M', note: '', box_map: 0.4, size_mb: 10, classes: ['car', 'truck', 'bus'], suggested: { car: 'car', truck: 'car', bus: 'car' } }], project_classes: ['car'] };
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const body = String(url).endsWith('public-catalog') ? catalog : { models: [] };
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try {
    document.body.appendChild(panel.node);
    [...panel.node.querySelectorAll('button')].find((b) => b.className === 'btn').click();
    await new Promise((resolve) => setTimeout(resolve, 20));
    const dlg = document.querySelector('dialog[open]');
    const names = () => [...dlg.querySelectorAll('.md-map-row .chip')].map((c) => c.firstChild.textContent);
    assert.deepEqual(names(), ['car', 'truck', 'bus']);
    dlg.querySelector('.chip-x[aria-label*="truck"]').click();
    assert.deepEqual(names(), ['car', 'bus']);
    assert.ok([...dlg.querySelector('#md-map-0').options].some((o) => o.value === 'truck' && !o.disabled), 'truck can be added again');
    dlg.querySelector('.md-map-clear').click();
    assert.deepEqual(names(), []);
    assert.ok(dlg.querySelector('.md-map-none'));
    assert.equal(dlg.querySelector('.md-map-clear'), null);
    dlg.close();
  } finally {
    globalThis.fetch = real;
    panel.node.remove();
  }
});
