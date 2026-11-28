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

test('a model card offers Evaluate and shows the result across the row, with the way to the worst images (#611)', async () => {
  setMessages({
    'md.evaluate': 'Evaluate', 'md.eval_help': 'Help', 'md.eval_title': 'Evaluation of m{n}', 'md.eval_where': '{part} of version {version}',
    'md.eval_worst': 'Worst ({n})', 'md.eval_images_n_one': '{n} image', 'md.eval_images_n_other': '{n} images', 'ds.part_test': 'Test',
    'md.eval_started': 'Started', 'md.eval_queued': 'Queued', 'md.eval_cancel': 'Cancel', 'tr.metric.accuracy': 'Accuracy', 'tr.metric.macro_f1': 'Macro F1',
    'md.eval_classes': 'Classes', 'md.eval_class': 'Class', 'md.eval_support_images': 'Images', 'md.eval_open_fo': 'Open', 'md.eval_fo_hint': '{key}',
    'tr.metric.precision': 'P', 'tr.metric.recall': 'R', 'tr.metric.f1': 'F1', 'tr.confusion': 'Confusion', 'tr.confusion_help': 'Rows', 'tr.actual_vs_pred': 'R/P',
  }, 'en');
  const shown = [];
  const tabs = [];
  const ctx = { project: { id: 4, task: 'classification' }, jobs: [], watchers: [], weightless: new Set(), panels: { images: { showIds: (ids, n) => shown.push([ids, n]) } } };
  const panel = modelsPanel({ ctx, projectId: 4, renderStepper: () => {}, showTab: (step) => tabs.push(step) });
  const result = {
    kind: 'classification', version: 2, part: 'test', fold: null, images: 3, evaluated: '2026-11-27T10:00:00+00:00', unreadable: [], view: 'evaluation m1',
    metrics: { accuracy: 0.5, macro_f1: 0.4 }, classes: [{ class: 'a', precision: 1, recall: 1, f1: 1, support: 2 }],
    confusion: { classes: ['a'], other: false, missing: false, matrix: [[2]] }, worst: ['x', 'y'], fields: [],
  };
  const evaluations = { 1: { status: 'done', result, active: null }, 2: { active: null } };
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    asked.push([(init && init.method) || 'GET', String(url)]);
    const m = String(url).match(/models\/(\d+)\/evaluation/);
    let body = { models: [] };
    if (m && init && init.method === 'POST') body = { status: 'queued', job: 'j', active: Number(m[1]) };
    else if (m) body = evaluations[m[1]];
    else if (String(url).endsWith('/models')) body = { models: [{ number: 1, model: 'resnet18', source: 'run', run: 1, dataset: 2, start: 'transfer', metric: 'accuracy', size: 10 }, { number: 2, model: 'resnet18', source: 'run', run: 2, dataset: 2, start: 'transfer', metric: 'accuracy', size: 10 }], current: 1 };
    else if (String(url).includes('public-catalog')) body = { models: [] };
    else if (String(url).includes('/training/catalog')) body = { models: [] };
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try {
    document.body.appendChild(panel.node);
    panel.mount();
    await new Promise((resolve) => setTimeout(resolve, 40));
    const first = panel.node.querySelector('#model-1');
    const second = panel.node.querySelector('#model-2');
    assert.ok(first.classList.contains('has-eval') && !second.classList.contains('has-eval'));
    assert.match(first.textContent, /Evaluation of m1/);
    assert.equal(second.querySelector('[data-act="evaluate"]').textContent, 'Evaluate');
    first.querySelector('[data-act="eval-worst"]').click();
    assert.deepEqual(shown, [[['x', 'y'], 1]]);
    assert.deepEqual(tabs, ['images']);
    second.querySelector('[data-act="evaluate"]').click();
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.deepEqual(asked.find((a) => a[0] === 'POST'), ['POST', '/api/projects/4/models/2/evaluation']);
    assert.match(panel.node.querySelector('#model-2').textContent, /Queued/);
    assert.equal(panel.node.querySelector('#model-1 [data-act="evaluate"]').disabled, true, 'only one evaluation at a time');
  } finally {
    panel.node.hidden = true;  // the page stops polling
    globalThis.fetch = real;
    panel.node.remove();
  }
});

test('a model card says which classes were renamed since training (#521)', async () => {
  setMessages({ 'md.renamed': 'Renamed: {list}', 'tr.dataset_short': 'v{n}', 'md.from_run': 'run {n}' }, 'en');
  const ctx = { project: { id: 6, task: 'classification' }, jobs: [], panels: {}, watchers: [], weightless: new Set() };
  const panel = modelsPanel({ ctx, projectId: 6, renderStepper: () => {}, showTab: () => {} });
  const trained = { number: 1, model: 'resnet18', source: 'run', run: 1, dataset: 1, start: 'transfer', metric: 'accuracy', size: 10 };
  const models = [
    { ...trained, classes: ['felino', 'cão'], trained_classes: ['gato', 'cão'] },
    { ...trained, number: 2, classes: ['gato', 'cão'] },
  ];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const body = String(url).endsWith('/models') ? { models, current: 1 } : { models: [] };
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try {
    document.body.appendChild(panel.node);
    panel.mount();
    await new Promise((resolve) => setTimeout(resolve, 40));
    assert.match(panel.node.querySelector('#model-1 .md-renamed').textContent, /Renamed: gato → felino/);
    assert.equal(panel.node.querySelector('#model-2 .md-renamed'), null);
  } finally {
    panel.node.hidden = true;
    globalThis.fetch = real;
    panel.node.remove();
  }
});
