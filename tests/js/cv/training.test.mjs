import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
for (const [name, value] of Object.entries({
  window, document: window.document, Node: window.Node, localStorage: window.localStorage, Event: window.Event,
})) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const proto = window.HTMLDialogElement.prototype;
if (!proto.showModal) proto.showModal = function showModal() { this.setAttribute('open', ''); };
if (!proto.close) proto.close = function close() { this.removeAttribute('open'); this.dispatchEvent(new window.Event('close')); };
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { trainingPanel } = await import('../../../src/activelabel/web/static/js/cv/training.js');

setMessages({
  'task.detection': 'Detection', 'tr.family_licence': 'needs its {name} licence', 'tr.licence_open': 'Accept {name}…',
  'tr.family.torchvision': 'torchvision', 'tr.family.timm': 'timm', 'tr.family.ultralytics': 'Ultralytics',
}, 'en');

const SPEC = { name: 'epochs', kind: 'int', default: 10, min: 1, max: 500, help: 'Epochs' };
const model = (id, family) => ({
  id, family, label: id, task_kind: 'detection', supports_pretrained: true, metrics: ['map'], default_metric: 'map', params: [SPEC], packages: [],
});
const ANSWERS = {
  '/api/training/catalog': { models: [model('frcnn', 'torchvision'), model('retina', 'torchvision'), model('yolo', 'ultralytics')], common: [], metrics: {} },
  '/api/training/augmentations': { kinds: {} },
  '/api/projects/5/datasets': { versions: [{ version: 1, created: '2026-01-01T00:00:00', options: {} }] },
  '/api/training/environment': { devices: ['auto', 'cpu'] },
  '/api/projects/5/runs': { runs: [] },
  '/api/training/licences': { licences: [{ id: 'ultralytics', name: 'AGPL-3.0', package: 'ultralytics', url: 'https://www.gnu.org/licenses/agpl-3.0.html', accepted: false }] },
};

/** Calls fn with the API answering from ANSWERS by path. */
async function withApi(fn) {
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const body = ANSWERS[url.split('?')[0]];
    return { ok: body !== undefined, status: body !== undefined ? 200 : 404, text: async () => JSON.stringify(body ?? {}) };
  };
  try {
    return await fn();
  } finally {
    globalThis.fetch = real;
  }
}

const ctx = { project: { id: 5, task: 'detection', classes: ['a'] }, jobs: [], panels: {}, watchers: [], weightless: new Set() };

test('the Training step counts the runs for the step bar', async () => {
  let stepper = 0;
  const panel = trainingPanel({ ctx, projectId: 5, renderStepper: () => { stepper += 1; } });
  assert.equal(panel.node.id, 'panel-training');
  await withApi(() => panel.count());
  assert.equal(stepper, 1);
});

test('the new-run dialog asks for the family first; only families with a model for the type are listed (#601)', async () => {
  const panel = trainingPanel({ ctx, projectId: 5, renderStepper: () => {} });
  document.body.appendChild(panel.node);
  await withApi(async () => {
    panel.startWith(1);
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
  const family = document.getElementById('tr-family');
  assert.ok(family, 'the family list is in the dialog');
  const options = Object.fromEntries([...family.options].map((o) => [o.value, o]));
  assert.equal(options.torchvision.disabled, false);
  assert.equal(options.timm, undefined, 'timm has no detection model here, so it is not listed');
  assert.equal(options.ultralytics.disabled, true);
  assert.match(options.ultralytics.textContent, /needs its AGPL-3.0 licence/);
  assert.equal(document.querySelector('[data-act="licence-ultralytics"]').textContent, 'Accept AGPL-3.0…');
  assert.deepEqual([...document.getElementById('tr-model').options].map((o) => o.value), ['frcnn', 'retina']);
  document.querySelector('dialog[open]').close();
  panel.node.remove();
});

test('a failed run says why and warns in the interface language, and offers Resume when it has a checkpoint (#560)', async () => {
  setMessages({
    'error.training_failed': 'The training stopped with an error.', 'tr.warn.batch_halved': 'Batch {from} → {to}.', 'tr.resume': 'Resume',
  }, 'en');
  ANSWERS['/api/projects/5/runs'] = { runs: [
    { number: 2, status: 'failed', model: 'frcnn', mode: 'single', fold: null, checkpoint_epoch: 3, error: 'raw', error_code: 'training_failed', error_params: {},
      warnings: [{ code: 'batch_halved', params: { from: 16, to: 8 }, message: 'raw warning' }, { code: '', params: {}, message: 'old text' }] },
    { number: 1, status: 'cancelled', model: 'frcnn', mode: 'grid', checkpoint_epoch: 2, warnings: [] },
  ] };
  const panel = trainingPanel({ ctx, projectId: 5, renderStepper: () => {} });
  document.body.appendChild(panel.node);
  await withApi(async () => {
    panel.mount();
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
  const failed = panel.node.querySelector('#run-2');
  assert.equal(failed.querySelector('.tr-fail').textContent, 'The training stopped with an error.');
  assert.deepEqual([...failed.querySelectorAll('.tr-warnings li')].map((li) => li.textContent), ['Batch 16 → 8.', 'old text']);
  assert.equal(failed.querySelector('[data-act="resume"]').textContent, 'Resume');
  assert.equal(panel.node.querySelector('#run-1 [data-act="resume"]'), null);  // a grid resumes later
  ANSWERS['/api/projects/5/runs'] = { runs: [] };
  panel.node.remove();
});

test('a finished run with quality alerts shows a badge on the card and the list in the details (#148)', async () => {
  setMessages({
    'tr.alerts_badge_one': '{n} quality alert', 'tr.alerts_badge_other': '{n} quality alerts',
    'tr.alert.metric_zero': 'The main metric ({metric}) is zero.', 'tr.alert.class_zero': 'Class {class} was never found.',
  }, 'en');
  const alerts = [{ code: 'metric_zero', params: { metric: 'map50' } }, { code: 'class_zero', params: { class: 'car' } }];
  ANSWERS['/api/projects/5/runs'] = { runs: [
    { number: 3, status: 'done', model: 'frcnn', mode: 'single', fold: null, alerts, warnings: [] },
    { number: 2, status: 'done', model: 'frcnn', mode: 'single', fold: null, alerts: [], warnings: [] },
    { number: 1, status: 'failed', model: 'frcnn', mode: 'single', fold: null, alerts, warnings: [] },
  ] };
  const panel = trainingPanel({ ctx, projectId: 5, renderStepper: () => {} });
  document.body.appendChild(panel.node);
  await withApi(async () => {
    panel.mount();
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
  const badge = panel.node.querySelector('#run-3 .tr-badge.alert');
  assert.equal(badge.textContent, '2 quality alerts');
  assert.match(badge.title, /Class car was never found\./);
  assert.equal(panel.node.querySelector('#run-2 .tr-badge.alert'), null);
  assert.equal(panel.node.querySelector('#run-1 .tr-badge.alert'), null);  // only a finished run has them
  ANSWERS['/api/projects/5/runs'] = { runs: [] };
  panel.node.remove();
});
