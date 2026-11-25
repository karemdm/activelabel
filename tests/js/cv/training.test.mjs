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
  'task.detection': 'Detection', 'tr.family_none': 'no model for {type} projects', 'tr.family_license': 'needs its licence',
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

test('the new-run dialog asks for the family first; families that cannot train here are disabled with why', async () => {
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
  assert.equal(options.timm.disabled, true);
  assert.match(options.timm.textContent, /no model for detection projects/);
  assert.equal(options.ultralytics.disabled, true);
  assert.match(options.ultralytics.textContent, /needs its licence/);
  assert.deepEqual([...document.getElementById('tr-model').options].map((o) => o.value), ['frcnn', 'retina']);
  document.querySelector('dialog[open]').close();
  panel.node.remove();
});
