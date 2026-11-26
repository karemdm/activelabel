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
const { trainingPanel, scheduleClash, startHelpText, listWithoutSeparator, listExample } = await import('../../../src/activelabel/web/static/js/cv/training.js');

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
  /* #632: the models without an architecture of their own are one architecture each */
  assert.deepEqual([...document.getElementById('tr-arch').options].map((o) => o.value), ['frcnn', 'retina']);
  document.querySelector('dialog[open]').close();
  panel.node.remove();
});

test('a very large model on the CPU is warned about, by device and by model (#668)', async () => {
  setMessages({ 'tr.cpu_large': '{model} has {params} M parameters: slow on the CPU.' }, 'en');
  const saved = ANSWERS['/api/training/catalog'];
  const big = { ...model('regnet', 'torchvision'), arch: 'RegNetY', variant: '128GF', params_m: 644.8 };
  const small = { ...model('resnet', 'torchvision'), arch: 'ResNet', variant: '18', params_m: 11.7 };
  ANSWERS['/api/training/catalog'] = { models: [big, small], common: [], metrics: {} };
  const panel = trainingPanel({ ctx, projectId: 5, renderStepper: () => {} });
  document.body.appendChild(panel.node);
  try {
    await withApi(async () => {
      panel.startWith(1);
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    const warning = document.getElementById('tr-cpu-warning');
    assert.equal(warning.hidden, false, 'automatic on a computer without a GPU is the CPU');
    assert.match(warning.textContent, /RegNetY 128GF has 644.8 M parameters/);
    const arch = document.getElementById('tr-arch');
    arch.value = 'ResNet';
    arch.dispatchEvent(new window.Event('change'));
    assert.equal(warning.hidden, true, 'a small model: nothing to say');
  } finally {
    ANSWERS['/api/training/catalog'] = saved;
    document.querySelector('dialog[open]') && document.querySelector('dialog[open]').close();
    panel.node.remove();
  }
});

test('another size of the same architecture keeps the edited values and says nothing (#312, #632)', async () => {
  const saved = ANSWERS['/api/training/catalog'];
  const sized = (variant) => ({ ...model(`net${variant}`, 'torchvision'), arch: 'Net', variant });
  ANSWERS['/api/training/catalog'] = { models: [sized('n'), sized('s'), sized('m')], common: [], metrics: {} };
  const panel = trainingPanel({ ctx, projectId: 5, renderStepper: () => {} });
  document.body.appendChild(panel.node);
  try {
    await withApi(async () => {
      panel.startWith(1);
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    const epochs = document.getElementById('tr-p-epochs');
    epochs.value = '30';
    epochs.dispatchEvent(new window.Event('input'));
    const m = document.querySelector('input[name="tr-size"][value="netm"]');
    m.checked = true;
    m.dispatchEvent(new window.Event('change', { bubbles: true }));
    assert.equal(document.getElementById('tr-p-epochs').value, '30');
    assert.equal(document.getElementById('tr-model-changes').textContent, '');
    document.querySelector('dialog[open]').close();
  } finally {
    ANSWERS['/api/training/catalog'] = saved;
    panel.node.remove();
  }
});

test('a value out of range keeps Start off and says which setting to fix (#635)', async () => {
  setMessages({ 'tr.blocked_fields': 'To start, fix: {fields}.', 'tr.p.epochs': 'Epochs' }, 'en');
  const panel = trainingPanel({ ctx, projectId: 5, renderStepper: () => {} });
  document.body.appendChild(panel.node);
  await withApi(async () => {
    panel.startWith(1);
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
  const epochs = document.getElementById('tr-p-epochs');
  const why = document.querySelector('dialog[open] .tr-blocked');
  assert.ok(why.hidden, 'nothing to fix at first');
  epochs.value = '0';
  epochs.dispatchEvent(new window.Event('input'));
  const start = document.querySelector('dialog[open] button[type="submit"]');
  assert.equal(start.disabled, true);
  assert.equal(why.hidden, false);
  assert.equal(why.textContent, 'To start, fix: Epochs.');
  epochs.value = '3';
  epochs.dispatchEvent(new window.Event('input'));
  assert.equal(start.disabled, false);
  assert.ok(why.hidden);
  document.querySelector('dialog[open]').close();
  panel.node.remove();
});

test('a size that must be a multiple of 32 is checked in the dialog like on the server (#635)', async () => {
  setMessages({ 'tr.blocked_fields': 'To start, fix: {fields}.', 'tr.p.image_size': 'Image size', 'tr.err_multiple': '{v} is not a multiple of {n}.' }, 'en');
  const models = ANSWERS['/api/training/catalog'].models;
  const before = models[0].params;
  models[0].params = [SPEC, { name: 'image_size', kind: 'int', default: 256, min: 128, max: 1024, multiple: 32, help: 'Size' }];
  const panel = trainingPanel({ ctx, projectId: 5, renderStepper: () => {} });
  document.body.appendChild(panel.node);
  try {
    await withApi(async () => {
      panel.startWith(1);
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    const size = document.getElementById('tr-p-image_size');
    size.value = '300';
    size.dispatchEvent(new window.Event('input'));
    assert.equal(document.getElementById('tr-p-image_size-err').textContent, '300 is not a multiple of 32.');
    assert.equal(document.querySelector('dialog[open] button[type="submit"]').disabled, true);
    size.value = '320';
    size.dispatchEvent(new window.Event('input'));
    assert.equal(document.querySelector('dialog[open] button[type="submit"]').disabled, false);
  } finally {
    models[0].params = before;
    document.querySelector('dialog[open]')?.close();
    panel.node.remove();
  }
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

test('scheduleClash refuses a warmup or a patience longer than the training, over every grid value (#546)', () => {
  assert.equal(scheduleClash({ epochs: [10], warmup_epochs: [30] }).field, 'warmup_epochs');
  assert.equal(scheduleClash({ epochs: [3, 20], warmup_epochs: [5] }).epochs, 3);
  assert.equal(scheduleClash({ epochs: [3], patience: [50] }).field, 'patience');
  assert.equal(scheduleClash({ epochs: [3], patience: [3] }).key, 'error.patience_exceeds_epochs');
  assert.equal(scheduleClash({ epochs: [10], warmup_epochs: [10], patience: [0] }), null);
  assert.equal(scheduleClash({ warmup_epochs: [5] }), null);
});

test('startHelpText says only what is true of the model for each way of starting (#653)', () => {
  setMessages({
    'tr.start_transfer_help': 'T', 'tr.start_transfer_rcnn': 'R', 'tr.start_transfer_yolo': 'Y',
    'tr.start_finetune_help': 'F', 'tr.start_finetune_factor': 'B', 'tr.start_finetune_layers': 'L', 'tr.start_finetune_yolo': 'FY',
    'tr.start_scratch_help': 'S',
  }, 'en');
  const rcnn = { family: 'torchvision', arch: 'Faster R-CNN', params: [{ name: 'trainable_backbone_layers' }] };
  const classifier = { family: 'torchvision', arch: 'ResNet', params: [] };
  const yolo = { family: 'ultralytics', arch: 'YOLO', params: [] };
  assert.equal(startHelpText(rcnn, 'transfer'), 'T R');
  assert.equal(startHelpText(rcnn, 'finetune'), 'F B L');
  assert.equal(startHelpText(classifier, 'finetune'), 'F B');
  assert.equal(startHelpText(classifier, 'transfer'), 'T');
  assert.equal(startHelpText(yolo, 'transfer'), 'T Y');
  assert.equal(startHelpText(yolo, 'finetune'), 'F FY');
  assert.equal(startHelpText(rcnn, 'scratch'), 'S');
});

test('listWithoutSeparator and listExample follow the decimal sign of the language (#653)', () => {
  setMessages({}, 'pt-BR');
  assert.equal(listWithoutSeparator('0,1,0,2'), true);
  assert.equal(listWithoutSeparator('0,1'), false);
  assert.equal(listWithoutSeparator('0,1; 0,2'), false);
  assert.equal(listExample(), '0,1; 0,2');
  setMessages({}, 'en');
  assert.equal(listWithoutSeparator('0,1,0,2'), false);
  assert.equal(listExample(), '0.1, 0.2');
});
