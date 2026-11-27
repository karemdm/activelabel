import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node, localStorage: window.localStorage })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const ev = await import('../../../src/activelabel/web/static/js/cv/model_evaluation.js');

setMessages({
  'md.evaluate': 'Evaluate',
  'md.eval_again': 'Evaluate again',
  'md.eval_help': 'Help',
  'md.eval_other': 'm{n} is being evaluated',
  'md.eval_cancel': 'Cancel',
  'md.eval_queued': 'Queued',
  'md.eval_environment': 'Preparing',
  'md.eval_predicting': 'Evaluating {done} of {total}',
  'md.eval_fiftyone': 'Computing in FiftyOne',
  'md.installing': 'Installing {pkg}',
  'md.eval_started': 'Started',
  'md.eval_failed': 'Failed: {error}',
  'md.interrupted': 'Interrupted',
  'md.eval_cancelled': 'Cancelled',
  'md.eval_title': 'Evaluation of m{n}',
  'md.eval_where': '{part} of version {version}',
  'md.eval_images_n_one': '{n} image',
  'md.eval_images_n_other': '{n} images',
  'md.eval_unreadable_n_one': '{n} unreadable',
  'md.eval_unreadable_n_other': '{n} unreadable',
  'md.eval_counts': '{tp} found, {fp} wrong, {fn} missed',
  'md.eval_classes': 'Metrics of each class',
  'md.eval_class': 'Class',
  'md.eval_support_images': 'Images',
  'md.eval_other_classes': '(other)',
  'md.eval_none': '(none)',
  'md.eval_none_help': 'About none',
  'md.eval_worst': 'See the worst images ({n})',
  'md.eval_no_worst': 'No errors',
  'md.eval_open_fo': 'Open in FiftyOne',
  'md.eval_fo_starting': 'Opening',
  'md.eval_fo_hint': 'Choose {key}',
  'tr.metric.accuracy': 'Accuracy',
  'tr.metric.macro_f1': 'Macro F1',
  'tr.metric.precision': 'Precision',
  'tr.metric.recall': 'Recall',
  'tr.metric.f1': 'F1',
  'tr.support.objects': 'Objects',
  'tr.confusion': 'Confusion matrix',
  'tr.confusion_help': 'Rows are real',
  'tr.actual_vs_pred': 'Real \\ Predicted',
  'ds.part_test': 'Test',
  'ds.part_val': 'Val',
  'ds.fold': 'Fold {n}',
}, 'en');

const tick = () => new Promise((r) => setTimeout(r, 0));
const text = (node) => node.textContent;

const DETECTION = {
  kind: 'detection', version: 3, part: 'test', fold: null, images: 12, evaluated: '2026-11-27T10:00:00+00:00', unreadable: [], view: 'evaluation m1',
  metrics: { precision: 0.8, recall: 0.5, f1: 0.615385, map: 0.4, tp: 8, fp: 2, fn: 8 },
  classes: [{ class: 'car', precision: 0.9, recall: 0.5, f1: 0.64, support: 10 }, { class: 'bus', precision: 0.5, recall: 0.5, f1: 0.5, support: 6 }],
  confusion: { classes: ['car', 'bus'], other: false, missing: true, matrix: [[5, 0, 5], [1, 3, 2], [2, 0, 0]] },
  worst: ['a', 'b'], fields: [],
};
const CLASSIFICATION = {
  kind: 'classification', version: 1, part: 'val', fold: 1, images: 4, evaluated: '2026-11-27T10:00:00+00:00', unreadable: ['x'], view: 'evaluation m2',
  metrics: { accuracy: 0.75, macro_f1: 0.7 }, classes: [{ class: 'a', precision: 1, recall: 1, f1: 1, support: 2 }],
  confusion: { classes: ['a'], other: true, missing: false, matrix: [[2, 0], [0, 0]] }, worst: [], fields: [],
};

test('the status line follows the phase of the evaluation', () => {
  assert.equal(ev.statusText({ status: 'queued' }), 'Queued');
  assert.equal(ev.statusText({ status: 'done' }), '');
  assert.equal(ev.statusText({ status: 'running', phase: 'environment' }), 'Preparing');
  assert.equal(ev.statusText({ status: 'running', phase: 'environment', message: 'train_env_install:torch' }), 'Installing torch');
  assert.equal(ev.statusText({ status: 'running', phase: 'predicting', done: 3, total: 12 }), 'Evaluating 3 of 12');
  assert.equal(ev.statusText({ status: 'running', phase: 'predicting', total: 0 }), 'Preparing');
  assert.equal(ev.statusText({ status: 'running', phase: 'fiftyone' }), 'Computing in FiftyOne');
  assert.ok(ev.running({ status: 'queued' }) && ev.running({ status: 'running' }) && !ev.running({ status: 'done' }) && !ev.running(undefined));
});

test('the part is named with its version, a fold with its number', () => {
  assert.equal(ev.whereText(DETECTION), 'Test of version 3');
  assert.equal(ev.whereText(CLASSIFICATION), 'Fold 2 of version 1');
  assert.equal(ev.whereText({ part: 'test', fold: 0, version: 2 }), 'Test of version 2');
});

test('the main metrics are those of the kind of evaluation', () => {
  assert.deepEqual(ev.mainMetrics(DETECTION).map(([label]) => label), ['Precision', 'Recall', 'F1', 'mAP']);
  assert.deepEqual(ev.mainMetrics(CLASSIFICATION), [['Accuracy', 0.75], ['Macro F1', 0.7]]);
});

test('the class table lists each class, and its last column counts objects or images', () => {
  const objects = ev.classTable(DETECTION.classes, 'detection');
  assert.equal(objects.querySelectorAll('tbody tr').length, 2);
  assert.match(text(objects.querySelector('thead')), /Objects/);
  assert.match(text(objects.querySelector('tbody tr')), /car0.90.50.6410/);
  assert.match(text(ev.classTable(CLASSIFICATION.classes, 'classification').querySelector('thead')), /Images/);
  assert.equal(ev.classTable([], 'detection'), null);
});

test('the confusion matrix names the extra row and column and shades by the share of the row', () => {
  const table = ev.confusionTable(DETECTION.confusion);
  const heads = [...table.querySelectorAll('thead th')].map((h) => h.textContent);
  assert.deepEqual(heads, ['Real \\ Predicted', 'car', 'bus', '(none)']);
  assert.equal(table.querySelectorAll('tbody tr').length, 3);
  assert.match(text(table), /About none/);
  const cells = table.querySelectorAll('tbody tr')[0].querySelectorAll('td');
  assert.ok(cells[0].classList.contains('diag') && cells[1].classList.contains('zero'));
  assert.equal(cells[0].style.getPropertyValue('--share'), '28%');
  assert.equal(cells[1].textContent, '·');
  const other = ev.confusionTable(CLASSIFICATION.confusion);
  assert.deepEqual([...other.querySelectorAll('thead th')].map((h) => h.textContent).slice(1), ['a', '(other)']);
  assert.doesNotMatch(text(other), /About none/);
  assert.equal(ev.confusionTable({ classes: [], matrix: [] }), null);
});

/** Runs fn with the API answering by `answer(method, url, body)`; returns what was asked. */
async function withApi(answer, fn) {
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url, opts) => {
    const method = (opts && opts.method) || 'GET';
    const body = opts && opts.body ? JSON.parse(opts.body) : undefined;
    asked.push([method, url, body]);
    const out = answer(method, url, body);
    if (out && out.status === 404) return { ok: false, status: 404, text: async () => JSON.stringify({ detail: 'x', code: 'c' }) };
    if (out && out.fail) return { ok: false, status: 409, text: async () => JSON.stringify({ detail: 'x', code: 'evaluation_running' }) };
    return { ok: true, status: 200, text: async () => JSON.stringify(out) };
  };
  try { await fn(); } finally { globalThis.fetch = real; }
  return asked;
}

test('the store reads each model once, then only the running ones, and tells which one is active', async () => {
  const states = { 1: { status: 'running', active: 1, job: 'j' }, 2: { active: 1 } };
  let changes = 0;
  const store = ev.evaluationStore(5, () => { changes += 1; });
  let asked = await withApi((m, url) => states[url.match(/models\/(\d+)/)[1]], () => store.sync([1, 2]));
  assert.deepEqual(asked.map((a) => a[1]).sort(), ['/api/projects/5/models/1/evaluation', '/api/projects/5/models/2/evaluation']);
  assert.equal(store.active(), 1);
  assert.ok(store.busy());
  states[1] = { status: 'done', active: null, result: DETECTION };
  asked = await withApi((m, url) => states[url.match(/models\/(\d+)/)[1]], () => store.sync([1, 2]));
  assert.deepEqual(asked.map((a) => a[1]), ['/api/projects/5/models/1/evaluation']);  // only the running one
  assert.ok(!store.busy() && store.active() === null);
  assert.equal(store.get(1).result.images, 12);
  assert.deepEqual(store.get(7), {});
  const before = store.signature();
  await withApi(() => ({}), () => store.sync([2]));  // model 1 was deleted
  assert.notEqual(store.signature(), before);
  assert.deepEqual(store.get(1), {});
  assert.equal(changes, 0);
});

test('a model that is not found is left without a state', async () => {
  const store = ev.evaluationStore(5, () => {});
  await withApi(() => ({ status: 404 }), () => store.sync([9]));
  assert.deepEqual(store.get(9), {});
});

test('starting posts the request, marks the model active and redraws; cancelling posts the cancel', async () => {
  let changes = 0;
  const store = ev.evaluationStore(5, () => { changes += 1; });
  let asked = await withApi(() => ({ status: 'queued', job: 'j', active: 1 }), () => store.start(1));
  assert.deepEqual(asked[0], ['POST', '/api/projects/5/models/1/evaluation', { device: 'auto' }]);
  assert.equal(store.get(1).status, 'queued');
  assert.equal(store.active(), 1);
  assert.equal(changes, 1);
  asked = await withApi(() => ({ status: 'cancelled', active: null }), () => store.cancel(1));
  assert.deepEqual(asked[0].slice(0, 2), ['POST', '/api/projects/5/models/1/evaluation/cancel']);
  assert.equal(store.get(1).status, 'cancelled');
  assert.equal(store.active(), null);
  assert.equal(changes, 2);
});

test('a refused start shows the error and reads the state again', async () => {
  const store = ev.evaluationStore(5, () => {});
  const asked = await withApi((m) => (m === 'POST' ? { fail: true } : { status: 'done', active: null }), () => store.start(1));
  assert.deepEqual(asked.map((a) => a[0]), ['POST', 'GET']);
  assert.equal(store.get(1).status, 'done');
});

test('the card offers Evaluate, then Evaluate again, and the progress with Cancel while it runs', async () => {
  const states = { 1: { active: 3 }, 2: { status: 'done', result: DETECTION, active: 3 }, 3: { status: 'running', phase: 'predicting', done: 1, total: 4, active: 3 } };
  const store = ev.evaluationStore(5, () => {});
  await withApi((m, url) => states[url.match(/models\/(\d+)/)[1]], () => store.sync([1, 2, 3]));
  assert.equal(store.active(), 3);
  const idle = ev.evaluateControl({ number: 1 }, store);
  assert.equal(idle.textContent, 'Evaluate');
  assert.equal(idle.disabled, true);          // m3 is running
  assert.equal(idle.title, 'm3 is being evaluated');
  assert.equal(ev.evaluateControl({ number: 2 }, store).textContent, 'Evaluate again');
  const busy = ev.evaluateControl({ number: 3 }, store);
  assert.match(busy.textContent, /Evaluating 1 of 4/);
  assert.equal(busy.getAttribute('role'), 'status');
  const asked = await withApi(() => ({ status: 'cancelled', active: null }), async () => { busy.querySelector('[data-act="eval-cancel"]').click(); await tick(); });
  assert.deepEqual(asked[0].slice(0, 2), ['POST', '/api/projects/5/models/3/evaluation/cancel']);
  const free = ev.evaluateControl({ number: 1 }, store);
  assert.equal(free.disabled, false);
  assert.equal(free.title, 'Help');
  const started = await withApi(() => ({ status: 'queued', active: 1 }), async () => { free.click(); await tick(); });
  assert.equal(started[0][1], '/api/projects/5/models/1/evaluation');
});

test('a finished evaluation shows its numbers, tables and the two ways on', () => {
  const calls = [];
  const ways = { worst: (ids, n) => calls.push(['worst', ids, n]), openApp: (r, n) => calls.push(['open', r.view, n]) };
  const node = ev.evaluationResult({ number: 1 }, { status: 'done', result: DETECTION }, ways);
  assert.match(text(node), /Evaluation of m1/);
  assert.match(text(node), /Test of version 3 · 12 images/);
  assert.equal(node.querySelectorAll('.tr-stat').length, 4);
  assert.match(text(node.querySelector('.tr-stats')), /Precision0\.8Recall0\.5F10\.615mAP0\.4/);
  assert.match(text(node), /8 found, 2 wrong, 8 missed/);
  assert.ok(node.querySelector('.md-eval-classes') && node.querySelector('.md-eval-matrix'));
  assert.match(text(node), /Choose eval_m1/);
  node.querySelector('[data-act="eval-worst"]').click();
  node.querySelector('[data-act="eval-open"]').click();
  assert.deepEqual(calls, [['worst', ['a', 'b'], 1], ['open', 'evaluation m1', 1]]);
  assert.match(text(node.querySelector('[data-act="eval-worst"]')), /See the worst images \(2\)/);
});

test('a classification with no errors has nothing to look at, and unreadable images are told', () => {
  const node = ev.evaluationResult({ number: 2 }, { status: 'done', result: CLASSIFICATION }, {});
  assert.equal(node.querySelector('[data-act="eval-worst"]'), null);
  assert.match(text(node), /No errors/);
  assert.match(text(node), /Fold 2 of version 1 · 4 images .* 1 unreadable/);
  assert.equal(node.querySelectorAll('.tr-stat').length, 2);
  assert.doesNotMatch(text(node), /found, .* wrong/);
});

test('a failed or cancelled evaluation says so and keeps the earlier result', () => {
  const failed = ev.evaluationResult({ number: 1 }, { status: 'failed', error: 'interrupted', result: DETECTION }, {});
  assert.match(text(failed), /Failed: Interrupted/);
  assert.match(text(failed), /Evaluation of m1/);
  const other = ev.evaluationResult({ number: 1 }, { status: 'failed', error: 'boom' }, {});
  assert.match(text(other), /Failed: boom/);
  assert.equal(other.querySelector('.md-eval-grid'), null);
  assert.match(text(ev.evaluationResult({ number: 1 }, { status: 'cancelled' }, {})), /Cancelled/);
  assert.equal(ev.evaluationResult({ number: 1 }, {}, {}), null);
  assert.equal(ev.evaluationResult({ number: 1 }, { status: 'running' }, {}), null);
});

test('opening the App asks for the saved view and registers the tab for the job', async () => {
  const ctx = { openWhenDone: new Map() };
  const started = [];
  const page = { ctx, projectId: 5, jobStarted: async (job) => started.push(job.id) };
  const asked = await withApi(() => ({ id: 'job1', status: 'queued' }), () => ev.openApp(page, DETECTION));
  assert.deepEqual(asked[0], ['POST', '/api/projects/5/fiftyone', { view: 'evaluation m1' }]);
  assert.ok(ctx.openWhenDone.has('job1'));
  assert.deepEqual(started, ['job1']);
});
