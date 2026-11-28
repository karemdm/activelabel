import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { modelOptionText, preannotateBlock } = await import('../../../src/activelabel/web/static/js/cv/preannotate.js');

setMessages({ 'pre.none_supported': 'No model can pre-annotate this project yet', 'pre.regression_later': 'Pre-annotation for regression is coming later.' }, 'en');

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
  assert.deepEqual(asked.sort(), ['/api/projects/4/preannotations', '/api/training/environment']);  // the devices are read once
  assert.match(pre.node.textContent, /No model can pre-annotate this project yet/);
  assert.equal(changed, 0);
  assert.equal(typeof pre.choose, 'function');
});

test('a regression project says pre-annotation is coming later, with no form (#619)', async () => {
  const pre = preannotateBlock({ projectId: 4, ctx: { project: { task: 'regression' } } }, { changed: () => {}, reviewCvat: () => {}, reviewFiftyone: () => {} });
  await withApi({ models: [], runs: [] }, () => pre.load());
  assert.match(pre.node.textContent, /coming later/);
  assert.equal(pre.node.querySelector('select, button'), null);
});

test('the confidence field counts its steps from a multiple of the step, and "no class" is said once (#663)', async () => {
  setMessages({ 'pre.classes_none': 'Choose at least one class', 'pre.classes_none_btn': 'None' }, 'en');
  const model = {
    number: 1, supported: true, current: true, label: 'Faster R-CNN', kind: 'detection',
    options: [{ name: 'confidence', type: 'float', default: 0.25, min: 0.001, max: 0.999, step: 0.01, level: 'basic' }],
    classes: [{ name: 'cat', in_project: true }, { name: 'dog', in_project: true }],
  };
  const pre = preannotateBlock({ projectId: 5 }, { changed: () => {}, reviewCvat: () => {}, reviewFiftyone: () => {} });
  document.body.appendChild(pre.node);
  await withApi({ models: [model], runs: [] }, () => pre.load());
  const confidence = pre.node.querySelector('#pre-opt-confidence');
  assert.equal(confidence.getAttribute('min'), '0');   // 0.25 and 0.7 are whole steps from 0, not from 0.001
  const none = [...pre.node.querySelectorAll('button')].find((b) => b.textContent === 'None');
  none.click();
  const said = [...pre.node.querySelectorAll('.pre-note')].filter((p) => p.textContent.includes('Choose at least one class'));
  assert.equal(said.length, 1);
  assert.equal(said[0].id, 'pre-set-problem');   // the disabled Start button points to it
  pre.node.remove();
});

test('the model list is short and the result of the chosen model is said under it (#338)', async () => {
  setMessages({ 'md.current': 'Current', 'pre.model_result': 'Result of this model: {metric} {value}', 'tr.metric.map50_95': 'mAP50-95' }, 'en');
  const model = { number: 1, supported: true, current: true, label: 'Faster R-CNN MobileNetV3 FPN', kind: 'detection', metric: 'map50_95', value: 0.357, options: [], classes: [] };
  assert.equal(modelOptionText(model), 'm1 (Current) · Faster R-CNN MobileNetV3 FPN');
  assert.equal(modelOptionText({ number: 2, label: 'YOLO26 m' }), 'm2 · YOLO26 m');
  const pre = preannotateBlock({ projectId: 5 }, { changed: () => {}, reviewCvat: () => {}, reviewFiftyone: () => {} });
  document.body.appendChild(pre.node);
  await withApi({ models: [model], runs: [] }, () => pre.load());
  assert.equal(pre.node.querySelector('#pre-model option').textContent, 'm1 (Current) · Faster R-CNN MobileNetV3 FPN');
  assert.equal(pre.node.querySelector('#pre-model-detail').textContent, 'Result of this model: mAP50-95 0.357');
  assert.match(pre.node.querySelector('#pre-model').getAttribute('aria-describedby'), /pre-model-detail/);
  pre.node.remove();
});

test('#342: without a saved model the empty state offers Models only when the catalogue has public models for the project', async () => {
  setMessages({ 'pre.no_models': 'Save a model or add a public one', 'pre.no_models_train': 'Train a model and save it', 'pre.go_models': 'Go to Models', 'md.go_training': 'Go to Training' }, 'en');
  const wait = () => new Promise((resolve) => setTimeout(resolve, 10));
  for (const [catalogue, text, link] of [[[], 'Train a model and save it', '#training'], [[{ id: 'p1' }], 'Save a model or add a public one', '#models']]) {
    const real = globalThis.fetch;
    globalThis.fetch = async (url) => {
      const body = String(url).endsWith('/models/public-catalog') ? { models: catalogue } : { models: [], runs: [] };
      return { ok: true, status: 200, text: async () => JSON.stringify(body) };
    };
    try {
      const pre = preannotateBlock({ projectId: 6 }, { changed: () => {}, reviewCvat: () => {}, reviewFiftyone: () => {} });
      await pre.load();
      await wait();
      assert.equal(pre.node.querySelector('p').textContent, text);
      assert.equal(pre.node.querySelector('a').getAttribute('href'), link);
    } finally {
      globalThis.fetch = real;
    }
  }
});
