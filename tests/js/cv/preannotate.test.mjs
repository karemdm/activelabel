import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { preannotateBlock } = await import('../../../src/activelabel/web/static/js/cv/preannotate.js');

setMessages({ 'pre.none_supported': 'No model can pre-annotate this project yet' }, 'en');

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
