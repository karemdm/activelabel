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
const { datasetsPanel, staleCount, sharesOf, wholeNumber, kfoldSumText, realSplitText, MAX_SEED, MAX_NOTE } = await import('../../../src/activelabel/web/static/js/cv/datasets.js');

setMessages({ 'step.datasets': 'Data split' }, 'en');

test('staleCount adds the images changed, added and removed since the version was made', () => {
  assert.equal(staleCount({}), 0);
  assert.equal(staleCount({ stale: { changed_images: 2, added: 1, removed: 3 } }), 6);
  assert.equal(staleCount({ stale: { changed_images: 0, added: 0, removed: 0, excluded: 2, restored: 1 } }), 3);   // #636
});

test('the Data split step counts the versions for the step bar', async () => {
  let stepper = 0;
  const ctx = { project: { id: 3, task: 'detection' }, jobs: [], panels: {}, watchers: [] };
  const panel = datasetsPanel({ ctx, projectId: 3, renderStepper: () => { stepper += 1; }, showTab: () => {} });
  assert.equal(panel.node.id, 'panel-datasets');
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(url);
    return { ok: true, status: 200, text: async () => JSON.stringify({ versions: [{ version: 1 }] }) };
  };
  try {
    await panel.count();
  } finally {
    globalThis.fetch = real;
  }
  assert.deepEqual(asked, ['/api/projects/3/datasets']);
  assert.equal(ctx.datasetCount, 1);
  assert.equal(stepper, 1);
});

test('the shares of the parts are whole percentages that add up to 100 (#636)', () => {
  assert.deepEqual(sharesOf([1, 0, 99]), [1, 0, 99]);
  assert.deepEqual(sharesOf([2, 1, 0]), [67, 33, 0]);
  const five = sharesOf([18, 18, 18, 18, 18, 10]);
  assert.equal(five.reduce((a, n) => a + n, 0), 100);
  assert.deepEqual(sharesOf([0, 0]), [0, 0]);
  const folds = sharesOf([2, 2, 2, 2, 2, 110]);
  assert.equal(new Set(folds.slice(0, 5)).size, 1, 'equal parts show the same percentage');
});

test('wholeNumber refuses decimals instead of cutting them (#544)', () => {
  assert.equal(wholeNumber('70'), 70);
  assert.equal(wholeNumber(' -3 '), -3);
  for (const bad of ['70.5', '1.7', '1e2', '', 'abc', '0x10']) assert.ok(Number.isNaN(wholeNumber(bad)), bad);
});

test('the seed and note limits match the server', () => {
  assert.equal(MAX_SEED, 2 ** 32 - 1);
  assert.equal(MAX_NOTE, 1000);
});

test('the K-fold sum line uses the singular for one fold and has a text without K (#544)', () => {
  setMessages({
    'ds.kfold_sum_one': '{n} fold shares {train}% · test {test}%',
    'ds.kfold_sum_other': '{n} folds share {train}% · test {test}%',
    'ds.kfold_sum_nok': 'Folds share {train}% · test {test}%',
  }, 'en');
  assert.equal(kfoldSumText(1, 90, 10), '1 fold shares 90% · test 10%');
  assert.equal(kfoldSumText(5, 90, 10), '5 folds share 90% · test 10%');
  assert.equal(kfoldSumText(NaN, 90, 10), 'Folds share 90% · test 10%');
});

test('realSplitText shows the shares a version really got, not for K-fold (#544)', () => {
  assert.equal(realSplitText({ sizes: { train: 36, val: 17, test: 7 } }), '60/28/12 %');
  assert.equal(realSplitText({ sizes: { test: 7, folds: [10, 10] } }), '');
  assert.equal(realSplitText({}), '');
});
