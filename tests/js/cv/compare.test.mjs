import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { LOWER_IS_BETTER, bestOf, compareBar, compareCheck, compareTable, comparisonRows } = await import('../../../src/activelabel/web/static/js/cv/compare.js');

setMessages({
  'tr.val': 'Validation', 'tr.test': 'Test', 'tr.metric.f1': 'F1', 'tr.metric.accuracy': 'Accuracy', 'md.size': 'Size', 'md.metric': 'Metric',
  'cmp.per_class': 'Per class · {metric}', 'cmp.zero_classes_one': '{n} class with 0', 'cmp.zero_classes_other': '{n} classes with 0',
  'cmp.choose': 'Compare {name}', 'cmp.bar': 'Comparison', 'cmp.select_all': 'Select all ({n})',
  'cmp.chosen_one': '{n} chosen', 'cmp.chosen_other': '{n} chosen', 'cmp.need_one_more': 'One more', 'cmp.need_two': 'Two or more', 'cmp.clear': 'Clear',
}, 'en');

const A = { n: 1, results: { val: { accuracy: 0.8, loss: 0.5, images: 10, per_class: { cat: { f1: 0.9 }, dog: { f1: 0 }, owl: { f1: 0 } } }, test: { accuracy: 0.7 } }, size: 2048 };
const B = { n: 2, results: { val: { accuracy: 0.9, loss: 0.4, images: 10, per_class: { cat: { f1: 0.8 }, dog: { f1: 0.1 } } } }, size: 1024 };

test('one row per metric, grouped by part; classes with 0 in every column are folded away', () => {
  const { groups, zeroClasses } = comparisonRows([A, B], (c) => c.results, (c) => c.size);
  assert.deepEqual(groups.map((g) => g.title), ['Validation', 'Test', 'Per class · F1', 'Size']);
  assert.deepEqual(groups[0].rows.map((r) => r.label), ['Accuracy', 'loss']);  // `images` is not a metric
  assert.deepEqual(groups[1].rows[0].values, [0.7, undefined]);
  assert.deepEqual(groups[2].rows.map((r) => r.label), ['cat', 'dog']);
  assert.deepEqual(zeroClasses, ['owl']);
});

test('the best of a row is the highest, the lowest for a loss or a size, and none for a tie', () => {
  assert.equal(bestOf({ values: [0.8, 0.9] }), 0.9);
  assert.equal(bestOf({ values: [0.5, 0.4], lower: true }), 0.4);
  assert.equal(bestOf({ values: [0.5, 0.5] }), null);
  assert.equal(bestOf({ values: [0.5, undefined] }), null);
});

test('errors such as RMSE and MAE are best when lowest, like the loss (#619)', () => {
  const R1 = { results: { val: { rmse: 2.5, mae: 1.5, r2: 0.6 } } };
  const R2 = { results: { val: { rmse: 3.0, mae: 1.0, r2: 0.7 } } };
  const rows = comparisonRows([R1, R2], (c) => c.results).groups[0].rows;
  assert.deepEqual(rows.map((r) => [r.label, r.lower, bestOf(r)]), [['rmse', true, 2.5], ['mae', true, 1.0], ['r2', false, 0.7]]);
  assert.deepEqual(LOWER_IS_BETTER, ['loss', 'mae', 'rmse']);
});

test('the table highlights the best value and lists the folded classes', () => {
  const node = compareTable({ id: 'x', title: 'Comparison', help: 'help', cols: [A, B], head: (c) => [`m${c.n}`], resultsOf: (c) => c.results });
  assert.deepEqual([...node.querySelectorAll('thead th')].map((th) => th.textContent), ['Metric', 'm1', 'm2']);
  assert.ok([...node.querySelectorAll('tr.cmp-group')].length >= 2);
  assert.deepEqual([...node.querySelectorAll('td.best')].map((td) => td.textContent), ['0.9', '0.4', '0.9', '0.1']);
  assert.match(node.querySelector('.cmp-zero summary').textContent, /1 class with 0/);
});

test('a card checkbox and the bar change the choice; the bar says how many are chosen', () => {
  const seen = [];
  const check = compareCheck({ id: 'c1', label: 'm1', checked: false, onChange: (on) => seen.push(on) });
  const input = check.querySelector('input');
  assert.equal(input.getAttribute('aria-label'), 'Compare m1');
  input.checked = true;
  input.dispatchEvent(new window.Event('change'));
  assert.deepEqual(seen, [true]);
  const bar = compareBar({ prefix: 'p', chosen: 1, total: 3, onAll: (on) => seen.push(`all:${on}`), onClear: () => seen.push('clear') });
  const all = bar.querySelector('#p-all');
  assert.equal(all.indeterminate, true);
  assert.match(bar.textContent, /Select all \(3\)/);
  assert.match(bar.textContent, /One more/);
  all.checked = true;
  all.dispatchEvent(new window.Event('change'));
  bar.querySelector('[data-act="compare-clear"]').click();
  assert.deepEqual(seen, [true, 'all:true', 'clear']);
  assert.match(compareBar({ prefix: 'q', chosen: 2, total: 3, onAll: () => {}, onClear: () => {} }).textContent, /2 chosen/);
  assert.equal(compareBar({ prefix: 'r', chosen: 0, total: 3, onAll: () => {}, onClear: () => {} }).querySelector('[data-act="compare-clear"]'), null);
});
