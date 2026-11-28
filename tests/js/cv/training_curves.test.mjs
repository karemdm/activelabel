import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node, Event: window.Event })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const {
  hasCurves, curveChoices, curveOf, prPoints, f1Points, bestF1, suggestionOf, tableRows, curvesBlock,
} = await import('../../../src/activelabel/web/static/js/cv/training_curves.js');

setMessages({
  'tr.curves_all': 'All classes (mean)', 'tr.curves_title': 'Curves', 'tr.curves_class': 'Class', 'tr.curves_table': 'Show as a table',
  'tr.confidence': 'Confidence', 'tr.metric.precision': 'Precision', 'tr.metric.recall': 'Recall', 'tr.metric.f1': 'F1',
  'tr.curves_best_f1': 'Best F1 {f1} at {c}', 'tr.curves_suggested': 'Suggested threshold {c}', 'tr.suggested_note': 'Suggested: {c}.', 'tr.curves_help': 'IoU {iou} ({match})', 'tr.curves_match_box': 'boxes', 'tr.curves_match_mask': 'masks',
}, 'en');

const grid = Array.from({ length: 11 }, (_, i) => i / 10);
const CURVES = {
  iou: 0.5,
  match: 'box',
  confidence: grid,
  classes: {
    cat: { precision: grid.map(() => 1), recall: grid.map((c) => 1 - c), f1: grid.map((c) => (2 * (1 - c)) / (2 - c)), support: 3 },
    dog: { precision: grid.map((c) => 0.5 + c / 2), recall: grid.map((c) => 1 - c), f1: grid.map((c) => (c === 0 ? 0.6 : c === 0.3 ? 0.8 : 0.5)), support: 9 },
  },
  all: { precision: grid.map(() => 0.9), recall: grid.map((c) => 1 - c), f1: grid.map(() => 0.4) },
};

test('older runs without curves draw nothing', () => {
  assert.equal(hasCurves(undefined), false);
  assert.equal(hasCurves({ confidence: [0, 1], classes: {} }), false);
  assert.equal(curvesBlock(null), null);
  assert.equal(curvesBlock({ map50: 0.5 }), null);
  assert.equal(hasCurves(CURVES), true);
});

test('the class picker offers the mean first, then the classes with the most objects', () => {
  assert.deepEqual(curveChoices(CURVES).map((c) => c.value), ['', 'dog', 'cat']);
  assert.equal(curveOf(CURVES, 'nope'), CURVES.all);
  assert.equal(curveOf(CURVES, 'cat'), CURVES.classes.cat);
});

test('the precision × recall points go from the most confident threshold to the least', () => {
  const pts = prPoints(CURVES, 'cat');
  assert.equal(pts.length, 11);
  assert.deepEqual(pts[0], [0, 1]);
  assert.deepEqual(pts[10], [1, 1]);
  assert.ok(pts.every((p, i) => i === 0 || p[0] >= pts[i - 1][0]));
});

test('the F1 × confidence points and the best F1', () => {
  assert.deepEqual(f1Points(CURVES, 'dog')[3], [0.3, 0.8]);
  assert.deepEqual(bestF1(CURVES, 'dog'), { confidence: 0.3, f1: 0.8 });
  assert.equal(bestF1({ ...CURVES, all: { precision: [], recall: [], f1: [] } }, ''), null);
});

test('the table lists every tenth confidence and the last one', () => {
  const fine = { ...CURVES, confidence: Array.from({ length: 101 }, (_, i) => i / 100) };
  fine.all = { precision: fine.confidence.map(() => 1), recall: fine.confidence, f1: fine.confidence };
  const rows = tableRows(fine, '');
  assert.equal(rows.length, 11);
  assert.deepEqual(rows.map((r) => r.confidence), grid.map((c) => Math.round(c * 100) / 100));
});

test('the block draws both charts, the table, and redraws for the picked class', () => {
  const block = curvesBlock({ curves: CURVES });
  document.body.appendChild(block);
  assert.equal(block.querySelectorAll('svg.tr-svg').length, 2);
  assert.equal(block.querySelectorAll('polyline').length, 2);   // only the mean
  assert.equal(block.querySelectorAll('details table tbody tr').length, 2);   // every tenth of an 11-point grid: 0 and 1
  assert.match(block.textContent, /Best F1 0\.4 at 0\.00/);
  const select = block.querySelector('select');
  select.value = 'dog';
  select.dispatchEvent(new window.Event('change'));
  assert.equal(block.querySelectorAll('polyline').length, 4);   // the class and the mean, in both charts
  assert.match(block.textContent, /Best F1 0\.8 at 0\.30/);
  assert.match(block.querySelector('table caption').textContent, /dog/);
  block.remove();
});

test('the F1 chart marks the threshold the validation suggested, for the mean or the class', () => {
  const part = { curves: CURVES, suggested_threshold: 0.35, suggested_thresholds: { dog: 0.3 } };
  assert.equal(suggestionOf(part, ''), 0.35);
  assert.equal(suggestionOf(part, 'dog'), 0.3);
  assert.equal(suggestionOf(part, 'cat'), null);
  const block = curvesBlock(part);
  assert.match(block.textContent, /Suggested: 0\.35\./);
  assert.match(block.textContent, /Suggested threshold 0\.35/);
  const select = block.querySelector('select');
  select.value = 'cat';
  select.dispatchEvent(new window.Event('change'));
  assert.match(block.textContent, /Best F1 1 at 0\.00/);   // no suggestion for cat: its best F1
});
