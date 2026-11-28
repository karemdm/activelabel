import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body><div id="toasts"></div></body></html>', { url: 'http://localhost/' });
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { exportFormats, exportRow, exportUrl } = await import('../../../src/activelabel/web/static/js/cv/export.js');

setMessages({
  'ds.export': 'Export', 'ds.export_aria': 'Export version {n} as {format}', 'ds.exporting': 'Preparing {format} of v{n}',
  'error.export_regression': 'No regression export.',
}, 'en');

test('each task has the formats that can hold it', () => {
  assert.deepEqual(exportFormats('detection'), ['yolo', 'coco']);
  assert.deepEqual(exportFormats('keypoints'), ['yolo', 'coco']);
  assert.deepEqual(exportFormats('classification'), ['yolo']);   // COCO has no label for a whole image
  assert.deepEqual(exportFormats('regression'), []);
  assert.equal(exportUrl(3, 2, 'coco'), '/api/projects/3/datasets/2/export?format=coco');
});

test('a regression version shows no export row; the others one button per format', () => {
  assert.equal(exportRow(1, { version: 1, type: 'regression' }), null);
  const row = exportRow(1, { version: 4, type: 'detection' });
  const buttons = [...row.querySelectorAll('button')];
  assert.deepEqual(buttons.map((b) => b.dataset.format), ['yolo', 'coco']);
  assert.equal(buttons[1].getAttribute('aria-label'), 'Export version 4 as COCO');
  assert.ok(exportRow(1, { version: 4, type: 'detection', incomplete: true }).querySelector('button').disabled);
});

test('the download starts only after the server says the export is possible', async () => {
  const asked = [];
  const clicked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(url);
    return { ok: true, status: 200, text: async () => JSON.stringify({ images: 3 }) };
  };
  window.HTMLAnchorElement.prototype.click = function click() { clicked.push(this.getAttribute('href')); };
  try {
    const row = exportRow(5, { version: 2, type: 'detection' });
    document.body.appendChild(row);
    row.querySelector('[data-format="coco"]').click();
    await new Promise((resolve) => setTimeout(resolve, 20));
  } finally {
    globalThis.fetch = real;
  }
  assert.deepEqual(asked, ['/api/projects/5/datasets/2/export/check?format=coco']);
  assert.deepEqual(clicked, ['/api/projects/5/datasets/2/export?format=coco']);
});

test('a refusal is shown as a message and nothing is downloaded', async () => {
  const clicked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: false, status: 400, text: async () => JSON.stringify({ detail: 'x', code: 'export_regression', params: {} }) });
  window.HTMLAnchorElement.prototype.click = function click() { clicked.push(this.getAttribute('href')); };
  try {
    const row = exportRow(5, { version: 2, type: 'detection' });
    document.body.appendChild(row);
    row.querySelector('[data-format="yolo"]').click();
    await new Promise((resolve) => setTimeout(resolve, 20));
  } finally {
    globalThis.fetch = real;
  }
  assert.deepEqual(clicked, []);
  assert.match(document.getElementById('toasts').textContent, /No regression export\./);
});
