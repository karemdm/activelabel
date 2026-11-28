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
const {
  classColour, containedRect, shapesOf, anchorOf, legendOf, chipText, annotationLayer, legendList, placeLayer,
  mountAnnotations, showAnnotations, rememberShow,
} = await import('../../../src/activelabel/web/static/js/cv/viewer_labels.js');

setMessages({
  'lightbox.annots': 'Annotations', 'lightbox.annot_show': 'Show annotations', 'lightbox.annot_none': 'No annotations on this image.',
  'lightbox.annot_error': 'Could not load.', 'lightbox.annot_pre': '{n} pre-annotated', 'lightbox.annot_pre_key': 'Dashed means pre-annotation.',
}, 'en');

const CLASSES = ['cat', 'dog'];

test('#339: a class has the colour of its place in the project list; an unknown class is neutral; the palette wraps', () => {
  assert.equal(classColour(CLASSES, 'cat'), 'var(--cls-1)');
  assert.equal(classColour(CLASSES, 'dog'), 'var(--cls-2)');
  assert.equal(classColour(CLASSES, 'bird'), 'var(--cls-none)');
  const many = Array.from({ length: 12 }, (_, i) => `c${i}`);
  assert.equal(classColour(many, 'c10'), 'var(--cls-1)');
  assert.equal(classColour(null, 'x'), 'var(--cls-none)');
});

test('#339: the image is drawn inside its box like object-fit: contain, and the layer covers just that', () => {
  assert.deepEqual(containedRect(400, 400, 200, 100), { x: 0, y: 100, w: 400, h: 200 });   // wide image: bars above and below
  assert.deepEqual(containedRect(400, 200, 100, 200), { x: 150, y: 0, w: 100, h: 200 });   // tall image: bars at the sides
  assert.deepEqual(containedRect(0, 0, 10, 10), { x: 0, y: 0, w: 0, h: 0 });               // not laid out yet
  assert.deepEqual(containedRect(300, 150, 0, 0), { x: 0, y: 0, w: 300, h: 150 });         // size unknown: the whole box
});

test('#339: coordinates are relative 0..1 and become percent of the image, clamped to it', () => {
  const [box] = shapesOf([{ label: 'cat', bbox: [0.1, 0.2, 0.5, 0.9] }], CLASSES);
  assert.equal(box.kind, 'box');
  assert.deepEqual(box.bbox, { x: 10, y: 20, w: 50, h: 80 });   // 0.9 high from 0.2 fits: 80 left, not 90
  const [out] = shapesOf([{ label: 'cat', bbox: [0.8, 0.5, 0.5, 0.2] }], CLASSES);
  assert.deepEqual(out.bbox, { x: 80, y: 50, w: 20, h: 20 });   // cut at the right edge
  const [poly] = shapesOf([{ label: 'dog', polygons: [[[0, 0], [1.5, 0], [0.5, 0.25]]] }], CLASSES);
  assert.equal(poly.kind, 'polygon');
  assert.deepEqual(poly.rings, [[[0, 0], [100, 0], [50, 25]]]);
  assert.equal(poly.colour, 'var(--cls-2)');
});

test('#339: keypoints keep hidden and visible points and drop the ones not labeled', () => {
  const [kp] = shapesOf([{ label: 'person', keypoints: [[0.1, 0.1, 2], [0.2, 0.3, 1], [0.9, 0.9, 0]] }], ['person']);
  assert.equal(kp.kind, 'keypoints');
  assert.deepEqual(kp.points, [{ x: 10, y: 10, v: 2 }, { x: 20, y: 30, v: 1 }]);
  assert.equal(kp.bbox, null);
  assert.deepEqual(anchorOf(kp), { x: 10, y: 10 });
});

test('#339: a classification label has no geometry, a regression value is not drawn, a score marks a pre-annotation', () => {
  const shapes = shapesOf([{ label: 'cat' }, { label: 'height', value: 1.8 }, { label: 'dog', bbox: [0, 0, 0.1, 0.1], score: 0.874, model: 'm1' }], CLASSES);
  assert.deepEqual(shapes.map((s) => [s.kind, s.pre]), [['class', false], ['box', true]]);
  assert.equal(anchorOf(shapes[0]), null);
  assert.equal(chipText(shapes[0]), 'cat');
  assert.equal(chipText(shapes[1]), 'dog 87%');
  assert.deepEqual(shapesOf(null, CLASSES), []);
});

test('#339: the legend counts per class, most frequent first, and counts the pre-annotations', () => {
  const shapes = shapesOf([
    { label: 'dog', bbox: [0, 0, 0.1, 0.1] }, { label: 'cat', bbox: [0.2, 0, 0.1, 0.1] }, { label: 'cat', bbox: [0.4, 0, 0.1, 0.1], score: 0.5 }], CLASSES);
  assert.deepEqual(legendOf(shapes).map((e) => [e.name, e.count, e.pre]), [['cat', 2, 1], ['dog', 1, 0]]);
  const list = legendList(shapes);
  assert.match(list.textContent, /cat\s*× 2\s*1 pre-annotated/);
  assert.equal(list.querySelectorAll('li').length, 2);
});

test('#339: the layer draws boxes, polygons and keypoints, dashed when pre-annotated, with named chips', () => {
  const layer = annotationLayer(shapesOf([
    { label: 'cat', bbox: [0.1, 0.1, 0.3, 0.3] },
    { label: 'dog', polygons: [[[0.5, 0.5], [0.9, 0.5], [0.7, 0.9]]], score: 0.9 },
    { label: 'cat', keypoints: [[0.3, 0.3, 2], [0.4, 0.4, 1]] },
    { label: 'cat' }], CLASSES));
  assert.equal(layer.querySelectorAll('rect.lb-shape').length, 1);
  assert.equal(layer.querySelector('rect').getAttribute('width'), '30');
  const polygon = layer.querySelector('polygon');
  assert.equal(polygon.getAttribute('points'), '50,50 90,50 70,90');
  assert.ok(polygon.getAttribute('class').includes('is-pre'));
  assert.equal(layer.querySelectorAll('.lb-dot').length, 2);
  assert.equal(layer.querySelectorAll('.lb-dot.is-hidden').length, 1);
  assert.deepEqual([...layer.querySelectorAll('.lb-chip')].map((c) => c.textContent), ['cat', 'dog 90%', 'cat', 'cat']);
  assert.ok(layer.querySelector('.lb-corner .lb-chip.static'));   // the class of the whole image
  assert.equal(layer.getAttribute('aria-hidden'), 'true');
  assert.equal(layer.querySelector('rect').getAttribute('style'), '--c:var(--cls-1)');
});

test('#339: the layer is placed over the drawn image, not over its letterboxed box', () => {
  const photo = document.createElement('img');
  const host = document.createElement('div');
  Object.defineProperty(photo, 'clientWidth', { value: 400 });
  Object.defineProperty(photo, 'clientHeight', { value: 400 });
  placeLayer(photo, host, { w: 200, h: 100 });
  assert.deepEqual([host.style.left, host.style.top, host.style.width, host.style.height], ['0px', '100px', '400px', '200px']);
});

test('#339: the choice to show the annotations is remembered, on by default', () => {
  localStorage.clear();
  assert.equal(showAnnotations(), true);
  rememberShow(false);
  assert.equal(showAnnotations(), false);
  rememberShow(true);
  assert.equal(showAnnotations(), true);
});

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

async function mountWith(answer) {
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(url);
    const body = JSON.stringify(answer);
    return { ok: true, status: 200, text: async () => body };
  };
  const host = document.createElement('div');
  const panel = document.createElement('div');
  try { mountAnnotations(4, 'abc', host, panel); await tick(); await tick(); } finally { globalThis.fetch = real; }
  return { asked, host, panel };
}

test('#339: opening an image makes one request and fills the layer and the panel, and the toggle hides the layer', async () => {
  localStorage.clear();
  const { asked, host, panel } = await mountWith({ labels: [{ label: 'cat', bbox: [0.1, 0.1, 0.2, 0.2] }], classes: CLASSES, revision: 'r', tags: [] });
  assert.deepEqual(asked, ['/api/projects/4/images/abc/labels']);
  assert.ok(host.querySelector('rect'));
  assert.match(panel.textContent, /Annotations/);
  assert.match(panel.textContent, /cat\s*× 1/);
  const button = panel.querySelector('[data-action="toggle-annotations"]');
  assert.equal(button.getAttribute('aria-pressed'), 'true');
  button.click();
  assert.equal(button.getAttribute('aria-pressed'), 'false');
  assert.ok(host.classList.contains('is-off'));
  assert.equal(showAnnotations(), false);
  button.click();
  assert.equal(host.classList.contains('is-off'), false);
  assert.equal(panel.querySelector('.lb-pre-key'), null);   // no pre-annotation: no key
});

test('#339: pre-annotations get a key in the panel; an image without labels says so; regression only shows nothing', async () => {
  const pre = await mountWith({ labels: [{ label: 'cat', bbox: [0, 0, 0.5, 0.5], score: 0.7, model: 'm1' }], classes: CLASSES });
  assert.match(pre.panel.querySelector('.lb-pre-key').textContent, /pre-annotation/);
  const none = await mountWith({ labels: [], classes: CLASSES });
  assert.match(none.panel.textContent, /No annotations on this image/);
  assert.equal(none.host.childNodes.length, 0);
  const regression = await mountWith({ labels: [{ label: 'height', value: 1.8 }], classes: ['height'] });
  assert.equal(regression.panel.textContent, '');
});

test('#339: a failed request is said in the panel, a cancelled one is silent', async () => {
  const real = globalThis.fetch;
  const host = document.createElement('div');
  const panel = document.createElement('div');
  globalThis.fetch = async () => ({ ok: false, status: 500, text: async () => '{}' });
  try { mountAnnotations(4, 'abc', host, panel); await tick(); await tick(); } finally { globalThis.fetch = real; }
  assert.match(panel.textContent, /Could not load/);
  const quiet = document.createElement('div');
  globalThis.fetch = async (url, init) => { init.signal.throwIfAborted(); return new Promise((_, reject) => init.signal.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })))); };
  try { mountAnnotations(4, 'abc', host, quiet).abort(); await tick(); await tick(); } finally { globalThis.fetch = real; }
  assert.equal(quiet.textContent, '');
});
