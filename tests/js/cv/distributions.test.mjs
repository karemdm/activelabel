import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { SHOWN, barList, distributionsBlock, partBalance } = await import('../../../src/activelabel/web/static/js/cv/distributions.js');

setMessages({
  'dist.title': 'Distribution', 'dist.summary': '{annotated} of {total}; {without} none', 'dist.none': 'Nothing',
  'dist.objects_per_class': 'Objects per class', 'dist.images_per_class': 'Images per class', 'dist.classes_note': 'note',
  'dist.class_title': '{name}: {objects} in {images}', 'dist.per_image': 'Per image', 'dist.per_image_note': 'n',
  'dist.box_sizes': 'Box sizes', 'dist.box_sizes_note': '{boxes} boxes', 'dist.area.tiny': 'Tiny', 'dist.class': 'Class',
  'dist.share_title': '{class} {pct}% of {part}', 'ds.show_all': 'All ({n})', 'ds.show_less': 'Less',
}, 'en');

const texts = (node, selector) => [...node.querySelectorAll(selector)].map((n) => n.textContent);

test('bars share one scale and show their values', () => {
  const list = barList([{ label: 'a', value: 10 }, { label: 'b', value: 5, text: '5 · 2' }]);
  assert.deepEqual(texts(list, '.bar-label'), ['a', 'b']);
  assert.deepEqual(texts(list, '.bar-value'), ['10', '5 · 2']);
  assert.deepEqual([...list.querySelectorAll('.bar-fill')].map((n) => n.style.width), ['100%', '50%']);
});

test('nothing is drawn without images or annotations', () => {
  assert.equal(distributionsBlock(null), null);
  assert.equal(distributionsBlock({ images: 4, annotated: 0 }), null);
});

test('a detection project shows classes, objects per image and box sizes', () => {
  const block = distributionsBlock({
    task: 'detection', images: 5, annotated: 4, without_annotations: 1, boxes: 6,
    classes: [{ name: 'car', objects: 8, images: 3 }, { name: 'bus', objects: 2, images: 1 }],
    objects_per_image: [{ key: '0', images: 1 }, { key: '1', images: 3 }],
    box_area: [{ key: 'tiny', objects: 6 }, { key: 'odd', objects: 0 }],
  });
  assert.equal(texts(block, 'figcaption').join('|'), 'Objects per class|Per image|Box sizes');
  assert.ok(block.textContent.includes('4 of 5; 1 none'));
  assert.deepEqual(texts(block.querySelectorAll('figure')[0], '.bar-value'), ['8 · 3', '2 · 1']);
  assert.deepEqual(texts(block.querySelectorAll('figure')[2], '.bar-label'), ['Tiny', 'odd']);  // a bucket with no text keeps its key
});

test('a classification project shows images per class only', () => {
  const classes = Array.from({ length: SHOWN + 2 }, (_, i) => ({ name: `c${i}`, objects: 3, images: 3 }));
  const block = distributionsBlock({ task: 'classification', images: 30, annotated: 30, without_annotations: 0, classes, objects_per_image: null, box_area: null });
  assert.equal(block.querySelectorAll('figure').length, 1);
  assert.equal(block.querySelector('figcaption').textContent, 'Images per class');
  assert.equal(block.querySelectorAll('li').length, SHOWN);
  const more = block.querySelector('button');
  assert.equal(more.hidden, false);
  more.click();
  assert.equal(block.querySelectorAll('li').length, SHOWN + 2);
  assert.equal(more.textContent, 'Less');
});

test('the class balance of the parts is the share of each class in each part', () => {
  const items = [{ label: 'Train', n: 8, cls: 'train' }, { label: 'Val', n: 2, cls: 'val' }];
  const per = [{ objects: { a: 6, b: 2 } }, { objects: { a: 1, b: 1 } }];
  const chart = partBalance(items, per, ['a', 'b']);
  assert.deepEqual(texts(chart, '.bar-value'), ['75%', '50%', '25%', '50%']);
  assert.equal(chart.style.getPropertyValue('--parts'), '2');
  assert.equal(chart.querySelectorAll('.dist-balance-cell')[1].title, 'a 50% of Val');
  assert.equal(partBalance(items, per, []), null);
  assert.deepEqual(texts(partBalance(items, [{}, {}], ['a']), '.bar-value'), ['0%', '0%']);
});
