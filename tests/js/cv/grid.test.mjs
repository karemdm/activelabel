import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
for (const [name, value] of Object.entries({
  window, document: window.document, Node: window.Node, localStorage: window.localStorage,
})) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const proto = window.HTMLDialogElement.prototype;
if (!proto.showModal) proto.showModal = function showModal() { this.setAttribute('open', ''); };
if (!proto.close) proto.close = function close() { this.removeAttribute('open'); };
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { imageGrid, openLightbox } = await import('../../../src/activelabel/web/static/js/cv/grid.js');

setMessages({ 'grid.range': '{from}–{to} of {total}', 'grid.empty': 'No images', 'tag.dark': 'Dark', 'lightbox.prev': 'Previous', 'lightbox.next': 'Next',
  'uniq.value': 'Uniqueness {value}', 'lightbox.uniqueness': 'Uniqueness', 'lightbox.uniqueness_help': '0 is almost the same as another image.' }, 'en');

const images = (n) => Array.from({ length: n }, (_, i) => ({ id: `i${i}`, name: `img${i}.png`, issues: i === 0 ? ['dark'] : [] }));

/** Calls fn with the API answering image pages of `total` images; returns the URLs asked. */
async function withImages(total, fn) {
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(url);
    const q = new URL(url, 'http://localhost').searchParams;
    const offset = Number(q.get('offset'));
    const items = images(total).slice(offset, offset + Number(q.get('limit')));
    return { ok: true, status: 200, text: async () => JSON.stringify({ total, items }) };
  };
  try {
    await fn();
  } finally {
    globalThis.fetch = real;
  }
  return asked;
}

test('the grid asks for one page with the filter and draws a card per image', async () => {
  let grid;
  const asked = await withImages(3, async () => {
    grid = imageGrid({ projectId: 7, query: () => ({ status: 'active', issue: 'dark', task: 2 }) });
    await grid.reload();
  });
  const url = new URL(asked[0], 'http://localhost');
  assert.equal(url.pathname, '/api/projects/7/images');
  assert.deepEqual(Object.fromEntries(url.searchParams), { status: 'active', offset: '0', limit: '60', issue: 'dark', task: '2' });
  assert.equal(grid.node.querySelectorAll('.thumb').length, 3);
  assert.equal(grid.node.querySelector('.grid-info').textContent, '1–3 of 3');
  assert.equal(grid.node.querySelector('.thumb .tag-dark').textContent, 'Dark');
  assert.equal(grid.node.querySelector('.pager-pages'), null);  // one page: no page buttons
});

test('a selectable grid keeps the selected ids, and allIds walks every page', async () => {
  let grid;
  let seen = null;
  await withImages(250, async () => {
    grid = imageGrid({ projectId: 1, query: () => ({}), selectable: true, onSelect: (s) => { seen = [...s]; } });
    await grid.reload();
    const box = grid.node.querySelector('.thumb input[type="checkbox"]');
    box.checked = true;
    box.dispatchEvent(new window.Event('change'));
    assert.deepEqual(seen, ['i0']);
    assert.ok(grid.node.querySelector('.thumb.is-selected'));
    assert.ok(grid.node.querySelector('.pager-pages'));
    assert.equal((await grid.allIds()).length, 250);
  });
  grid.setSelected(['i1']);
  assert.deepEqual(seen, ['i1']);
});

test('the lightbox shows one image and moves with its buttons', () => {
  openLightbox(3, images(2), 0);
  const box = document.querySelector('dialog.lightbox');
  assert.ok(box.hasAttribute('open'));
  assert.equal(box.querySelector('h2').textContent, 'img0.png');
  assert.equal(box.querySelector('img').getAttribute('src'), '/api/projects/3/images/i0/file');
  assert.equal(box.querySelector('[data-nav="prev"]').disabled, true);
  box.querySelector('[data-nav="next"]').click();
  assert.equal(box.querySelector('h2').textContent, 'img1.png');
  assert.equal(box.querySelector('[data-nav="next"]').disabled, true);
  box.close();
});

test('#667: the sort, the uniqueness filter and the group go to the server, in every page and in allIds', async () => {
  let grid;
  const asked = await withImages(3, async () => {
    grid = imageGrid({ projectId: 7, query: () => ({ status: 'active', sort: 'uniqueness_asc', uniqueBelow: 0.3, group: -1 }) });
    await grid.reload();
    await grid.allIds();
  });
  for (const url of asked) {
    const q = new URL(url, 'http://localhost').searchParams;
    assert.equal(q.get('sort'), 'uniqueness_asc');
    assert.equal(q.get('unique_below'), '0.3');
    assert.equal(q.get('group'), '-1');
  }
  const none = await withImages(1, async () => { await imageGrid({ projectId: 7, query: () => ({ uniqueBelow: 0, group: 0 }) }).reload(); });
  const q = new URL(none[0], 'http://localhost').searchParams;
  assert.equal(q.get('unique_below'), '0');   // zero is a value, not "no filter"
  assert.equal(q.get('group'), '0');
  assert.equal(q.has('sort'), false);
});

test('#667: the value shows on the card only when the grid is sorted or filtered by it, and always in the lightbox', async () => {
  const scored = async (query) => {
    let grid;
    const real = globalThis.fetch;
    globalThis.fetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify({ total: 1, items: [{ id: 'a', name: 'a.png', issues: [], uniqueness: 0.5 }] }) });
    try { grid = imageGrid({ projectId: 1, query }); await grid.reload(); } finally { globalThis.fetch = real; }
    return grid.node;
  };
  assert.match((await scored(() => ({ sort: 'uniqueness_desc' }))).querySelector('.thumb-uniq').textContent, /Uniqueness 0\.50/);
  assert.ok((await scored(() => ({ uniqueBelow: 0.9 }))).querySelector('.thumb-uniq'));
  assert.equal((await scored(() => ({}))).querySelector('.thumb-uniq'), null);

  openLightbox(1, [{ id: 'a', name: 'a.png', issues: [], uniqueness: 0.5 }, { id: 'b', name: 'b.png', issues: [] }], 0);
  const box = document.querySelector('dialog.lightbox');
  assert.match(box.textContent, /Uniqueness0\.50.*almost the same/);
  box.querySelector('[data-nav="next"]').click();
  assert.doesNotMatch(box.textContent, /Uniqueness/);   // an image without a score shows no value
  box.close();
});
