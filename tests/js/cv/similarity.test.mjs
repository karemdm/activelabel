import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const proto = window.HTMLDialogElement.prototype;
if (!proto.showModal) proto.showModal = function showModal() { this.setAttribute('open', ''); };
if (!proto.close) proto.close = function close(value) { this.returnValue = value || ''; this.removeAttribute('open'); this.dispatchEvent(new window.Event('close')); };
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { GROUP_COLOURS, groupColour, mapSvg, similarityBlock, statusText } = await import('../../../src/activelabel/web/static/js/cv/similarity.js');

setMessages({
  'sim.title': 'Similarity map',
  'sim.compute': 'Compute map',
  'sim.recompute': 'Compute again',
  'sim.cancel': 'Cancel',
  'sim.open_fo': 'Open in FiftyOne',
  'sim.queued': 'Queued',
  'sim.embedding': 'Reading {done} of {total}',
  'sim.phase.environment': 'Preparing',
  'sim.phase.map': 'Building the map',
  'sim.failed': 'Failed: {error}',
  'sim.interrupted': 'closed',
  'sim.cancelled': 'Cancelled',
  'sim.summary': '{images} images in {groups} groups ({method})',
  'sim.method.tsne': 't-SNE',
  'sim.outdated': 'Outdated',
  'sim.group': 'Group {n}',
  'sim.images_n_one': '{n} image',
  'sim.images_n_other': '{n} images',
  'sim.map_label': 'Map of {n} images',
  'sim.too_few': 'Needs {min} images, has {n}',
  'sim.install_title': 'Install PyTorch for the map?',
  'sim.install_confirm': 'Install and compute',
  'tr.packages_note': 'This computer still needs: {list}.',
  'tr.device_label': 'Device',
}, 'en');

const READY = { packages: ['torch'], missing: [], install_needed: false, devices: ['auto', 'cpu', 'cuda:0'], images: 9, min_images: 5 };

/** Runs fn with the API answering `body` (the requirements answer `reqs`); returns the [method, url, body] asked. */
async function withApi(body, fn, reqs = READY) {
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url, opts) => {
    asked.push([(opts && opts.method) || 'GET', url, opts && opts.body ? JSON.parse(opts.body) : undefined]);
    const answer = url.endsWith('/requirements') ? reqs : body;
    return { ok: true, status: 200, text: async () => JSON.stringify(answer) };
  };
  try { await fn(); } finally { globalThis.fetch = real; }
  return asked;
}

const tick = () => new Promise((r) => setTimeout(r, 0));

test('the status line follows the phase of the computation', () => {
  assert.equal(statusText({ status: 'queued' }), 'Queued');
  assert.equal(statusText({ status: 'running', phase: 'embedding', done: 2, total: 9 }), 'Reading 2 of 9');
  assert.equal(statusText({ status: 'running', phase: 'map' }), 'Building the map');
  assert.equal(statusText({ status: 'running' }), 'Preparing');
  assert.equal(statusText({ status: 'failed', error: 'interrupted' }), 'Failed: closed');
  assert.equal(statusText({ status: 'cancelled' }), 'Cancelled');
  assert.equal(statusText({ status: 'done' }), '');
});

test('the map draws one dot per image, coloured by group, y upwards', () => {
  const svg = mapSvg({ xy: [[0, 0], [1, 1]], group: [0, 9] });
  const doc = new JSDOM(svg).window.document;
  const dots = [...doc.querySelectorAll('circle')];
  assert.equal(dots.length, 2);
  assert.equal(dots[0].getAttribute('cy'), '392.0');  // y = 0 is at the bottom
  assert.match(dots[1].getAttribute('style'), /--group-2/);  // groups beyond the palette reuse the colours
  assert.equal(groupColour(GROUP_COLOURS), 'var(--group-1)');
  assert.match(svg, /aria-label="Map of 2 images"/);
});

test('a finished map shows its summary, groups and actions', async () => {
  const state = {
    status: 'done', method: 'tsne', outdated: true,
    groups: [{ group: 0, size: 2, typical: 'a' }, { group: 1, size: 1, typical: 'c' }],
    points: { ids: ['a', 'b', 'c'], xy: [[0, 0], [0.1, 0.1], [1, 1]], group: [0, 0, 1] },
  };
  const block = similarityBlock({ projectId: 3, ctx: { openWhenDone: new Map() } });
  const asked = await withApi(state, () => block.load());
  assert.deepEqual(asked.map((a) => a.slice(0, 2)), [
    ['GET', '/api/projects/3/curation/similarity'], ['GET', '/api/projects/3/curation/similarity/requirements']]);
  const text = block.node.textContent;
  assert.match(text, /3 images in 2 groups \(t-SNE\)/);
  assert.match(text, /Outdated/);
  assert.match(text, /Group 1.*2 images/);
  assert.match(text, /Compute again/);
  assert.match(text, /Open in FiftyOne/);
  assert.equal(block.node.querySelectorAll('.sim-group img')[1].getAttribute('src'), '/api/projects/3/images/c/thumb');
});

test('without a map the block offers to compute it, and starting posts the request', async () => {
  const block = similarityBlock({ projectId: 3 });
  assert.match(block.node.textContent, /Compute map/);
  assert.doesNotMatch(block.node.textContent, /Open in FiftyOne/);
  const button = [...block.node.querySelectorAll('button')].find((b) => b.textContent === 'Compute map');
  const asked = await withApi({ status: 'queued' }, async () => { button.click(); await tick(); await tick(); });
  assert.deepEqual(asked[0].slice(0, 2), ['GET', '/api/projects/3/curation/similarity/requirements']);
  assert.deepEqual(asked[1], ['POST', '/api/projects/3/curation/similarity', { device: 'auto' }]);
  assert.match(block.node.textContent, /Queued/);
  assert.match(block.node.textContent, /Cancel/);
});

test('#664: the chosen device is sent, and too few images keep the button off', async () => {
  const block = similarityBlock({ projectId: 3 });
  await withApi({}, () => block.load());
  const select = block.node.querySelector('#sim-device');
  assert.deepEqual([...select.options].map((o) => o.value), ['auto', 'cpu', 'cuda:0']);
  select.value = 'cuda:0';
  select.dispatchEvent(new window.Event('change'));
  const button = [...block.node.querySelectorAll('button')].find((b) => b.textContent === 'Compute map');
  const asked = await withApi({ status: 'queued' }, async () => { button.click(); await tick(); await tick(); });
  assert.deepEqual(asked.at(-1), ['POST', '/api/projects/3/curation/similarity', { device: 'cuda:0' }]);

  const few = similarityBlock({ projectId: 4 });
  await withApi({}, () => few.load(), { ...READY, images: 2 });
  const off = [...few.node.querySelectorAll('button')].find((b) => b.textContent === 'Compute map');
  assert.equal(off.disabled, true);
  assert.match(few.node.textContent, /Needs 5 images, has 2/);
});

test('#664: the first map asks before installing PyTorch, and nothing starts when declined', async () => {
  const need = { ...READY, missing: ['torch', 'torchvision'], install_needed: true };
  for (const answer of ['ok', 'cancel']) {
    const block = similarityBlock({ projectId: 3 });
    const button = [...block.node.querySelectorAll('button')].find((b) => b.textContent === 'Compute map');
    const asked = await withApi({ status: 'queued' }, async () => {
      button.click();
      await tick(); await tick();
      const dialog = document.querySelector('dialog');
      assert.match(dialog.textContent, /This computer still needs: torch, torchvision\./);
      assert.match(dialog.textContent, /Install and compute/);
      dialog.close(answer);
      await tick(); await tick();
    }, need);
    const posted = asked.filter((a) => a[0] === 'POST');
    assert.equal(posted.length, answer === 'ok' ? 1 : 0);
  }
});
