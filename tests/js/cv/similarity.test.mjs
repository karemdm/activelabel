import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
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
}, 'en');

/** Runs fn with the API answering `body`; returns the [method, url] pairs asked. */
async function withApi(body, fn) {
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url, opts) => {
    asked.push([(opts && opts.method) || 'GET', url]);
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try { await fn(); } finally { globalThis.fetch = real; }
  return asked;
}

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
  const block = similarityBlock({ projectId: 3, ctx: { openWhenDone: new Set() } });
  const asked = await withApi(state, () => block.load());
  assert.deepEqual(asked, [['GET', '/api/projects/3/curation/similarity']]);
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
  const asked = await withApi({ status: 'queued' }, async () => { button.click(); await new Promise((r) => setTimeout(r, 0)); });
  assert.deepEqual(asked[0], ['POST', '/api/projects/3/curation/similarity']);
  assert.match(block.node.textContent, /Queued/);
  assert.match(block.node.textContent, /Cancel/);
});
