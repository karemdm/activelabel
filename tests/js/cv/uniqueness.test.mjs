import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body><div id="toasts"></div></body></html>', { url: 'http://localhost/' });
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { SORTS, parseBelow, statusText, uniquenessBlock } = await import('../../../src/activelabel/web/static/js/cv/uniqueness.js');

setMessages({
  'uniq.title': 'Uniqueness',
  'uniq.compute': 'Compute uniqueness',
  'uniq.recompute': 'Compute again',
  'uniq.cancel': 'Cancel',
  'uniq.started': 'Queued',
  'uniq.queued': 'Waiting in the queue',
  'uniq.phase.scoring': 'Scoring',
  'uniq.phase.fiftyone': 'Sending to FiftyOne',
  'uniq.failed': 'Failed: {error}',
  'uniq.interrupted': 'closed',
  'uniq.cancelled': 'Cancelled',
  'uniq.summary': '{n} images scored.',
  'uniq.missing': '{n} are not on the map.',
  'uniq.outdated': 'Outdated scores',
  'uniq.fo_failed': 'FiftyOne: {error}',
  'uniq.needs_map': 'Compute the similarity map first.',
  'uniq.sort_label': 'Sort by',
  'uniq.sort.none': 'Project order',
  'uniq.sort.uniqueness_desc': 'Most unique first',
  'uniq.sort.uniqueness_asc': 'Most repeated first',
  'uniq.below_label': 'Below',
  'uniq.below_apply': 'Filter',
  'uniq.below_clear': 'Show all',
  'uniq.below_help': 'A value from 0 to 1.',
  'uniq.below_invalid': 'Type a number from 0 to 1.',
  'error.similarity_map_needed': 'Compute the similarity map first: this step uses its embeddings.',
  'error.uniqueness_running': 'The uniqueness is already being computed.',
}, 'en');

const DONE = { status: 'done', scores: { a: 0.9, b: 0.1 }, missing: 1, outdated: false };
const tick = () => new Promise((r) => setTimeout(r, 0));
const button = (root, label) => [...root.querySelectorAll('button')].find((b) => b.textContent === label);

/** Runs fn with the API answering `answer` (or the refusal `refuse` to the POSTs); returns the [method, url, body] asked. */
async function withApi(answer, fn, refuse = null) {
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url, opts) => {
    const method = (opts && opts.method) || 'GET';
    asked.push([method, url, opts && opts.body ? JSON.parse(opts.body) : undefined]);
    if (refuse && method === 'POST') return { ok: false, status: refuse.status, text: async () => JSON.stringify(refuse.body) };
    return { ok: true, status: 200, text: async () => JSON.stringify(answer) };
  };
  try { await fn(); } finally { globalThis.fetch = real; }
  return asked;
}

test('the status line follows the scoring', () => {
  assert.equal(statusText({ status: 'queued' }), 'Waiting in the queue');
  assert.equal(statusText({ status: 'running', phase: 'fiftyone' }), 'Sending to FiftyOne');
  assert.equal(statusText({ status: 'running' }), 'Scoring');
  assert.equal(statusText({ status: 'failed', error: 'interrupted' }), 'Failed: closed');
  assert.equal(statusText({ status: 'cancelled' }), 'Cancelled');
  assert.equal(statusText({ status: 'done' }), '');
});

test('the value of the "below" field is a number from 0 to 1, empty means no filter', () => {
  assert.equal(parseBelow(''), null);
  assert.equal(parseBelow('  '), null);
  assert.equal(parseBelow('0.3'), 0.3);
  assert.equal(parseBelow('0'), 0);
  assert.equal(parseBelow('1'), 1);
  for (const bad of ['1.5', '-0.1', 'abc']) assert.ok(Number.isNaN(parseBelow(bad)), bad);
});

test('without the similarity map the button is off and says why', async () => {
  const block = uniquenessBlock({ projectId: 3 });
  await withApi({}, () => block.load());
  const compute = button(block.node, 'Compute uniqueness');
  assert.equal(compute.disabled, true);
  assert.match(block.node.textContent, /Compute the similarity map first\./);
  assert.equal(compute.getAttribute('aria-describedby'), 'uniq-reason');
  assert.equal(block.controls.hidden, true);   // nothing to sort by yet

  block.setMap(true);
  assert.equal(button(block.node, 'Compute uniqueness').disabled, false);
  // one primary action per screen (#340): this optional step is a secondary button
  assert.equal(button(block.node, 'Compute uniqueness').classList.contains('primary'), false);
  assert.doesNotMatch(block.node.textContent, /Compute the similarity map first\./);
});

test('starting posts the request, shows progress and Cancel, and cancel posts too', async () => {
  const block = uniquenessBlock({ projectId: 3 });
  block.setMap(true);
  const asked = await withApi({ status: 'queued' }, async () => {
    button(block.node, 'Compute uniqueness').click();
    await tick(); await tick();
    assert.match(block.node.textContent, /Waiting in the queue/);
    assert.equal(button(block.node, 'Compute uniqueness').disabled, true);
    button(block.node, 'Cancel').click();
    await tick(); await tick();
  });
  assert.deepEqual(asked[0], ['POST', '/api/projects/3/curation/uniqueness', undefined]);
  assert.ok(asked.some((a) => a[0] === 'POST' && a[1] === '/api/projects/3/curation/uniqueness/cancel'));
});

test('a refusal of the server is said in plain words', async () => {
  for (const [code, text] of [['similarity_map_needed', /Compute the similarity map first: this step/], ['uniqueness_running', /already being computed/]]) {
    document.getElementById('toasts').replaceChildren();
    const block = uniquenessBlock({ projectId: 3 });
    block.setMap(true);
    await withApi({}, async () => {
      button(block.node, 'Compute uniqueness').click();
      await tick(); await tick();
    }, { status: 409, body: { code, params: {} } });
    assert.match(document.getElementById('toasts').textContent, text);
    assert.equal(button(block.node, 'Compute uniqueness').disabled, false);   // try again
  }
});

test('with scores: summary, sort and filter controls, and the notice when they are outdated', async () => {
  let changes = 0;
  const block = uniquenessBlock({ projectId: 3, onChange: () => { changes += 1; } });
  block.setMap(true);
  await withApi({ ...DONE, outdated: true, fiftyone_error: 'not installed' }, () => block.load());
  assert.match(block.node.textContent, /2 images scored\. 1 are not on the map\./);
  assert.match(block.node.textContent, /Outdated scores/);
  assert.match(block.node.textContent, /FiftyOne: not installed/);
  assert.ok(button(block.node, 'Compute again'));
  assert.equal(block.controls.hidden, false);
  assert.deepEqual([...block.controls.querySelectorAll('#uniq-sort option')].map((o) => o.value), SORTS);
  assert.deepEqual([...block.controls.querySelectorAll('#uniq-sort option')].map((o) => o.textContent), ['Project order', 'Most unique first', 'Most repeated first']);
  assert.deepEqual(block.query(), { sort: '', uniqueBelow: null });
  assert.equal(changes, 1);   // the scores appeared: the grid is asked again

  const select = block.controls.querySelector('#uniq-sort');
  select.value = 'uniqueness_asc';
  select.dispatchEvent(new window.Event('change'));
  assert.deepEqual(block.query(), { sort: 'uniqueness_asc', uniqueBelow: null });
  assert.equal(changes, 2);
  assert.equal(block.controls.querySelector('#uniq-sort').value, 'uniqueness_asc');   // kept after the redraw
});

test('a fresh scoring without outdated scores shows no warning', async () => {
  const block = uniquenessBlock({ projectId: 3 });
  block.setMap(true);
  await withApi(DONE, () => block.load());
  assert.doesNotMatch(block.node.textContent, /Outdated scores/);
});

test('the filter below a value is applied, validated and cleared', async () => {
  let changes = 0;
  const block = uniquenessBlock({ projectId: 3, onChange: () => { changes += 1; } });
  await withApi(DONE, () => block.load());
  changes = 0;
  const submit = () => block.controls.querySelector('form').dispatchEvent(new window.Event('submit', { cancelable: true }));

  block.controls.querySelector('#uniq-below').value = '0.25';
  submit();
  assert.deepEqual(block.query(), { sort: '', uniqueBelow: 0.25 });
  assert.equal(changes, 1);
  assert.ok(button(block.controls, 'Show all'));

  block.controls.querySelector('#uniq-below').value = '3';
  submit();
  assert.equal(block.controls.querySelector('#uniq-below').getAttribute('aria-invalid'), 'true');
  assert.match(block.controls.textContent, /Type a number from 0 to 1\./);
  assert.equal(block.query().uniqueBelow, 0.25);   // an invalid value changes nothing
  assert.equal(changes, 1);

  button(block.controls, 'Show all').click();
  assert.deepEqual(block.query(), { sort: '', uniqueBelow: null });
  assert.equal(block.controls.querySelector('#uniq-below').value, '');
  assert.equal(changes, 2);
});

test('when the scores go away the sort and the filter are dropped', async () => {
  let changes = 0;
  const block = uniquenessBlock({ projectId: 3, onChange: () => { changes += 1; } });
  await withApi(DONE, () => block.load());
  block.controls.querySelector('#uniq-below').value = '0.5';
  block.controls.querySelector('form').dispatchEvent(new window.Event('submit', { cancelable: true }));
  changes = 0;
  await withApi({}, () => block.load());
  assert.deepEqual(block.query(), { sort: '', uniqueBelow: null });
  assert.equal(block.controls.hidden, true);
  assert.equal(changes, 1);
});
