import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body><div id="toasts"></div></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const proto = window.HTMLDialogElement.prototype;
if (!proto.showModal) proto.showModal = function showModal() { this.setAttribute('open', ''); };
if (!proto.close) proto.close = function close(value) { this.returnValue = value || ''; this.removeAttribute('open'); this.dispatchEvent(new window.Event('close')); };
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { GROUP_COLOURS, NO_GROUP, clusterStatusText, groupColour, mapSvg, noiseCount, regroupDialog, similarityBlock, statusText } = await import('../../../src/activelabel/web/static/js/cv/similarity.js');

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
  'common.cancel': 'Cancel',
  'sim.no_group': 'No group',
  'sim.regroup': 'Regroup',
  'sim.regroup.title': 'Regroup the map',
  'sim.regroup.auto': 'Automatic',
  'sim.regroup.auto.help': 'the map chooses',
  'sim.regroup.kmeans': 'Number of groups',
  'sim.regroup.kmeans.help': 'you say how many',
  'sim.regroup.hdbscan': 'Groups of any size',
  'sim.regroup.hdbscan.help': 'some images may stay without a group',
  'sim.regroup.groups': 'Groups',
  'sim.regroup.min_size': 'Smallest group',
  'sim.regroup.noise_hint': 'Images outside every group are shown as no group.',
  'sim.regroup.groups_invalid': 'Groups: whole number from 2 to {max}',
  'sim.regroup.min_size_invalid': 'Smallest group: whole number, at least {min}',
  'sim.regroup.start': 'Start regrouping',
  'sim.regroup.started': 'Regrouping queued',
  'sim.regroup.cancel': 'Cancel regrouping',
  'sim.regroup.wait': 'Wait for the map to finish',
  'sim.regroup.done.kmeans': '{groups} groups asked for',
  'sim.regroup.done.hdbscan': 'Free size, smallest {min_size}, {noise} without group',
  'sim.cluster.queued': 'Regrouping queued',
  'sim.cluster.running': 'Regrouping',
  'sim.cluster.failed': 'Not regrouped: {error}',
  'sim.cluster.cancelled': 'Regrouping cancelled',
  'error.clustering_running': 'The map is already being regrouped.',
  'error.cluster_invalid': 'Invalid grouping: {reason}',
  'error.similarity_running': 'The similarity map is already being computed.',
}, 'en');

const READY = { packages: ['torch'], missing: [], install_needed: false, devices: ['auto', 'cpu', 'cuda:0'], images: 9, min_images: 5 };

/** Runs fn with the API answering `body` (the requirements answer `reqs`); returns the [method, url, body] asked. */
async function withApi(body, fn, reqs = READY, cluster = {}, refuse = null) {
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url, opts) => {
    asked.push([(opts && opts.method) || 'GET', url, opts && opts.body ? JSON.parse(opts.body) : undefined]);
    if (refuse && refuse.url === url && opts && opts.method === 'POST') return { ok: false, status: refuse.status, text: async () => JSON.stringify(refuse.body) };
    const answer = url.endsWith('/requirements') ? reqs : url.endsWith('/clustering') ? cluster : body;
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
    ['GET', '/api/projects/3/curation/similarity/clustering'], ['GET', '/api/projects/3/curation/similarity'],
    ['GET', '/api/projects/3/curation/similarity/requirements']]);
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

const DONE = {
  status: 'done', method: 'tsne',
  groups: [{ group: 0, size: 2, typical: 'a' }],
  points: { ids: ['a', 'b', 'c'], xy: [[0, 0], [0.1, 0.1], [1, 1]], group: [0, 0, -1] },
};
const button = (root, label) => [...root.querySelectorAll('button')].find((b) => b.textContent === label);

test('#667: the images without a group have a neutral colour, a row of their own and are counted', async () => {
  assert.equal(groupColour(NO_GROUP), 'var(--group-none)');
  assert.equal(noiseCount(DONE.points), 1);
  assert.equal(noiseCount(null), 0);
  assert.match(mapSvg(DONE.points), /--group-none/);
  const seen = [];
  const block = similarityBlock({ projectId: 3, onMap: (m) => seen.push(m) });
  await withApi(DONE, () => block.load());
  assert.deepEqual(seen.at(-1), { ready: true, noise: 1 });
  const row = block.node.querySelector('.sim-nogroup');
  assert.match(row.textContent, /No group.*1 image/);
  assert.match(row.querySelector('.sim-swatch').getAttribute('style'), /--group-none/);
});

test('#667: the regroup button appears with a map, is off while the map computes, and Cancel replaces it while regrouping', async () => {
  const none = similarityBlock({ projectId: 3 });
  await withApi({}, () => none.load());
  assert.equal(button(none.node, 'Regroup'), undefined);  // no map: nothing to regroup

  const block = similarityBlock({ projectId: 3 });
  await withApi(DONE, () => block.load());
  assert.equal(button(block.node, 'Regroup').disabled, false);

  const busy = similarityBlock({ projectId: 3 });
  await withApi({ ...DONE, status: 'running' }, () => busy.load());
  assert.equal(button(busy.node, 'Regroup').disabled, true);
  assert.match(busy.node.textContent, /Wait for the map to finish/);

  const regrouping = similarityBlock({ projectId: 3 });
  const asked = await withApi(DONE, async () => {
    await regrouping.load();
    assert.equal(button(regrouping.node, 'Regroup'), undefined);
    assert.match(regrouping.node.textContent, /Regrouping/);
    button(regrouping.node, 'Cancel regrouping').click();
    await tick(); await tick();
  }, READY, { status: 'running', method: 'kmeans', groups: 4 });
  assert.ok(asked.some((a) => a[0] === 'POST' && a[1] === '/api/projects/3/curation/similarity/clustering/cancel'));
  assert.equal(clusterStatusText({ status: 'queued' }), 'Regrouping queued');
  assert.equal(clusterStatusText({ status: 'failed', error: 'interrupted' }), 'Not regrouped: closed');
  assert.equal(clusterStatusText({ status: 'cancelled' }), 'Regrouping cancelled');
  assert.equal(clusterStatusText({ status: 'done' }), '');
});

/** Opens the dialog of the block, runs `fn(dialog)` and returns the [method, url, body] asked. */
async function inDialog(fn, { state = DONE, cluster = {}, refuse = null } = {}) {
  const block = similarityBlock({ projectId: 3 });
  document.querySelectorAll('dialog').forEach((d) => d.remove());
  document.getElementById('toasts').replaceChildren();
  return withApi(state, async () => {
    await block.load();
    button(block.node, 'Regroup').click();
    const dialog = document.querySelector('dialog');
    await fn(dialog);
    await tick(); await tick();
  }, READY, cluster, refuse);
}

const choose = (dialog, method) => {
  const radio = dialog.querySelector(`input[name="rg-method"][value="${method}"]`);
  radio.checked = true;
  radio.dispatchEvent(new window.Event('change'));
};
const submit = (dialog) => dialog.querySelector('form').dispatchEvent(new window.Event('submit', { cancelable: true }));

test('#667: the dialog offers the three ways and sends the chosen one', async () => {
  const asked = await inDialog(async (dialog) => {
    assert.deepEqual([...dialog.querySelectorAll('input[type=radio]')].map((r) => r.value), ['auto', 'kmeans', 'hdbscan']);
    assert.equal(dialog.querySelector('input[value=auto]').checked, true);
    assert.equal(dialog.querySelector('#rg-groups'), null);   // the number is asked only for its method
    choose(dialog, 'kmeans');
    dialog.querySelector('#rg-groups').value = '3';
    submit(dialog);
  });
  assert.deepEqual(asked.at(-1), ['POST', '/api/projects/3/curation/similarity/clustering', { method: 'kmeans', groups: 3, min_size: 5 }]);

  const hdb = await inDialog(async (dialog) => {
    choose(dialog, 'hdbscan');
    assert.match(dialog.textContent, /Images outside every group/);
    dialog.querySelector('#rg-size').value = '8';
    submit(dialog);
  });
  assert.deepEqual(hdb.at(-1)[2], { method: 'hdbscan', groups: 0, min_size: 8 });

  const auto = await inDialog(async (dialog) => submit(dialog));
  assert.deepEqual(auto.at(-1)[2], { method: 'auto', groups: 0, min_size: 5 });
});

test('#667: the dialog refuses a number out of range before asking the server', async () => {
  const asked = await inDialog(async (dialog) => {
    choose(dialog, 'kmeans');
    for (const bad of ['1', '2,5', 'abc', '', '51']) {
      dialog.querySelector('#rg-groups').value = bad;
      submit(dialog);
      assert.equal(dialog.hasAttribute('open'), true, bad);
      assert.equal(dialog.querySelector('#rg-error').hidden, false, bad);
      assert.equal(dialog.querySelector('#rg-groups').getAttribute('aria-invalid'), 'true');
    }
    assert.match(dialog.querySelector('#rg-error').textContent, /from 2 to 3/);   // never more groups than the map has images
    choose(dialog, 'hdbscan');
    assert.equal(dialog.querySelector('#rg-error').hidden, true);
    dialog.querySelector('#rg-size').value = '1';
    submit(dialog);
    assert.match(dialog.querySelector('#rg-error').textContent, /at least 2/);
    dialog.close('cancel');
  });
  assert.equal(asked.filter((a) => a[0] === 'POST').length, 0);
});

test('#667: Cancel in the dialog sends nothing', async () => {
  const asked = await inDialog(async (dialog) => { button(dialog, 'Cancel').click(); });
  assert.equal(asked.filter((a) => a[0] === 'POST').length, 0);
  assert.equal(document.querySelector('dialog'), null);
});

test('#667: a refused regrouping shows the server error in plain words', async () => {
  for (const [status, code, text] of [
    [409, 'clustering_running', /already being regrouped/],
    [409, 'similarity_running', /already being computed/],
    [400, 'cluster_invalid', /Invalid grouping: groups out of range/],
  ]) {
    const refuse = { url: '/api/projects/3/curation/similarity/clustering', status, body: { code, params: { reason: 'groups out of range' } } };
    let seen = null;
    const asked = await inDialog(async (dialog) => {
      submit(dialog);
      await tick(); await tick();
      seen = document.getElementById('toasts').textContent;
    }, { refuse });
    assert.ok(asked.some((a) => a[0] === 'POST'));
    assert.match(seen, text);
  }
});
