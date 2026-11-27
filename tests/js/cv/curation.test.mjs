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
const { curationPanel } = await import('../../../src/activelabel/web/static/js/cv/curation.js');

setMessages({ 'step.curation': 'Curation', 'cur.not_analyzed': 'The images have not been analyzed yet.' }, 'en');

test('the Curation step loads the analysis, updates the step bar and draws the not-analyzed notice', async () => {
  let stepper = 0;
  const ctx = { project: { id: 6, counts: { active: 2 } }, jobs: [], panels: {}, watchers: [] };
  const page = {
    ctx, projectId: 6, renderStepper: () => { stepper += 1; }, refreshProject: async () => true,
    jobStarted: async () => {}, kindActive: () => false,
  };
  const panel = curationPanel(page);
  assert.equal(panel.node.id, 'panel-curation');
  document.body.appendChild(panel.node);
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(url.split('?')[0]);
    const body = url.includes('/images') ? { total: 0, items: [] } : { totals: { measured: 0 }, issue_ids: {}, thresholds: {} };
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try {
    await panel.mount();
    await new Promise((resolve) => setTimeout(resolve, 20));
  } finally {
    globalThis.fetch = real;
  }
  assert.equal(asked[0], '/api/projects/6/curation');
  assert.ok(asked.includes('/api/projects/6/images'));
  assert.equal(stepper, 1);
  assert.equal(ctx.project.counts.analyzed, 0);
  assert.match(panel.node.textContent, /not been analyzed/);
  panel.node.remove();
});

const BODIES = {
  '/api/projects/6/curation': { totals: { measured: 0 }, issue_ids: {}, thresholds: {} },
  '/api/projects/6/curation/uniqueness': { status: 'done', scores: { a: 0.2, b: 0.9 }, missing: 0, outdated: false },
  '/api/projects/6/curation/similarity': { status: 'done', method: 'tsne', groups: [], points: { ids: ['a', 'b', 'c'], xy: [[0, 0], [0, 1], [1, 1]], group: [0, -1, -1] } },
  '/api/projects/6/curation/similarity/clustering': { status: 'done', method: 'hdbscan', min_size: 5, noise: 2 },
  '/api/projects/6/curation/similarity/requirements': { packages: [], missing: [], install_needed: false, devices: ['auto'], images: 9, min_images: 5 },
};

test('#667: the step sorts and filters the grid by uniqueness and by the images without a group', async () => {
  const ctx = { project: { id: 6, counts: { active: 3 } }, jobs: [], panels: {}, watchers: [] };
  const panel = curationPanel({ ctx, projectId: 6, renderStepper: () => {}, refreshProject: async () => true, jobStarted: async () => {}, kindActive: () => false });
  document.body.appendChild(panel.node);
  const gridAsks = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const [path, query] = url.split('?');
    if (path.endsWith('/images')) gridAsks.push(Object.fromEntries(new URLSearchParams(query)));
    const body = path.endsWith('/images') ? { total: 0, items: [] } : BODIES[path] || {};
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  const wait = () => new Promise((resolve) => setTimeout(resolve, 20));
  try {
    await panel.mount();
    await wait();
    const last = () => gridAsks.at(-1);
    assert.equal(last().sort, undefined);
    assert.equal(last().group, undefined);
    assert.ok(panel.node.querySelector('.uniq'));
    assert.equal(panel.node.querySelector('.uniq-controls').hidden, false);

    const select = panel.node.querySelector('#uniq-sort');
    select.value = 'uniqueness_desc';
    select.dispatchEvent(new window.Event('change'));
    await wait();
    assert.equal(last().sort, 'uniqueness_desc');

    const below = panel.node.querySelector('#uniq-below');
    below.value = '0.4';
    panel.node.querySelector('.uniq-below').dispatchEvent(new window.Event('submit', { cancelable: true }));
    await wait();
    assert.equal(last().unique_below, '0.4');
    assert.equal(last().sort, 'uniqueness_desc');

    const chip = panel.node.querySelector('[data-act="no-group"]');
    assert.ok(chip, 'two images are in no group: the filter is offered');
    assert.equal(chip.querySelector('.count').textContent, '2');
    chip.click();
    await wait();
    assert.equal(last().group, '-1');
    assert.equal(panel.node.querySelector('[data-act="no-group"]').getAttribute('aria-pressed'), 'true');
    panel.node.querySelector('[data-act="no-group"]').click();
    await wait();
    assert.equal(last().group, undefined);
  } finally {
    globalThis.fetch = real;
    panel.node.remove();
  }
});
