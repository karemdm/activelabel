import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { initImport } = await import('../../../src/activelabel/web/static/js/pages/import.js');

setMessages({
  'annotate.tool.cvat': 'CVAT', 'annotate.ready': 'Ready', 'annotate.not_ready': 'Not set up', 'annotate.setup': 'Set up',
  'annotate.not_connected': 'Not connected', 'annotate.checking': 'Checking…', 'import.no_images.project': 'No images in {tool}',
}, 'en');

/** Runs initImport with the API answering from `routes` (path -> body); returns the root and the paths asked. */
async function render(routes) {
  const root = document.createElement('div');
  root.id = 'import-root';
  document.body.appendChild(root);
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(url);
    const body = routes[url];
    return { ok: body !== undefined, status: body !== undefined ? 200 : 404, text: async () => JSON.stringify(body ?? {}) };
  };
  try {
    initImport();
    await new Promise((resolve) => setTimeout(resolve, 30));
  } finally {
    globalThis.fetch = real;
    root.remove();
  }
  return { root, asked };
}

test('without a CVAT login the page offers the setup and does not ask CVAT for projects', async () => {
  const { root, asked } = await render({ '/api/annotation-tools': [{ tool: 'cvat', configured: false }] });
  assert.deepEqual(asked, ['/api/annotation-tools']);
  const card = root.querySelector('.tool-card[data-tool="cvat"]');
  assert.ok(card.classList.contains('selected'));
  assert.equal(card.querySelector('.ready-badge').textContent, 'Not set up');
  assert.equal(card.querySelector('.tool-card-foot button').textContent, 'Set up');
});

test('with a login it lists the remote projects', async () => {
  const { root, asked } = await render({
    '/api/annotation-tools': [{ tool: 'cvat', configured: true }],
    '/api/import/cvat/projects': { items: [{ id: 4, name: 'Remote cats', kind: 'project', task: 'detection', classes: ['cat'] }] },
  });
  assert.deepEqual(asked, ['/api/annotation-tools', '/api/import/cvat/projects']);
  assert.equal(root.querySelector('.ready-badge').textContent, 'Ready');
  assert.match(root.querySelector('.import-main').textContent, /Remote cats/);
});

test('saved settings of a server that does not answer are not shown as ready (#634)', async () => {
  const { root, asked } = await render({ '/api/annotation-tools': [{ tool: 'cvat', configured: true }] });
  assert.deepEqual(asked, ['/api/annotation-tools', '/api/import/cvat/projects']);   // the list fails (404 here)
  const badge = root.querySelector('.ready-badge');
  assert.equal(badge.textContent, 'Not connected');
  assert.ok(badge.classList.contains('bad'));
});

test('a remote project without images cannot be imported and says why (#634)', async () => {
  const root = document.createElement('div');
  root.id = 'import-root';
  document.body.appendChild(root);
  const routes = {
    '/api/annotation-tools': [{ tool: 'cvat', configured: true }],
    '/api/import/cvat/projects': { items: [{ id: 7, name: 'Empty', kind: 'project', images: 0, classes: ['x'] }] },
  };
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => ({ ok: true, status: 200, text: async () => JSON.stringify(routes[url] ?? {}) });
  try {
    initImport();
    await new Promise((resolve) => setTimeout(resolve, 30));
    window.matchMedia = () => ({ matches: false });   // jsdom has none
    root.querySelector('.import-table tbody input').dispatchEvent(new window.Event('change'));
    const side = root.querySelector('#import-side');
    assert.ok(side.querySelector('button[type="submit"]').disabled);
    assert.match(side.textContent, /No images in CVAT/);
  } finally {
    globalThis.fetch = real;
    root.remove();
  }
});
