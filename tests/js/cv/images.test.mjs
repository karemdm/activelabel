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
const { imagesPanel } = await import('../../../src/activelabel/web/static/js/cv/images.js');

setMessages({ 'step.images': 'Images' }, 'en');

/** The page object with a tabs stand-in that records what is selected. */
function makePage(hasImages) {
  const selected = [];
  const ctx = { project: { id: 2, task: 'detection', classes: [], counts: {} }, jobs: [], panels: {}, watchers: [], busy: new Set() };
  const page = {
    ctx, projectId: 2, hasImages: () => hasImages, jobStarted: async () => {}, refreshCounts: async () => {},
    refreshProject: async () => true, showTab: () => {},
    tabsView: () => ({ node: document.createElement('div'), remembered: false, select: (id) => selected.push(id), setCount: () => {}, current: () => 'gallery' }),
  };
  return { page, selected };
}

test('the Images step opens on its tasks when the project has no images, and loads tasks and images', async () => {
  const { page, selected } = makePage(false);
  const panel = imagesPanel(page);
  assert.equal(panel.node.id, 'panel-images');
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(url.split('?')[0]);
    const body = url.includes('/tasks') ? { tasks: [] } : { total: 0, items: [] };
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try {
    panel.mount();
    await new Promise((resolve) => setTimeout(resolve, 30));
  } finally {
    globalThis.fetch = real;
  }
  assert.deepEqual(selected, ['tasks']);
  assert.ok(asked.includes('/api/projects/2/tasks'));
  assert.ok(asked.includes('/api/projects/2/images'));
});

test('a deleted dataset version stops filtering the gallery', () => {
  const { page, selected } = makePage(true);
  const panel = imagesPanel(page);
  const real = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify({ total: 0, items: [] }) });
  try {
    panel.showDataset(4, 'train');
    assert.deepEqual(selected, ['gallery']);
    panel.forgetDataset(4);
  } finally {
    globalThis.fetch = real;
  }
});
