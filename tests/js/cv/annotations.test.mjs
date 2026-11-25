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
const { annotationsPanel } = await import('../../../src/activelabel/web/static/js/cv/annotations.js');

setMessages({ 'step.annotations': 'Annotations' }, 'en');

test('the Annotations step is a hidden panel with its tabs, and it can open pre-annotation on a model', () => {
  const tabs = [];
  const page = {
    ctx: { project: { id: 1, task: 'detection', classes: [], counts: {} }, jobs: [], links: [], panels: {}, watchers: [], syncDismissed: new Set(), syncTool: {} },
    projectId: 1, refreshProject: async () => true,
    tabsView: (key, label, list) => {
      tabs.push(...list.map((x) => x.id));
      return { node: document.createElement('div'), select: (id) => tabs.push(`selected:${id}`), current: () => list[0].id, setCount: () => {} };
    },
    isActive: () => false, jobStarted: async () => {}, startSync: async () => {}, diffView: () => null, issueName: (x) => x,
    syncBlock: () => ({ node: document.createElement('div'), draw: () => {}, reveal: () => {} }),
  };
  const panel = annotationsPanel(page);
  assert.equal(panel.node.id, 'panel-annotations');
  assert.equal(panel.node.hidden, true);
  assert.ok(tabs.includes('preannotate') && tabs.includes('cvat') && tabs.includes('fiftyone'));
  panel.preannotateWith(3);
  assert.ok(tabs.includes('selected:preannotate'));
});
