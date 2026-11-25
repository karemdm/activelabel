import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/projects/9' });
for (const [name, value] of Object.entries({
  window, document: window.document, Node: window.Node, localStorage: window.localStorage,
  location: window.location, history: window.history, requestAnimationFrame: (fn) => setTimeout(fn, 0),
})) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { diffView, initProject } = await import('../../../src/activelabel/web/static/js/cv/project.js');

setMessages({ 'project.gone_title': 'This project no longer exists', 'annotate.diff_class': 'Class' }, 'en');

test('diffView lists the classes that changed, and only those', () => {
  const view = diffView(3, { per_class: { cat: { added: 2, removed: 0, edited: 1 }, dog: { added: 0, removed: 0, edited: 0 } } });
  const rows = [...view.querySelectorAll('.diff-table tbody tr')];
  assert.equal(rows.length, 1);
  assert.match(rows[0].textContent, /cat/);
});

test('a project that does not exist (404) shows the "no longer exists" card', async () => {
  const page = document.getElementById('project-root');  // a page that already has one (a browser run)
  if (page) page.id = '';
  const root = document.createElement('div');
  root.id = 'project-root';
  document.body.appendChild(root);
  window.VL = { ...(window.VL || {}), projectId: 999 };
  const real = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: false, status: 404, text: async () => JSON.stringify({ code: 'project_not_found', detail: 'x', params: {} }) });
  try {
    initProject();
    await new Promise((resolve) => setTimeout(resolve, 50));
  } finally {
    globalThis.fetch = real;
  }
  assert.equal(root.querySelector('h1').textContent, 'This project no longer exists');
  root.remove();
  if (page) page.id = 'project-root';
});
