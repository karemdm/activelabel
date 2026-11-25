import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { initProjects, projectCard } = await import('../../../src/activelabel/web/static/js/pages/projects.js');

setMessages({ 'projects.empty_title': 'No projects yet', 'task.detection': 'Detection', 'project.created': 'Created' }, 'en');

/** Runs initProjects in a fresh #projects-root with the API answering `projects`. */
async function render(projects) {
  const page = document.getElementById('projects-root');  // a page that already has one (a browser run)
  if (page) page.id = '';
  const root = document.createElement('div');
  root.id = 'projects-root';
  document.body.appendChild(root);
  const real = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify(projects) });
  try {
    await initProjects();
  } finally {
    globalThis.fetch = real;
    root.remove();
    if (page) page.id = 'projects-root';
  }
  return root;
}

test('without projects the home page invites to create or import one', async () => {
  const root = await render([]);
  assert.equal(root.querySelector('h2').textContent, 'No projects yet');
  assert.deepEqual([...root.querySelectorAll('a')].map((a) => a.getAttribute('href')), ['/projects/new', '/projects/import']);
});

test('each project is a card that links to it, with its task and a coverage bar when it has images', async () => {
  const root = await render([
    { id: 1, name: 'Cats', task: 'detection', counts: { active: 4, annotated: 1 }, created: '2026-01-02T00:00:00' },
    { id: 2, name: 'Empty', task: 'detection', counts: {}, created: '2026-01-02T00:00:00' },
  ]);
  const cards = root.querySelectorAll('.project-card');
  assert.equal(cards.length, 2);
  assert.equal(cards[0].getAttribute('href'), '/projects/1');
  assert.equal(cards[0].querySelector('h2').textContent, 'Cats');
  assert.ok(cards[0].querySelector('.badge.task-detection'));
  assert.equal(cards[0].querySelector('[role="progressbar"]').getAttribute('aria-valuenow'), '25');
  assert.equal(cards[1].querySelector('[role="progressbar"]'), null);
  assert.equal(projectCard({ id: 3, name: 'x', task: 'detection', description: 'About' }).querySelector('.project-desc').textContent, 'About');
});
