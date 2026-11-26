import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node, Event: window.Event })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const queue = await import('../../../src/activelabel/web/static/js/cv/queue.js');
const { HEAVY_KINDS, cancelPath, loadQueue, queueRow, rowsFromJobs, rowsFromQueue } = queue;

setMessages({
  'job.kind.training': 'Training', 'job.kind.export': 'Model export', 'queue.running': 'Running', 'queue.position': 'Waiting · #{n}',
  'queue.waiting': 'Waiting', 'queue.cancel': 'Cancel', 'queue.no_cancel': 'Cannot be cancelled',
}, 'en');

/** Calls fn with the API answering from `answers` by path (a missing path is a 404). */
async function withApi(answers, fn) {
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const body = answers[url.split('?')[0]];
    return { ok: body !== undefined, status: body !== undefined ? 200 : 404, text: async () => JSON.stringify(body ?? { detail: 'Not Found' }) };
  };
  try {
    return await fn();
  } finally {
    globalThis.fetch = real;
  }
}

test('the queue shows the heavy kinds of work only', () => {
  assert.deepEqual(HEAVY_KINDS, ['training', 'preannotate', 'analyze', 'export']);
});

test('each kind is cancelled through its own route; an export cannot be', () => {
  assert.equal(cancelPath({ kind: 'training', project_id: 2, params: { number: 4 } }), '/projects/2/runs/4/cancel');
  assert.equal(cancelPath({ kind: 'preannotate', project_id: 2, params: { number: 1 } }), '/projects/2/preannotations/1/cancel');
  assert.equal(cancelPath({ kind: 'analyze', project_id: 2, id: 'a b' }), '/jobs/a%20b/cancel');
  assert.equal(cancelPath({ kind: 'export', project_id: 2, params: { number: 1 } }), null);
});

test('from the activity lists: active heavy work, running first, the waiting ones numbered in queue order', () => {
  const projects = [{ id: 1, name: 'A' }, { id: 2, name: 'B' }];
  const rows = rowsFromJobs(projects, {
    1: [
      { id: 'j3', kind: 'training', status: 'queued', created: '2026-01-01T00:03', params: { number: 3 } },
      { id: 'j0', kind: 'training', status: 'done', created: '2026-01-01T00:00', params: { number: 1 } },
      { id: 'jx', kind: 'fiftyone', status: 'running', created: '2026-01-01T00:00' },
    ],
    2: [
      { id: 'j2', kind: 'preannotate', status: 'queued', created: '2026-01-01T00:02', params: { number: 1 } },
      { id: 'j1', kind: 'analyze', status: 'running', created: '2026-01-01T00:01', done: 3, total: 8 },
    ],
  });
  assert.deepEqual(rows.map((r) => [r.id, r.status, r.position, r.project_name]), [['j1', 'running', null, 'B'], ['j2', 'queued', 1, 'B'], ['j3', 'queued', 2, 'A']]);
  assert.equal(rows[2].cancel, '/projects/1/runs/3/cancel');
});

test('from the global queue: its positions, and Cancel through the queue route', () => {
  const rows = rowsFromQueue({ items: [
    { id: 'q2', project_id: 1, project_name: 'A', kind: 'training', status: 'queued', position: 1 },
    { id: 'q1', project_id: 1, project_name: 'A', kind: 'export', status: 'running', done: 1, total: 2 },
    { id: 'q0', project_id: 1, kind: 'import_folder', status: 'running' },
  ] });
  assert.deepEqual(rows.map((r) => [r.id, r.position, r.cancel]), [['q1', null, '/queue/q1/cancel'], ['q2', 1, '/queue/q2/cancel']]);
});

test('loadQueue reads the global queue when it exists, else the activity lists of the projects', async () => {
  const global = await withApi({ '/api/queue': { items: [{ id: 'q', project_id: 1, kind: 'training', status: 'running' }] } }, loadQueue);
  assert.deepEqual(global.map((r) => r.id), ['q']);
  const fallback = await withApi({
    '/api/projects': [{ id: 7, name: 'P' }],
    '/api/projects/7/jobs': [{ id: 't', project_id: 7, kind: 'training', status: 'running', params: { number: 1 } }],
  }, loadQueue);
  assert.deepEqual(fallback.map((r) => [r.id, r.project_name]), [['t', 'P']]);
});

test('a row names the work, links to its project step, says where it is and offers Cancel', () => {
  const seen = [];
  const row = queueRow({ id: 'j', project_id: 3, project_name: 'Cats', kind: 'training', status: 'queued', position: 2, done: 0, total: 0, message: '', cancel: '/x' }, (r) => seen.push(r.id));
  assert.equal(row.querySelector('a').getAttribute('href'), '/projects/3#training');
  assert.match(row.textContent, /Training.*Cats.*Waiting · #2/);
  row.querySelector('[data-act="cancel"]').click();
  assert.deepEqual(seen, ['j']);
  const running = queueRow({ id: 'e', project_id: 3, project_name: 'Cats', kind: 'export', status: 'running', done: 1, total: 4, message: '', cancel: null }, () => {});
  assert.ok(running.querySelector('[role="progressbar"]'));
  assert.match(running.textContent, /Cannot be cancelled/);
});
