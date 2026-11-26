import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body><div id="toasts"></div></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
// jsdom may not parse `dialog:modal` (toasts look for an open modal): such a selector then matches nothing
const query = window.document.querySelectorAll.bind(window.document);
window.document.querySelectorAll = (selector) => {
  try { return query(selector); } catch (err) { return []; }
};
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const { createSync } = await import('../../../src/activelabel/web/static/js/integrations/sync.js');

setMessages({ 'annotate.tool.cvat': 'CVAT', 'sync.started': 'Comparing with {tool}' }, 'en');

/** A page object with what the sync reads; `started` collects the jobs handed to jobStarted. */
function makePage() {
  const started = [];
  const ctx = { jobs: [], links: [], syncTool: {}, syncDismissed: new Set(), panels: {} };
  const page = {
    ctx, projectId: 5, started, isActive: () => false, isRemoteGone: () => false, isStale: () => false,
    jobError: () => '', jobStarted: async (job) => { started.push(job); }, syncJobTool: () => 'cvat',
    planIsEmpty: () => true, refreshProject: async () => true, diffView: () => null, openLightbox: () => {},
  };
  return { page, ctx, sync: createSync(page) };
}

/** Calls fn with the API answering `body`; returns the [url, init] pairs asked. */
async function withApi(body, fn) {
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    asked.push([url, init]);
    return { ok: true, status: 200, text: async () => JSON.stringify(body) };
  };
  try {
    await fn();
  } finally {
    globalThis.fetch = real;
  }
  return asked;
}

test('loadLinks keeps the links of the project in ctx', async () => {
  const { ctx, sync } = makePage();
  const asked = await withApi({ links: [{ tool: 'cvat', remote_name: 'Cats' }] }, () => sync.loadLinks());
  assert.equal(asked[0][0], '/api/projects/5/links');
  assert.deepEqual(ctx.links, [{ tool: 'cvat', remote_name: 'Cats' }]);
});

test('startSync asks for a comparison, remembers its tool and hands the job to the activity strip', async () => {
  const { page, ctx, sync } = makePage();
  const asked = await withApi({ id: 'j1', kind: 'sync_plan' }, () => sync.startSync('cvat', { task: 2 }));
  assert.equal(asked[0][0], '/api/projects/5/sync/cvat');
  assert.equal(asked[0][1].method, 'POST');
  assert.equal(asked[0][1].body, '{"task":2}');
  assert.equal(ctx.syncTool.j1, 'cvat');
  assert.deepEqual(page.started.map((j) => j.id), ['j1']);
  assert.equal(document.getElementById('toasts').lastElementChild.dataset.text, 'Comparing with CVAT');
});

test('syncBlock is the sync section of the Annotations step', () => {
  const { sync } = makePage();
  const block = sync.syncBlock();
  assert.equal(block.node.id, 'annotate-sync');
  assert.equal(typeof block.draw, 'function');
  assert.equal(typeof block.reveal, 'function');
});

test('the review shows task, class and project renames in their direction, and names kept on both sides (#587)', () => {
  setMessages({
    'annotate.tool.cvat': 'CVAT', 'sync.push': 'To {tool}', 'sync.pull': 'From {tool}', 'sync.row_tasks_rename': 'Task names',
    'sync.row_classes_rename': 'Renamed classes', 'sync.row_project_rename': 'Project name', 'sync.rename_item': '{old} → {new}',
    'sync.rename_conflicts': 'Renamed on both sides:', 'sync.rename_conflict_item': 'here "{here}", in {tool} "{there}"',
  }, 'en');
  const { ctx, sync } = makePage();
  ctx.jobs = [{
    id: 'p1', kind: 'sync_plan', status: 'done', created: '2026-11-26T10:00:00',
    result: {
      plan_id: 'x1', tool: 'cvat', remote: { name: 'P' }, scope: { kind: 'project' }, direction: 'both',
      tasks_rename: [{ task: 1, remote_task: '7', old: 'day 1', new: 'Day one', direction: 'push' }, { task: 2, remote_task: '8', old: 'b', new: 'Birds', direction: 'pull' }],
      renamed_classes: [{ from: 'car', to: 'vehicle' }],
      project_rename: { old: 'Street', new: 'Streets', direction: 'push' },
      tasks_rename_conflicts: [{ task: 3, remote_task: '9', local: 'A', remote: 'B' }],
    },
  }];
  const block = sync.syncBlock();
  document.body.append(block.node);
  block.draw(true);
  const [push, pull] = [...block.node.querySelectorAll('.sync-dir')].map((s) => s.textContent);
  assert.match(push, /Project name.*Street → Streets/);
  assert.match(push, /Task names.*day 1 → Day one/);
  assert.doesNotMatch(push, /Birds/);
  assert.match(pull, /Task names.*b → Birds/);
  assert.match(pull, /Renamed classes.*car → vehicle/);
  assert.match(block.node.textContent, /Renamed on both sides:.*here "A", in CVAT "B"/);
  block.node.remove();
});
