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
      project_rename_conflict: { local: 'Roads', remote: 'Streets 2' },
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
  assert.match(block.node.textContent, /here "Roads", in CVAT "Streets 2"/);
  block.node.remove();
});

/** Draws a review with `plan` (merged over a plan with no changes) and returns the block. */
function drawPlan(plan) {
  setMessages({
    'annotate.tool.cvat': 'CVAT', 'sync.push': 'To {tool}', 'sync.pull': 'From {tool}', 'sync.dir_empty': 'Nothing',
    'sync.row_not_converted': 'Shapes that stay only in {tool}', 'sync.not_converted_item': '{kind}: {n}',
    'sync.shape.ellipse': 'ellipses', 'sync.shape.mask': 'masks', 'sync.kind_geometry': 'Same classes: the position differs.',
    'sync.kind_labels': 'Edited on both sides', 'sync.kind_removed_local': 'Removed here', 'sync.kind_removed_remote': 'Removed in {tool}',
    'sync.no_labels': 'no labels', 'sync.side_local': 'Here', 'sync.pull_class_only_one': '{n} of them changes only by the class name', 'sync.pull_class_only_other': '{n} of them change only by the class name',
  }, 'en');
  const { ctx, sync } = makePage();
  ctx.jobs = [{
    id: 'p2', kind: 'sync_plan', status: 'done', created: '2026-11-26T10:00:00',
    result: { plan_id: 'x2', tool: 'cvat', remote: { name: 'P' }, scope: { kind: 'project' }, direction: 'both', ...plan },
  }];
  const block = sync.syncBlock();
  document.body.append(block.node);
  block.draw(true);
  return block;
}

test('the review lists the shapes that stay in the tool, with the total and each kind (#519)', () => {
  const block = drawPlan({ not_converted: { ellipse: 2, mask: 3, spiral: 1, cuboid: 0 } });
  const pull = block.node.querySelectorAll('.sync-dir')[1].textContent;
  assert.match(pull, /Shapes that stay only in CVAT.*6/);
  assert.match(pull, /ellipses: 2/);
  assert.match(pull, /masks: 3/);
  assert.match(pull, /spiral: 1/); // a kind without a text shows its raw name
  assert.doesNotMatch(pull, /cuboid/); // a zero count is left out
  assert.doesNotMatch(block.node.querySelectorAll('.sync-dir')[0].textContent, /Shapes that stay/);
  block.node.remove();
});

test('the review has no line for shapes that stay in the tool when there are none', () => {
  for (const not_converted of [{}, undefined, { mask: 0 }]) {
    const block = drawPlan({ not_converted });
    assert.doesNotMatch(block.node.textContent, /Shapes that stay/);
    block.node.remove();
  }
});

test('a conflict with the same classes on both sides says that the position differs (#684)', () => {
  const side = (n) => ({ per_class: { car: n } });
  const block = drawPlan({ conflicts: [
    { image_id: 'a', name: 'a.png', kind: 'labels', local: side(2), remote: side(2) },
    { image_id: 'b', name: 'b.png', kind: 'labels', local: side(2), remote: side(3) },
    { image_id: 'c', name: 'c.png', kind: 'removed_local', local: side(2), remote: side(2) },
    { image_id: 'd', name: 'd.png', kind: 'removed_remote', local: side(2), remote: side(2) },
  ] });
  const rows = [...block.node.querySelectorAll('li.conflict')];
  assert.equal(rows.length, 4);
  const why = rows.map((r) => r.querySelector('.conflict-why')?.textContent ?? null);
  assert.deepEqual(why, ['Same classes: the position differs.', null, null, null]);
  block.node.remove();
});

test('the result card lists the renamed classes with their names', () => {
  setMessages({
    'annotate.tool.cvat': 'CVAT', 'sync.result_title': 'Done', 'sync.res_pulled': 'Images updated here', 'sync.res_renamed_classes': 'Classes renamed',
    'sync.rename_item': '{old} → {new}', 'common.close': 'Close',
  }, 'en');
  const { ctx, sync } = makePage();
  ctx.jobs = [{ id: 'a2', kind: 'sync_apply', status: 'done', created: '2026-11-26T10:00:00', result: { pulled: 10, renamed_classes: [{ from: 'pessoa', to: 'pedestre' }] } }];
  const block = sync.syncBlock();
  document.body.append(block.node);
  block.draw(true);
  const text = block.node.querySelector('#sync-result').textContent;
  assert.match(text, /Classes renamed.*1/);
  assert.match(text, /pessoa → pedestre/);
  block.node.remove();
});

test('the review says how many images change only by the class name when classes were renamed', () => {
  const block = drawPlan({ renamed_classes: [{ from: 'a', to: 'b' }], pull: 10, diff_pull: { images_changed: 3 } });
  assert.match(block.node.querySelectorAll('.sync-dir')[1].textContent, /7 of them change only by the class name/);
  block.node.remove();
  const same = drawPlan({ renamed_classes: [{ from: 'a', to: 'b' }], pull: 3, diff_pull: { images_changed: 3 } });
  assert.doesNotMatch(same.node.textContent, /change only by the class name/);
  same.node.remove();
  const none = drawPlan({ pull: 10, diff_pull: { images_changed: 3 } });
  assert.doesNotMatch(none.node.textContent, /change only by the class name/);
  none.node.remove();
});

test('the result card lists the renames, not "everything is in sync" (#587)', () => {
  setMessages({
    'annotate.tool.cvat': 'CVAT', 'sync.result_title': 'Done', 'sync.in_sync': 'Everything is in sync',
    'sync.res_tasks_renamed': 'Tasks renamed', 'sync.res_project_renamed': 'Project name updated', 'common.close': 'Close',
  }, 'en');
  const { ctx, sync } = makePage();
  ctx.jobs = [{ id: 'a1', kind: 'sync_apply', status: 'done', created: '2026-11-26T10:00:00', result: { tasks_renamed: 2, project_renamed: true, pulled: 0 } }];
  const block = sync.syncBlock();
  document.body.append(block.node);
  block.draw(true);
  const text = block.node.querySelector('#sync-result').textContent;
  assert.match(text, /Tasks renamed.*2/);
  assert.match(text, /Project name updated.*1/);
  assert.doesNotMatch(text, /Everything is in sync/);
  block.node.remove();
});
