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
const { createActivities } = await import('../../../src/activelabel/web/static/js/cv/activities.js');

setMessages({
  'jobs.title': 'Activities', 'jobs.running_n_one': '{n} running', 'jobs.running_n_other': '{n} running',
  'job.status.failed': 'Failed', 'jobs.dismiss': 'Dismiss', 'sync.stale': 'Compare again',
  'error.disk_full': 'Disk full',
}, 'en');

/** A page object with the parts the activity strip reads, and a fresh strip. */
function makePage(jobs = []) {
  const jobsBox = document.createElement('section');
  document.body.appendChild(jobsBox);
  const ctx = {
    jobs, jobStatus: new Map(), panels: {}, watchers: [], syncDismissed: new Set(), openWhenDone: new Map(), pollTimer: null,
  };
  const page = {
    ctx, jobsBox, projectId: 9, planIsEmpty: () => true, refreshProject: async () => true, showMissing: () => {}, showTab: () => {},
  };
  return { page, ctx, jobsBox, acts: createActivities(page) };
}

test('a job is active while queued or running, and kindActive looks at the kinds given', () => {
  const { acts } = makePage([{ id: 'a', kind: 'curation', status: 'running' }, { id: 'b', kind: 'training', status: 'done' }]);
  assert.ok(acts.isActive({ status: 'queued' }) && acts.isActive({ status: 'running' }));
  assert.ok(!acts.isActive({ status: 'failed' }));
  assert.ok(acts.kindActive('curation', 'export'));
  assert.ok(!acts.kindActive('training'));
});

test('a stale comparison and a deleted remote project are told apart from other failures', () => {
  const { acts } = makePage();
  assert.ok(acts.isStale({ kind: 'sync_apply', error_code: 'plan_not_found' }));
  assert.ok(!acts.isStale({ kind: 'sync_plan', error_code: 'plan_not_found' }));
  assert.ok(acts.isRemoteGone({ kind: 'sync_plan', error_code: 'remote_gone' }));
  assert.equal(acts.jobError({ kind: 'sync_apply', error_code: 'sync_remote_changed' }), 'Compare again');
  assert.equal(acts.jobError({ kind: 'curation', error_code: 'disk_full', error: 'x' }), 'Disk full');
});

test('the strip is hidden without jobs and shows the running ones', () => {
  const { ctx, jobsBox, acts } = makePage();
  acts.renderJobs();
  assert.equal(jobsBox.hidden, true);
  ctx.jobs = [{ id: 'r', kind: 'curation', status: 'running', created: '2026-01-01T00:00:00' }];
  acts.renderJobs();
  assert.equal(jobsBox.hidden, false);
  assert.match(jobsBox.textContent, /1 running/);
  jobsBox.remove();
});

test('pollJobs reads the jobs of the project and draws them', async () => {
  const { ctx, jobsBox, acts } = makePage();
  const asked = [];
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    asked.push(url);
    return { ok: true, status: 200, text: async () => JSON.stringify([{ id: 'f', kind: 'curation', status: 'failed', error_code: 'disk_full', created: '2026-01-01T00:00:00' }]) };
  };
  try {
    await acts.pollJobs();
  } finally {
    globalThis.fetch = real;
  }
  assert.deepEqual(asked, ['/api/projects/9/jobs']);
  assert.equal(ctx.jobs.length, 1);
  assert.equal(ctx.pollTimer, null);  // nothing running: no next poll
  assert.equal(jobsBox.hidden, false);
  jobsBox.remove();
});

/** A tab the browser opened for the FiftyOne App. */
const fakeTab = () => ({ closed: false, location: { href: 'about:blank' }, close() { this.closed = true; } });

/** Runs one poll that finds `job` finished after the strip saw it running, with `tab` reserved for it. */
async function finish(job, tab) {
  const { ctx, acts } = makePage();
  ctx.jobStatus.set(job.id, 'running');
  ctx.openWhenDone.set(job.id, tab);
  globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => [job], text: async () => JSON.stringify([job]) });
  await acts.pollJobs();
  return ctx;
}

test('a finished FiftyOne job takes the reserved tab to the App', async () => {
  const tab = fakeTab();
  const ctx = await finish({ id: 'f', kind: 'fiftyone', status: 'done', result: { url: 'http://localhost:5161/datasets/x' }, created: '1' }, tab);
  assert.equal(tab.location.href, 'http://localhost:5161/datasets/x');
  assert.equal(tab.closed, false);
  assert.equal(ctx.openWhenDone.size, 0);
});

test('an edit session takes the reserved tab to the link of its view', async () => {
  const tab = fakeTab();
  const result = { links: [{ label: 'FiftyOne', url: 'http://localhost:5161/datasets/x?view=v' }], session: { id: 's' } };
  await finish({ id: 'e', kind: 'annotate_send', status: 'done', result, created: '1' }, tab);
  assert.equal(tab.location.href, 'http://localhost:5161/datasets/x?view=v');
});

test('a job that failed or was stopped closes the reserved tab and a failure says why', async () => {
  const tab = fakeTab();
  await finish({ id: 'g', kind: 'fiftyone', status: 'failed', error: 'No free port for the FiftyOne App', created: '1' }, tab);
  assert.equal(tab.closed, true);
  assert.equal(tab.location.href, 'about:blank');
  const stopped = fakeTab();
  await finish({ id: 'h', kind: 'fiftyone', status: 'cancelled', created: '1' }, stopped);
  assert.equal(stopped.closed, true);
});

test('an address that is not http is never opened', async () => {
  const tab = fakeTab();
  await finish({ id: 'i', kind: 'fiftyone', status: 'done', result: { url: 'javascript:alert(1)' }, created: '1' }, tab);
  assert.equal(tab.location.href, 'about:blank');
  assert.equal(tab.closed, true);
});
