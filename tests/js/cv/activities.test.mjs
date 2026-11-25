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
    jobs, jobStatus: new Map(), panels: {}, watchers: [], syncDismissed: new Set(), openWhenDone: new Set(), pollTimer: null,
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
