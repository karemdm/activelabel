import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const open = await import('../../../src/activelabel/web/static/js/integrations/open.js');

const { CVAT_FIXABLE, cvatProjectGone, cvatTarget, fiftyoneReason, cvatOpenRow, fiftyoneOpenRow, headerCvatButton, headerFiftyoneButton, sentence } = open;

setMessages({
  'annotate.tool.cvat': 'CVAT', 'annotate.tool.fiftyone': 'FiftyOne', 'annotate.open_in': 'Open in {tool}',
  'open.cvat_project': 'Open the project in CVAT', 'open.cvat_task': 'Open task {task} in CVAT', 'open.cvat_job': 'Open job {n} of task {task} in CVAT',
  'open.cvat_back': 'Edits come back through the comparison.', 'open.cvat_not_configured': 'CVAT is not set up.',
  'open.cvat_not_synced': 'Not in CVAT yet.', 'open.cvat_go': 'Set it up and compare in Annotations → CVAT.',
  'open.fo_back': 'Edits come back through an edit session.', 'open.fo_no_images': 'Add images first.',
  'tasks.cvat_not_synced': 'Synchronise to create this task there', 'open.cvat_remote_gone': 'The CVAT project was deleted.', 'tasks.fo_no_keypoints': 'FiftyOne does not edit keypoints',
}, 'en');

const LINK = { tool: 'cvat', url: 'https://cvat.example/projects/7', remote_name: 'p' };
const TASK = {
  id: 1, name: 'T1', cvat: 'ready',
  remote: { task: 40, url: 'https://cvat.example/tasks/40', jobs: [{ n: 2, id: 91, url: 'https://cvat.example/tasks/40/jobs/91' }] },
};

test('CVAT opens the linked project, task or job, or says why it cannot yet', () => {
  assert.deepEqual(cvatTarget({ link: null, configured: false }), { reason: 'open.cvat_not_configured' });
  assert.deepEqual(cvatTarget({ link: null, configured: true }), { reason: 'open.cvat_not_synced' });
  assert.deepEqual(cvatTarget({ link: LINK, configured: true }), { url: LINK.url, label: 'open.cvat_project' });
  assert.deepEqual(cvatTarget({ link: { ...LINK, url: '' }, configured: true }), { reason: 'open.cvat_no_address' });
  assert.deepEqual(cvatTarget({ link: LINK, configured: true, task: TASK }), { url: TASK.remote.url, label: 'open.cvat_task' });
  assert.deepEqual(cvatTarget({ link: LINK, configured: true, task: TASK, job: 2 }), { url: TASK.remote.jobs[0].url, label: 'open.cvat_job' });
  assert.deepEqual(cvatTarget({ link: LINK, configured: true, task: TASK, job: 5 }), { reason: 'tasks.cvat_not_synced' });
  assert.deepEqual(cvatTarget({ link: LINK, configured: true, task: { id: 2, cvat: 'unreachable' } }), { reason: 'tasks.cvat_unreachable' });
});

test('the CVAT row is a link to the chosen scope and how the edits come back, or only the reason, as a sentence (#602)', () => {
  const row = cvatOpenRow({ link: LINK, configured: true, task: TASK, job: 2 });
  const a = row.querySelector('a#cvat-open');
  assert.equal(a.getAttribute('href'), TASK.remote.jobs[0].url);
  assert.equal(a.getAttribute('target'), '_blank');
  assert.match(a.textContent, /Open job 2 of task T1 in CVAT/);
  assert.match(row.textContent, /Edits come back through the comparison\./);
  const none = cvatOpenRow({ link: null, configured: true });
  assert.equal(none.querySelector('a'), null);
  assert.match(none.querySelector('#cvat-open-reason').textContent, /Not in CVAT yet\./);
  assert.doesNotMatch(none.textContent, /Edits come back/, 'nothing to open: the way back is not said');
  assert.equal(sentence('CVAT did not answer: try again in a moment'), 'CVAT did not answer: try again in a moment.');
  assert.equal(sentence('Done.'), 'Done.');
});

test('FiftyOne opens only with images; a keypoints project says FiftyOne shows its points without editing them', () => {
  assert.equal(fiftyoneReason({ counts: { active: 3 } }), null);
  assert.equal(fiftyoneReason({ counts: { active: 0 } }), 'open.fo_no_images');
  const page = { projectId: 4, ctx: { project: { task: 'detection', counts: { active: 3 } }, links: [], openWhenDone: new Set() } };
  const row = fiftyoneOpenRow(page);
  assert.equal(row.querySelector('#fo-open').disabled, false);
  assert.match(row.textContent, /Edits come back through an edit session\./);
  page.ctx.project = { task: 'keypoints', counts: { active: 3 } };
  assert.match(fiftyoneOpenRow(page).textContent, /does not edit keypoints/);
  page.ctx.project = { task: 'detection', counts: { active: 0 } };
  const empty = fiftyoneOpenRow(page);
  assert.equal(empty.querySelector('#fo-open').disabled, true);
  assert.match(empty.textContent, /Add images first\./);
});

test('the header opens the linked CVAT project, or takes the person to Annotations → CVAT', () => {
  const calls = [];
  const page = {
    projectId: 4,
    showTab: (step) => calls.push(step),
    ctx: { project: { counts: { active: 0 } }, links: [LINK], openWhenDone: new Set(), panels: { annotations: { reveal: (what) => calls.push(what) } } },
  };
  assert.equal(headerCvatButton(page).getAttribute('href'), LINK.url);
  page.ctx.links = [];
  const button = headerCvatButton(page);
  assert.equal(button.tagName, 'BUTTON');
  assert.equal(button.title, 'Set it up and compare in Annotations → CVAT.');
  button.click();
  assert.deepEqual(calls, ['annotations', 'sync']);
  /* #602/#675: not disabled, so a tap on a touch screen still says why */
  const fo = headerFiftyoneButton(page);
  assert.equal(fo.disabled, false);
  assert.equal(fo.getAttribute('aria-disabled'), 'true');
  assert.ok(fo.classList.contains('is-unavailable'));
  assert.equal(fo.title, 'Add images first.');
});

test('a task deleted in CVAT, or in a deleted CVAT project, has its own reason (#684)', () => {
  const reason = (cvat) => cvatTarget({ link: LINK, configured: true, task: { id: 2, cvat } });
  assert.deepEqual(reason('removed_remote'), { reason: 'tasks.cvat_removed_remote' });
  assert.deepEqual(reason('remote_gone'), { reason: 'tasks.cvat_remote_gone' });
  assert.deepEqual(reason('not_linked'), { reason: 'tasks.cvat_not_linked' });
});

test('the reasons a comparison can fix are the fixable ones; not set up and unreachable are not', () => {
  assert.deepEqual([...CVAT_FIXABLE].sort(), ['not_linked', 'not_synchronised', 'remote_gone', 'removed_remote']);
  assert.equal(CVAT_FIXABLE.has('not_configured'), false);
  assert.equal(CVAT_FIXABLE.has('unreachable'), false);
});

const plan = (id, status, created, kind = 'sync_plan') => ({ id, kind, status, created });
const goneFor = (...ids) => (job) => ids.includes(job.id);

test('cvatProjectGone is true only when the last finished comparison failed because the project is gone', () => {
  const page = (jobs, isRemoteGone) => ({ ctx: { jobs }, isRemoteGone });
  const failed = plan('a', 'failed', '2026-01-02');
  assert.equal(cvatProjectGone(page([failed], goneFor('a'))), true);
  assert.equal(cvatProjectGone(page([plan('b', 'done', '2026-01-03'), failed], goneFor('a'))), false);  // done after failed
  assert.equal(cvatProjectGone(page([failed, plan('b', 'done', '2026-01-01')], goneFor('a'))), true);  // done before
  assert.equal(cvatProjectGone(page([failed], goneFor())), false);  // failed for another reason
  assert.equal(cvatProjectGone(page([], goneFor('a'))), false);  // no jobs
  assert.equal(cvatProjectGone({ ctx: {} }), false);
  assert.equal(cvatProjectGone(page([plan('c', 'running', '2026-01-09'), failed], goneFor('a', 'c'))), true);  // running is ignored
  assert.equal(cvatProjectGone(page([plan('d', 'failed', '2026-01-09', 'sync_apply')], goneFor('d'))), false);
  assert.equal(cvatProjectGone({ ctx: { jobs: [failed] } }), false);  // the page cannot tell
});

test('the header button of a deleted CVAT project goes to the CVAT tab and says why', () => {
  const calls = [];
  const page = {
    projectId: 4,
    showTab: (step) => calls.push(step),
    isRemoteGone: () => true,
    ctx: { project: { counts: { active: 0 } }, links: [LINK], jobs: [plan('a', 'failed', '2026-01-02')], panels: { annotations: { reveal: (what) => calls.push(what) } } },
  };
  const button = headerCvatButton(page);
  assert.equal(button.tagName, 'BUTTON');
  assert.equal(button.getAttribute('href'), null);
  assert.equal(button.title, 'The CVAT project was deleted.');
  button.click();
  assert.deepEqual(calls, ['annotations', 'sync']);
  page.ctx.jobs = [];  // no failed comparison: the link opens as usual
  assert.equal(headerCvatButton(page).getAttribute('href'), LINK.url);
});
