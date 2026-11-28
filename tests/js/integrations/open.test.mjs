import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const { window } = new JSDOM('<!doctype html><html><body></body></html>');
for (const [name, value] of Object.entries({ window, document: window.document, Node: window.Node })) {
  Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
}
const { setMessages } = await import('../../../src/activelabel/web/static/js/shell/i18n.js');
const open = await import('../../../src/activelabel/web/static/js/integrations/open.js');

const { CVAT_FIXABLE, cvatProjectGone, cvatTarget, fiftyoneReason, cvatOpenRow, fiftyoneOpenRow, sentence } = open;

setMessages({
  'annotate.tool.cvat': 'CVAT', 'annotate.tool.fiftyone': 'FiftyOne', 'annotate.open_in': 'Open in {tool}',
  'open.cvat_project': 'Open the project in CVAT', 'open.cvat_task': 'Open task {task} in CVAT', 'open.cvat_job': 'Open job {n} of task {task} in CVAT',
  'open.cvat_back': 'Edits come back through the comparison.', 'open.cvat_not_configured': 'CVAT is not set up.',
  'open.cvat_not_synced': 'Not in CVAT yet.',
  'open.fo_edit': 'Edit in {tool}', 'open.fo_view': 'Open in {tool} (view only)', 'open.fo_keypoints': 'Keypoints can only be viewed here.', 'open.fo_regression': 'Edit values in ActiveLabel; FiftyOne is for looking.', 'annotate.will_send_one': '{n} image opens in {tool}.', 'annotate.will_send_other': '{n} images open in {tool}.', 'annotate.nothing_to_send': 'No images.', 'open.fo_no_images': 'Add images first.',
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

test('one FiftyOne button edits the chosen images; keypoints only look, with the reason on one line (#699)', () => {
  assert.equal(fiftyoneReason({ counts: { active: 3 } }), null);
  assert.equal(fiftyoneReason({ counts: { active: 0 } }), 'open.fo_no_images');
  const page = { projectId: 4, ctx: { project: { task: 'detection', counts: { active: 3 } }, links: [], openWhenDone: new Map() } };
  let edited = 0;
  const row = fiftyoneOpenRow(page, { edit: () => { edited += 1; }, count: 3 });
  assert.equal(row.querySelectorAll('button').length, 1, 'one button, not Open and Send');
  assert.match(row.querySelector('#fo-open').textContent, /Edit in FiftyOne/);
  assert.equal(row.querySelector('#fo-open').disabled, false);
  assert.match(row.textContent, /3 images open in FiftyOne\./);
  row.querySelector('#fo-open').click();
  assert.equal(edited, 1);
  assert.equal(fiftyoneOpenRow(page, { edit: () => {}, count: 0 }).querySelector('#fo-open').disabled, true);
  page.ctx.project = { task: 'keypoints', counts: { active: 3 } };
  const view = fiftyoneOpenRow(page, { edit: () => { edited += 1; } });
  assert.match(view.querySelector('#fo-open').textContent, /view only/);
  assert.equal(view.querySelector('#fo-open').disabled, false);
  assert.match(view.textContent, /Keypoints can only be viewed here\./);
  page.ctx.project = { task: 'regression', counts: { active: 3 } };   // #619: values are edited in ActiveLabel
  const values = fiftyoneOpenRow(page, { edit: () => { edited += 1; } });
  assert.match(values.querySelector('#fo-open').textContent, /view only/);
  assert.match(values.textContent, /Edit values in ActiveLabel/);
  values.querySelector('#fo-open').click();
  assert.equal(edited, 1, 'the click opens to look; it does not send the images for editing');
  page.ctx.project = { task: 'detection', counts: { active: 0 } };
  const empty = fiftyoneOpenRow(page);
  assert.equal(empty.querySelector('#fo-open').disabled, true);
  assert.match(empty.textContent, /Add images first\./);
});

test('the header has no Open buttons: the CVAT and FiftyOne tabs open the tools (#602)', () => {
  assert.equal(open.headerCvatButton, undefined);
  assert.equal(open.headerFiftyoneButton, undefined);
  const row = cvatOpenRow({ link: LINK, configured: true });
  assert.equal(row.querySelector('a#cvat-open').getAttribute('href'), LINK.url);
  assert.match(row.querySelector('a#cvat-open').textContent, /Open the project in CVAT/);
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

test('the CVAT tab of a deleted CVAT project says why instead of a dead link (#602, #684)', () => {
  const row = cvatOpenRow({ link: LINK, configured: true, gone: true });
  assert.equal(row.querySelector('a'), null);
  assert.match(row.querySelector('#cvat-open-reason').textContent, /The CVAT project was deleted\./);
  const task = { id: 1, name: 'T1', cvat: 'ready', remote: { url: 'http://cvat.test/tasks/9', jobs: [] } };
  assert.equal(cvatOpenRow({ link: LINK, configured: true, task, gone: true }).querySelector('a#cvat-open').getAttribute('href'), 'http://cvat.test/tasks/9');
});
