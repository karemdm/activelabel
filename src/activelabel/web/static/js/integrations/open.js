/* Opening the project in CVAT and in FiftyOne to look at and edit the annotations there (#602): what can be
 * opened, or why not, and the buttons of the project header and of the Annotations tabs. The edits come back
 * through the CVAT comparison and the FiftyOne edit sessions, as before. */
import { isHttp, toolName } from './tools.js';
import { api } from '../shell/api.js';
import { fiftyoneButton, fiftyoneError, fiftyoneMissing, fiftyoneMissingNote, sayFiftyoneMissing } from '../shell/features.js';
import { el, icon } from '../shell/dom.js';
import { t } from '../shell/i18n.js';
import { toast, toastError } from '../shell/toast.js';

/* why a task cannot be opened in CVAT, by the `cvat` state the task list gives */
const TASK_REASONS = {
  not_configured: 'tasks.cvat_not_configured',
  not_linked: 'tasks.cvat_not_linked',
  unreachable: 'tasks.cvat_unreachable',
  not_synchronised: 'tasks.cvat_not_synced',
  removed_remote: 'tasks.cvat_removed_remote',
  remote_gone: 'tasks.cvat_remote_gone',
};

/* the reasons the CVAT tab can do something about: a comparison creates the project or the task there, or
   decides what to do with what was deleted there (#684); the others (not set up, CVAT down) cannot */
export const CVAT_FIXABLE = new Set(['not_linked', 'not_synchronised', 'removed_remote', 'remote_gone']);

/** Whether the last comparison with CVAT found the linked CVAT project deleted (`page.isRemoteGone` tells a job). */
export function cvatProjectGone(page) {
  const plans = (page.ctx.jobs || []).filter((j) => j.kind === 'sync_plan' && (j.status === 'done' || j.status === 'failed'));
  const last = plans.sort((a, b) => String(b.created).localeCompare(String(a.created)))[0];
  return !!last && last.status === 'failed' && !!page.isRemoteGone && page.isRemoteGone(last);
}

/** What "Open in CVAT" opens: the linked project, or one of its tasks or jobs.
 *  `link` is the project's CVAT link (or null), `configured` whether the CVAT connection is set up, `task` a
 *  task of the task list (with its `cvat` state and `remote` tasks and jobs) and `job` the number of one of its
 *  jobs. Returns `{ url, label }`, or `{ reason }` (the key of the text that says why it cannot be opened). */
export function cvatTarget({ link = null, configured = false, task = null, job = null } = {}) {
  if (!link) return { reason: configured ? 'open.cvat_not_synced' : 'open.cvat_not_configured' };
  if (!task) {
    return isHttp(link.url) ? { url: link.url, label: 'open.cvat_project' } : { reason: 'open.cvat_no_address' };
  }
  const remote = task.remote || {};
  const remoteJob = job != null ? (remote.jobs || []).find((x) => x.n === job) : null;
  const url = job != null ? remoteJob && remoteJob.url : remote.url;
  if (task.cvat === 'ready' && isHttp(url)) return { url, label: job != null ? 'open.cvat_job' : 'open.cvat_task' };
  return { reason: TASK_REASONS[task.cvat] || 'tasks.cvat_not_synced' };
}

/** Why the project cannot be opened in FiftyOne (`null` when it can): FiftyOne must be installed (#658) and the
 *  project needs images. */
export const fiftyoneReason = (project) => {
  if (fiftyoneMissing()) return 'open.fo_not_installed';
  return (project && project.counts && project.counts.active) ? null : 'open.fo_no_images';
};

/** Prepares the project's FiftyOne dataset (its labels and label schema) and opens the App in a new tab when
 *  the job ends. `page` is the project page (projectId, ctx, jobStarted). */
export async function openInFiftyOne(page) {
  try {
    const job = await api('POST', `/projects/${page.projectId}/fiftyone`, { json: {} });
    if (job && job.id) page.ctx.openWhenDone.add(job.id);
    toast(t('cur.fo_starting'), 'info', 4000);
    await page.jobStarted(job);
  } catch (err) { if (!fiftyoneError(err)) toastError(err); }
}

/** A reason as a sentence of a paragraph: the task-list texts (also used as short titles) end without a stop. */
export const sentence = (text) => (/[.!?…]$/.test(text) ? text : `${text}.`);

/** A link that opens `url` in a new tab, as a button. */
const openLink = (id, url, tool, text) => el('a', { class: 'btn', id, href: url, target: '_blank', rel: 'noopener' },
  icon(tool, 18), text, ' ', icon('external', 14));

/** The open row of the CVAT tab, the one place (with the menu of each task) that opens CVAT (#602): the project,
 *  task or job chosen below opens in CVAT, or a clear line says why not; then one line on how the edits come back.
 *  Arguments as in cvatTarget; `gone` when the last comparison found the linked CVAT project deleted (#684). */
export function cvatOpenRow({ link, configured, task, job, gone = false }) {
  const target = gone && !task ? { reason: 'open.cvat_remote_gone' } : cvatTarget({ link, configured, task, job });
  /* the way back is said only when there is something to open; a reason stands alone, as a sentence (#602) */
  if (!target.url) return el('div', { class: 'open-row' }, el('p', { class: 'open-reason', id: 'cvat-open-reason' }, icon('info', 16), sentence(t(target.reason))));
  return el('div', { class: 'open-row' },
    openLink('cvat-open', target.url, 'cvat', t(target.label, { task: task ? task.name : '', n: job })),
    el('p', { class: 'muted small' }, t('open.cvat_back')));
}

/** The open row of the FiftyOne tab, the one place that opens the project's dataset in FiftyOne (#602): open it (to
 *  look at and edit), and how the edits come back; or why not, with what to do. */
export function fiftyoneOpenRow(page) {
  const reason = fiftyoneReason(page.ctx.project);
  const keypoints = page.ctx.project && page.ctx.project.task === 'keypoints';
  return el('div', { class: 'open-row' },
    /* without FiftyOne: dimmed, not disabled, so a tap says how to install it (#675); without images: disabled */
    fiftyoneButton(el('button', { type: 'button', class: `btn${reason === 'open.fo_not_installed' ? ' is-unavailable' : ''}`, id: 'fo-open', disabled: !!reason && reason !== 'open.fo_not_installed',
      'aria-disabled': reason === 'open.fo_not_installed' ? 'true' : null, 'aria-describedby': 'fo-open-help', onclick: () => (sayFiftyoneMissing() ? null : openInFiftyOne(page)) },
      icon('fiftyone', 18), t('annotate.open_in', { tool: toolName('fiftyone') }))),
    reason === 'open.fo_not_installed'
      ? el('div', { id: 'fo-open-help' }, fiftyoneMissingNote())
      : el('p', { class: 'muted small', id: 'fo-open-help' }, t(reason || (keypoints ? 'tasks.fo_no_keypoints' : 'open.fo_back'))));
}
