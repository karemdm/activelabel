/* The queue of heavy work, from any screen (#603): a button in the top bar (and the shortcut Alt+Q) opens a
 * small panel with the trainings, pre-annotations, curation analyses and model exports that run or wait, with
 * their place in the queue, progress and Cancel. It reads the global queue (GET /api/queue, the scheduler of
 * #604) and, while the server has no such route, the activity lists of the projects. */
import { jobMessage } from '../integrations/tools.js';
import { api } from '../shell/api.js';
import { confirmDialog } from '../shell/dialogs.js';
import { el, fill, icon } from '../shell/dom.js';
import { num, t, tn, tr } from '../shell/i18n.js';
import { toastError } from '../shell/toast.js';
import { progressBar } from '../shell/widgets.js';

/** The kinds of work the queue shows (one at a time per device; the light jobs are not in it). */
export const HEAVY_KINDS = ['training', 'preannotate', 'analyze', 'export'];

/* the step of the project page where a kind of work is followed */
const STEP_OF = { training: 'training', preannotate: 'annotations', analyze: 'curation', export: 'models' };

const isActive = (status) => status === 'queued' || status === 'running';

/** How a job of the activity lists is cancelled: the runs, pre-annotations and analyses have their own route;
 *  an export cannot be stopped (null). */
export function cancelPath(job) {
  const n = job.params && job.params.number;
  if (job.kind === 'training' && n != null) return `/projects/${job.project_id}/runs/${n}/cancel`;
  if (job.kind === 'preannotate' && n != null) return `/projects/${job.project_id}/preannotations/${n}/cancel`;
  if (job.kind === 'analyze') return `/jobs/${encodeURIComponent(job.id)}/cancel`;
  return null;
}

/** The rows of the panel from the scheduler's snapshot (`{items}`): the heavy work, running first, then waiting
 *  in the order of its place. Returns `[{ id, project_id, project_name, kind, status, position, done, total,
 *  message, cancel }]`, `cancel` being the route to POST to. */
export function rowsFromQueue(snapshot) {
  const items = ((snapshot && snapshot.items) || []).filter((i) => HEAVY_KINDS.includes(i.kind) && isActive(i.status));
  return items
    .map((i) => ({ ...pick(i), project_name: i.project_name || '', position: i.position ?? null, cancel: `/queue/${encodeURIComponent(i.id)}/cancel` }))
    .sort(order);
}

/** The rows of the panel from the activity lists of the projects (before the global queue exists): `projects` is
 *  the project list and `jobsOf` maps a project id to its jobs. The waiting jobs are numbered in the order they
 *  were queued (the server runs the heavy work one at a time). Returns rows as rowsFromQueue. */
export function rowsFromJobs(projects, jobsOf) {
  const names = Object.fromEntries((projects || []).map((p) => [p.id, p.name]));
  const rows = [];
  for (const p of projects || []) {
    for (const j of jobsOf[p.id] || []) {
      if (!HEAVY_KINDS.includes(j.kind) || !isActive(j.status)) continue;
      rows.push({ ...pick({ ...j, project_id: j.project_id ?? p.id }), project_name: names[p.id] || '', position: null, cancel: cancelPath({ ...j, project_id: j.project_id ?? p.id }) });
    }
  }
  rows.filter((r) => r.status === 'queued').sort((a, b) => String(a.created).localeCompare(String(b.created)))
    .forEach((r, i) => { r.position = i + 1; });
  return rows.sort(order);
}

/* the fields the panel shows */
function pick(j) {
  return { id: j.id, project_id: j.project_id, kind: j.kind, status: j.status, done: j.done || 0, total: j.total || 0, message: j.message || '', created: j.created || '' };
}
/* running first, then by place in the queue */
function order(a, b) {
  if (a.status !== b.status) return a.status === 'running' ? -1 : 1;
  return (a.position ?? 0) - (b.position ?? 0);
}

/** Reads the heavy work of every project: the global queue when the server has it, else the activity lists. */
export async function loadQueue() {
  try {
    return rowsFromQueue(await api('GET', '/queue'));
  } catch (err) {
    if (err.status !== 404 && err.status !== 405) throw err;
  }
  const projects = (await api('GET', '/projects')) || [];
  const lists = await Promise.all(projects.map((p) => api('GET', `/projects/${p.id}/jobs`).catch(() => [])));
  return rowsFromJobs(projects, Object.fromEntries(projects.map((p, i) => [p.id, Array.isArray(lists[i]) ? lists[i] : []])));
}

/** One line of the panel: what, of which project (a link to its step), where it is, progress and Cancel. */
export function queueRow(row, onCancel) {
  const name = tr(`job.kind.${row.kind}`, row.kind);
  const where = row.status === 'running' ? t('queue.running') : row.position ? t('queue.position', { n: num(row.position) }) : t('queue.waiting');
  const href = row.project_id ? `/projects/${row.project_id}#${STEP_OF[row.kind] || 'images'}` : null;
  return el('li', { class: `queue-row st-${row.status}`, 'data-job': row.id },
    el('div', { class: 'queue-row-top' },
      el('strong', null, name),
      href ? el('a', { href, class: 'queue-project' }, row.project_name || `#${row.project_id}`) : null,
      el('span', { class: `status status-${row.status}` }, where),
      row.cancel
        ? el('button', { type: 'button', class: 'btn small danger-outline', 'data-act': 'cancel', onclick: () => onCancel(row) }, t('queue.cancel'))
        : el('span', { class: 'muted small' }, t('queue.no_cancel'))),
    row.status === 'running'
      ? el('div', { class: 'queue-progress' }, progressBar(row.done, row.total, name), row.total ? el('span', { class: 'muted small' }, `${num(row.done)} / ${num(row.total)}`) : null)
      : null,
    row.status === 'running' && row.message ? el('p', { class: 'muted small queue-msg' }, jobMessage(row)) : null);
}

/** Wires the top bar button (#queue-toggle) and the Alt+Q shortcut to the queue panel; the button shows how
 *  many heavy jobs run or wait. Does nothing on a page without the button. */
export function initQueue() {
  const button = document.getElementById('queue-toggle');
  if (!button) return;
  const count = el('span', { class: 'queue-count', hidden: true });
  button.appendChild(count);
  const list = el('div', { class: 'queue-list', 'aria-live': 'polite' });
  const panel = el('div', { class: 'queue-panel card', id: 'queue-panel', role: 'dialog', 'aria-labelledby': 'queue-title', hidden: true },
    el('div', { class: 'queue-head' },
      el('h2', { id: 'queue-title' }, t('queue.title')),
      el('button', { type: 'button', class: 'icon-btn', 'aria-label': t('common.close'), title: t('common.close'), onclick: () => close(true) }, icon('close', 18))),
    el('p', { class: 'muted small' }, t('queue.help')),
    list);
  button.setAttribute('aria-controls', 'queue-panel');
  button.after(panel);
  let rows = [];
  let timer = null;

  function draw() {
    const n = rows.length;
    count.hidden = !n;
    count.textContent = num(n);
    button.setAttribute('aria-label', n ? tn('queue.button_n', n) : t('queue.title'));
    if (panel.hidden) return;
    fill(list, n
      ? el('ul', { class: 'plain queue-rows' }, ...rows.map((r) => queueRow(r, cancel)))
      : el('p', { class: 'muted queue-empty' }, icon('check', 16), t('queue.empty')));
  }
  async function refresh() {
    clearTimeout(timer);
    if (!document.hidden) {
      try { rows = await loadQueue(); } catch (err) { if (!panel.hidden) toastError(err); }
      draw();
    }
    timer = setTimeout(refresh, panel.hidden ? 15000 : 3000);
  }
  async function cancel(row) {
    const ok = await confirmDialog({ title: t('queue.cancel_title'), text: t('queue.cancel_text', { what: tr(`job.kind.${row.kind}`, row.kind), project: row.project_name }), confirmLabel: t('queue.cancel') });
    if (!ok) return;
    try { await api('POST', row.cancel); } catch (err) { toastError(err); }
    refresh();
  }
  function open() {
    panel.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    draw();
    refresh();
    panel.querySelector('.icon-btn').focus();
  }
  function close(focusButton) {
    if (panel.hidden) return;
    panel.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    if (focusButton) button.focus();
  }
  const toggle = () => (panel.hidden ? open() : close(true));
  button.addEventListener('click', toggle);
  document.addEventListener('keydown', (e) => {
    if (e.altKey && !e.ctrlKey && !e.metaKey && (e.key === 'q' || e.key === 'Q')) { e.preventDefault(); toggle(); }
    else if (e.key === 'Escape' && !panel.hidden && !document.querySelector('dialog[open]')) close(true);
  });
  document.addEventListener('click', (e) => {
    if (!panel.hidden && !panel.contains(e.target) && !button.contains(e.target) && !document.querySelector('dialog[open]')) close(false);
  });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  refresh();
}
