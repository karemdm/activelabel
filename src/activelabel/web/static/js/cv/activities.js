/* The activity strip of a project (running and finished jobs), its history, and the job helpers the steps
 * share (is it running, did it go stale, start one and wait for the strip to show it). */
import { imagesOf, jobMessage, statusText, toolName } from '../integrations/tools.js';
import { api, jobFailure } from '../shell/api.js';
import { openModal } from '../shell/dialogs.js';
import { el, fill, icon } from '../shell/dom.js';
import { fmtDateTime, num, t, tn, tr } from '../shell/i18n.js';
import { toast, toastError } from '../shell/toast.js';
import { progressBar } from '../shell/widgets.js';

/** Builds the activity strip of the project page (`page.jobsBox`) and returns the job helpers the steps share.
 *  `page` is the object of the project page (cv/project.js will build it; today initProject in main.js). */
export function createActivities(page) {
  const { ctx, jobsBox, planIsEmpty, projectId, refreshProject, showMissing, showTab } = page;
  /* the comparison no longer matches the project or the tool: compare again */
  const STALE_CODES = new Set(['plan_not_found', 'sync_remote_changed', 'sync_local_changed', 'sync_link_changed', 'sync_task_has_new_images']);
  const isStale = (job) => job.kind === 'sync_apply' && STALE_CODES.has(job.error_code);
  /* the linked project was deleted in the tool: say so plainly and how to go on */
  const isRemoteGone = (job) => /^sync_/.test(job.kind) && job.error_code === 'remote_gone';
  /* a failed job's text: by its code in the interface language, else the server's own words */
  const jobError = (job) => (isStale(job) ? t('sync.stale') : jobFailure(job));
  const syncJobTool = (job) => {
    const r = job.result || {};
    const x = ctx.syncTool[job.id] || job.tool || r.tool || (job.params && job.params.tool) || (job.meta && job.meta.tool);
    return typeof x === 'string' ? x : '';
  };
  const kindText = (kind, job) => {
    if (kind === 'sync_plan') {
      const tool = job ? syncJobTool(job) : '';
      return tool ? t('job.kind.sync_plan', { tool: toolName(tool) }) : t('job.kind.sync_plan_generic');
    }
    return tr(`job.kind.${kind}`, kind);
  };
  const isActive = (job) => job.status === 'queued' || job.status === 'running';

  /* Runs, exports and pre-annotations stop through their own endpoint (it ends the process and marks the
     run), which needs the run number: the job does not carry it, so these are cancelled in their step. */
  const RUN_KINDS = new Set(['training', 'export', 'preannotate']);

  function jobStep(job) {
    const kind = String(job.kind || '');
    if (/^sync_/.test(kind)) return 'annotations';
    if (kind === 'training') return 'training';
    if (kind === 'export') return 'models';
    if (kind === 'preannotate') return 'annotations';
    if (kind === 'video_task') return 'images';
    if (/fiftyone/.test(kind)) return 'fiftyone';
    if (/analy|curat/.test(kind)) return 'curation';
    if (/annot/.test(kind)) return 'annotations';
    return 'images';
  }

  const jobSessionId = (job) => {
    const s = job.result && job.result.session;
    return s && typeof s === 'object' ? s.id : s;
  };

  /* one job row (used by the bar and by the history dialog) */
  function jobRow(job) {
    const active = isActive(job);
    const step = jobStep(job);
    const url = job.result && job.result.url;
    return el('li', { class: `job job-${job.status}`, 'data-job': job.id },
      el('div', { class: 'job-top' },
        el('strong', null, kindText(job.kind, job)),
        el('span', { class: `status status-${job.status}` }, statusText(job.status)),
        el('span', { class: 'muted small' }, fmtDateTime(job.finished || job.started || job.created)),
        active && !RUN_KINDS.has(job.kind) ? el('button', { type: 'button', class: 'btn small', onclick: () => cancelJob(job) }, t('jobs.stop')) : null,
        !active && job.status === 'done' && step === 'fiftyone' && url
          ? el('a', { class: 'btn small', href: url, target: '_blank', rel: 'noopener' }, t('jobs.open'))
          : null,
        !active && job.status === 'done' && step !== 'fiftyone'
          ? el('button', { type: 'button', class: 'btn small', onclick: () => { closeHistory(); showTab(step, true); if (/^annotate_/.test(job.kind)) setTimeout(() => ctx.panels.annotations.reveal(jobSessionId(job)), 80); else if (/^sync_/.test(job.kind)) setTimeout(() => ctx.panels.annotations.reveal('sync'), 80); } }, t('jobs.see_result'))
          : null),
      active ? el('div', { class: 'job-progress' }, progressBar(job.done, job.total, kindText(job.kind, job)),
        el('span', { class: 'muted' }, job.total ? `${num(job.done)} / ${num(job.total)}` : '')) : null,
      active && job.message ? el('div', { class: 'muted job-msg' }, jobMessage(job)) : null,
      !active && job.status === 'done' && resultText(job) ? el('div', { class: 'muted job-msg' }, resultText(job)) : null,
      job.error ? el('div', { class: 'job-error' }, `${t('common.error')} ${jobError(job)}`) : null);
  }

  /* A small pill in the corner (#262): what is running, or the latest finished activity. A failure stays
     on the pill (red, with its icon and word) until the person opens it; a success fades after a few
     seconds. The pill opens the card with the details and the history; finished activities that were
     closed stay closed (remembered per project in this browser). A hidden live line tells screen
     readers when the activity changes (not at every percent). */
  const jobsKey = `al.jobs.${projectId}`;
  const jobsPrefs = (() => {
    try { const v = JSON.parse(localStorage.getItem(jobsKey) || '{}'); return { min: true, closed: new Set(v.closed || []) }; } catch (err) { return { min: true, closed: new Set() }; }
  })();
  function saveJobsPrefs() {
    try { localStorage.setItem(jobsKey, JSON.stringify({ min: jobsPrefs.min, closed: [...jobsPrefs.closed].slice(-200) })); } catch (err) { /* not remembered */ }
  }
  let jobsFocus = null;
  let jobsFade = null;
  const jobsSeen = new Set();  // failures the person opened: from then on they fade like successes
  const jobsLive = el('p', { class: 'sr-only', role: 'status' });
  document.body.appendChild(jobsLive);
  function renderJobs() {
    if (ctx.syncDraw) ctx.syncDraw();
    ctx.watchers.forEach((watch) => watch());
    const all = [...ctx.jobs].sort((a, b) => String(b.created).localeCompare(String(a.created)));
    const running = all.filter(isActive);
    const finished = all.filter((j) => !isActive(j));
    const latest = finished.slice(0, 1).filter((j) => !jobsPrefs.closed.has(j.id));
    const shown = [...running, ...latest];
    jobsBox.hidden = !shown.length;
    jobsBox.classList.toggle('min', jobsPrefs.min);
    clearTimeout(jobsFade);
    if (!shown.length) { fill(jobsBox); jobsLive.textContent = ''; return; }
    const pct = running.length === 1 && running[0].total ? Math.round((running[0].done / running[0].total) * 100) : null;
    const summary = running.length
      ? `${tn('jobs.running_n', running.length)}${pct != null ? ` · ${pct}%` : ''}`
      : `${statusText(latest[0].status)} · ${kindText(latest[0].kind, latest[0])}`;
    const glyph = running.length ? el('span', { class: 'tr-dot', 'aria-hidden': 'true' }) : icon(latest[0].status === 'done' ? 'check' : 'error', 14);
    const said = `${t('jobs.title')}: ${running.length ? tn('jobs.running_n', running.length) : summary}`;
    if (jobsLive.textContent !== said) jobsLive.textContent = said;
    const state = running.length ? 'running' : latest[0].status;
    const fades = !running.length && (state !== 'failed' || jobsSeen.has(latest[0].id));
    if (fades && jobsPrefs.min) {
      jobsFade = setTimeout(() => { jobsPrefs.closed.add(latest[0].id); saveJobsPrefs(); renderJobs(); }, 6000);
    }
    const toggle = (min) => {
      if (!min) finished.forEach((j) => jobsSeen.add(j.id));  // opened: the failure was seen
      jobsPrefs.min = min; saveJobsPrefs(); jobsFocus = min ? 'open' : 'min'; renderJobs();
    };
    const close = () => {
      finished.forEach((j) => jobsPrefs.closed.add(j.id));
      if (running.length) jobsPrefs.min = true;
      saveJobsPrefs();
      jobsFocus = running.length ? 'open' : null;
      renderJobs();
    };
    if (jobsPrefs.min) {
      const pill = el('button', { type: 'button', class: `jobs-pill is-${state}${fades ? ' fades' : ''}`, 'data-jobs': 'open', 'aria-expanded': 'false', onclick: () => toggle(false) },
        glyph, el('span', null, summary), el('span', { class: 'sr-only' }, t('jobs.show')));
      /* a failure that stays (a comparison that failed, for instance) can be dismissed without opening it */
      const dismiss = state === 'failed' && !fades
        ? el('button', { type: 'button', class: 'icon-btn small jobs-dismiss', 'data-jobs': 'dismiss', 'aria-label': `${t('jobs.dismiss')}: ${summary}`, title: t('jobs.dismiss'), onclick: close }, icon('close', 14))
        : null;
      fill(jobsBox, el('div', { class: 'jobs-pill-row' }, pill, dismiss));
    } else {
      const head = el('div', { class: 'jobs-head' },
        el('h2', null, t('jobs.title')),
        el('span', { class: 'muted jobs-summary' }, summary),
        finished.length > 1 ? el('button', { type: 'button', class: 'btn small', onclick: openHistory }, t('jobs.history', { n: num(finished.length) })) : null,
        el('button', { type: 'button', class: 'icon-btn', 'data-jobs': 'min', 'aria-expanded': 'true', 'aria-label': t('jobs.minimize'), title: t('jobs.minimize'), onclick: () => toggle(true) }, icon('chevron', 16)),
        el('button', { type: 'button', class: 'icon-btn', 'data-jobs': 'close', 'aria-label': t('jobs.close'), title: t('jobs.close'), onclick: close }, icon('close', 16)));
      fill(jobsBox, head, el('ul', { class: 'jobs' }, ...shown.map(jobRow)));
    }
    if (jobsFocus) { const b = jobsBox.querySelector(`[data-jobs="${jobsFocus}"]`); if (b) b.focus(); jobsFocus = null; }
    if (historyDlg && historyDlg.open) fillHistory();
  }

  let historyDlg = null;
  function fillHistory() {
    const finished = [...ctx.jobs].filter((j) => !isActive(j))
      .sort((a, b) => String(b.created).localeCompare(String(a.created)));
    const clear = el('button', { type: 'button', class: 'btn danger-outline', onclick: async () => {
      try {
        await api('DELETE', `/projects/${projectId}/jobs`);
        await pollJobs();
        closeHistory();
      } catch (err) { toastError(err); }
    } }, t('jobs.clear'));
    fill(historyDlg, el('div', { class: 'dialog-body' },
      el('h2', null, t('jobs.history_title')),
      finished.length ? el('ul', { class: 'jobs scroll-box' }, ...finished.map(jobRow)) : el('p', { class: 'muted' }, t('jobs.history_empty')),
      el('div', { class: 'form-actions' },
        finished.length ? clear : null,
        el('button', { type: 'button', class: 'btn', onclick: closeHistory }, t('common.close')))));
  }
  function openHistory() {
    if (!historyDlg) { historyDlg = el('dialog', { class: 'dialog wide', 'aria-label': t('jobs.history_title') }); document.body.appendChild(historyDlg); }
    fillHistory();
    openModal(historyDlg);
  }
  function closeHistory() { if (historyDlg && historyDlg.open) historyDlg.close(); }

  /* one line saying what a finished job changed */
  function resultText(job) {
    const r = job.result || {};
    if (job.kind === 'analyze' && 'measured' in r) {
      const flagged = Object.values(r.issues || {}).reduce((a, n) => a + (Number(n) || 0), 0);
      return t(r.measured ? 'jobs.result_analyze' : 'jobs.result_analyze_none', { measured: num(r.measured), flagged: num(flagged) });
    }
    if (job.kind === 'import_folder' && 'added' in r) return tn('images.added', r.added);
    if (job.kind === 'annotate_send' && r.session && typeof r.session === 'object') {
      return tn('jobs.result_annotate_send', imagesOf(r.session), { tool: toolName(r.session.tool) });
    }
    if (job.kind === 'sync_plan' && 'plan_id' in r) {
      /* every change counts: images, annotations and tasks, each way */
      const len = (x) => (Array.isArray(x) ? x.length : Number(x) || 0);
      const incoming = len(r.pull) + len(r.remove_local) + len(r.restore_local) + len(r.tasks_new);
      const sent = 'images_new' in r ? len(r.images_new) : len(r.add) + len(r.tasks_create);  /* images sent for the first time, in new tasks or added */
      const outgoing = len(r.push) + sent + len(r.remove_remote) + len(r.restore_remote) + len(r.tasks_delete);
      const conflicts = len(r.conflicts);
      if (!incoming && !outgoing && !conflicts) return t('jobs.result_sync_plan_none');
      return t('jobs.result_sync_plan', { pull: num(incoming), push: num(outgoing), conflicts: num(conflicts) });
    }
    if (job.kind === 'sync_apply') {
      const n = (...keys) => Number(keys.map((k) => r[k]).find((v) => v != null)) || 0;
      const incoming = n('pulled') + n('removed_local') + n('restored_local') + n('tasks_imported');
      const outgoing = n('pushed') + n('images_new') + n('removed_remote') + n('restored_remote') + n('tasks_deleted');
      const base = t('jobs.result_sync_apply', { pull: num(incoming), push: num(outgoing) });
      const left = n('conflicts_left');
      return left ? `${base} ${tn('jobs.result_sync_left', left)}` : base;
    }
    if (job.kind === 'annotate_pull' && r.diff) return tn('jobs.result_annotate_pull', r.diff.images_changed || 0);
    return '';
  }

  async function cancelJob(job) {
    try {
      await api('POST', `/jobs/${job.id}/cancel`);
      await pollJobs();
    } catch (err) { toastError(err); }
  }

  async function pollJobs() {
    clearTimeout(ctx.pollTimer);
    ctx.pollTimer = null;
    let data;
    try {
      data = await api('GET', `/projects/${projectId}/jobs`);
    } catch (err) {
      if (err.status === 404) { showMissing(); return; }  // the project was deleted (maybe in another tab)
      if (!ctx.pollFailing) toastError(err);
      ctx.pollFailing = true;
      ctx.pollTimer = setTimeout(pollJobs, 5000);
      return;
    }
    if (ctx.pollFailing) { ctx.pollFailing = false; toast(t('jobs.back_online'), 'info', 3000); }
    const jobs = Array.isArray(data) ? data : (data && data.jobs) || [];
    ctx.jobs = jobs;
    if (!ctx.jobsSeen) {  // a sync finished before this page was opened is old news: its result card is not shown again
      ctx.jobsSeen = true;
      jobs.filter((j) => j.kind === 'sync_apply' && j.status === 'done').forEach((j) => ctx.syncDismissed.add(j.id));
    }
    let finished = false;
    for (const job of jobs) {
      const before = ctx.jobStatus.get(job.id);
      ctx.jobStatus.set(job.id, job.status);
      if (before && (before === 'queued' || before === 'running') && !isActive(job)) {
        finished = true;
        if (job.status === 'failed') toastError(new Error(jobError(job) || statusText('failed')));
        if (job.status === 'done' && ctx.openWhenDone.has(job.id)) {
          ctx.openWhenDone.delete(job.id);
          const url = job.result && job.result.url;
          if (url && !window.open(url, '_blank', 'noopener')) toast(t('cur.fo_popup'), 'info', 8000);
        }
      }
    }
    renderJobs();
    for (const job of jobs) {
      if (job.status === 'done' && ctx.syncNotify && ctx.syncNotify.has(job.id)) {
        ctx.syncNotify.delete(job.id);
        if (job.kind === 'sync_plan' && ctx.tab === 'annotations') continue;  // the plan is already on screen
        if (job.kind === 'sync_apply') toast(`${t('sync.applied')} ${resultText(job)}`, 'success', 8000);
        else if (!planIsEmpty(job.result || {})) toast(t('sync.plan_ready'), 'info', 8000);
      }
    }
    if (finished) {
      await refreshProject();
      Object.values(ctx.panels).forEach((panel) => panel.mounted && panel.refresh && panel.refresh());
    }
    if (jobs.some(isActive)) ctx.pollTimer = setTimeout(pollJobs, 2000);
  }

  async function jobStarted(job) {
    if (job && job.id) {
      ctx.jobStatus.set(job.id, job.status || 'queued');
      if (/^sync_/.test(String(job.kind))) (ctx.syncNotify = ctx.syncNotify || new Set()).add(job.id);
    }
    await pollJobs();  // so the buttons that wait for this job are disabled when the caller redraws
  }
  const kindActive = (...kinds) => ctx.jobs.some((j) => kinds.includes(j.kind) && isActive(j));

  return { isActive, isRemoteGone, isStale, jobError, jobStarted, kindActive, pollJobs, renderJobs, syncJobTool };
}
