/* Synchronisation with CVAT on the project page: the links, the comparison (plan) and applying it, and the
 * sync block of the Annotations step. */
import { TOOLS, isHttp, toolName } from './tools.js';
import { api } from '../shell/api.js';
import { confirmDialog } from '../shell/dialogs.js';
import { add, el, fill, icon } from '../shell/dom.js';
import { fmtDateTime, num, t, tn } from '../shell/i18n.js';
import { toast, toastError } from '../shell/toast.js';

/** Builds the CVAT synchronisation of the project page and returns loadLinks, startSync and syncBlock (the
 *  block shown in the Annotations step). `page` is the object of the project page; it also brings diffView and
 *  openLightbox, which live above this layer. */
export function createSync(page) {
  const { ctx, diffView, isActive, isRemoteGone, isStale, jobError, jobStarted, openLightbox, planIsEmpty, projectId, refreshProject, syncJobTool } = page;
  async function loadLinks() {
    try {
      const data = await api('GET', `/projects/${projectId}/links`);
      ctx.links = (data && data.links) || [];
    } catch (err) {
      if (err.status !== 404) toastError(err);
    }
  }

  async function startSync(tool, body = {}) {
    const job = await api('POST', `/projects/${projectId}/sync/${tool}`, { json: body });
    if (job && job.id) ctx.syncTool[job.id] = tool;
    toast(t('sync.started', { tool: toolName(tool) }), 'info', 3000);
    await jobStarted(job);
  }

  function syncBlock() {
    /* the review area of the CVAT section: the plan, its result and (paused tools) older links */
    const node = el('div', { class: 'sync-section', id: 'annotate-sync', tabindex: '-1' });
    const box = el('div', { class: 'sync-box' });
    add(node, box);
    const states = new Map();  // plan_id -> {res, def, error}
    let lastSig = null;

    const planJob = () => {
      const jobs = [...ctx.jobs].sort((a, b) => String(b.created).localeCompare(String(a.created)));
      const job = jobs.find((j) => j.kind === 'sync_plan' && j.status === 'done' && j.result && j.result.plan_id);
      if (!job || ctx.syncDismissed.has(job.result.plan_id)) return null;
      const later = jobs.some((j) => j.kind === 'sync_apply' && j.status !== 'failed' && String(j.created) >= String(job.created));
      return later ? null : job;
    };
    const planningTools = () => ctx.jobs.filter((j) => j.kind === 'sync_plan' && isActive(j)).map(syncJobTool);

    function perClassText(side) {
      const pc = (side && side.per_class) || {};
      const e = Object.entries(pc).filter(([, n]) => Number(n) > 0);
      return e.length ? e.map(([k, n]) => `${k} ${num(n)}`).join(' · ') : t('sync.no_labels');
    }

    function linkCard(link) {
      const tool = link.tool;
      const planning = planningTools().length > 0;
      const open = isHttp(link.url)
        ? el('a', { href: link.url, target: '_blank', rel: 'noopener' }, link.remote_name || toolName(tool), ' ', icon('external', 14))
        : el('span', null, link.remote_name || toolName(tool));
      const run = link.supported === false ? null : el('button', { type: 'button', class: 'btn primary', disabled: planning, title: planning ? t('sync.compare_busy_any') : null, onclick: async (e) => {
        const b = e.currentTarget;
        b.disabled = true;
        try { await startSync(tool); } catch (err) { b.disabled = false; toastError(err); }
      } }, t('sync.run'));
      const removed = link.supported === false;
      const unlink = el('button', { type: 'button', class: 'btn danger-outline', onclick: async () => {
        const ok = await confirmDialog(removed
          ? { title: t('sync.unlink_title_removed'), text: t('sync.unlink_text_removed'), confirmLabel: t('sync.unlink') }
          : { title: t('sync.unlink_title', { tool: toolName(tool) }), text: t('sync.unlink_text', { tool: toolName(tool) }), confirmLabel: t('sync.unlink') });
        if (!ok) return;
        try {
          await api('DELETE', `/projects/${projectId}/links/${tool}`);
          toast(removed ? t('sync.unlinked_removed') : t('sync.unlinked', { tool: toolName(tool) }), 'success', 4000);
          await refreshProject();
          draw(true);
        } catch (err) { toastError(err); }
      } }, t('sync.unlink'));
      return el('li', { class: 'link-card card', 'data-tool': tool },
        el('div', { class: 'session-head' },
          el('span', { class: 'session-tool' }, icon(TOOLS.includes(tool) ? tool : 'info', 22), el('strong', null, toolName(tool)))),
        el('dl', { class: 'facts link-facts' },
          el('div', null, el('dt', null, removed ? t('sync.remote_project_removed') : t('sync.remote_project', { tool: toolName(tool) })), el('dd', null, open)),
          el('div', null, el('dt', null, t('sync.images_linked')), el('dd', null, num(link.images || 0))),
          el('div', null, el('dt', null, t('sync.last')), el('dd', null, link.last_sync ? fmtDateTime(link.last_sync) : t('sync.never')))),
        removed ? el('p', { class: 'muted' }, t('sync.link_unsupported')) : null,
        el('div', { class: 'session-actions' }, run, unlink));
    }

    /* "Whole project", "Task Batch A" or "Task Batch A · Job 2" */
    function scopeText(sc) {
      if (!sc || sc.scope === 'project' || sc.task == null) return t('sync.scope_project');
      const found = (ctx.tasksList || []).find((x) => x.id === sc.task);
      const base = `${t('sync.scope_task')} ${found ? found.name : sc.task}`;
      return sc.scope === 'job' && sc.job != null ? `${base} · ${t('sync.scope_job')} ${sc.job}` : base;
    }

    const RESULT_KEYS = ['pulled', 'pushed', 'images_new', 'labels_new', 'removed_remote', 'removed_local', 'restored_remote', 'restored_local',
      'tasks_created', 'tasks_imported', 'tasks_deleted', 'conflicts_left'];
    const resultJob = () => {
      const last = [...ctx.jobs].sort((a, b) => String(b.created).localeCompare(String(a.created))).find((j) => j.kind === 'sync_apply');
      if (!last || last.status !== 'done' || ctx.syncDismissed.has(last.id)) return null;
      /* a comparison that failed afterwards (the project is gone, CVAT is down...) makes this result old news */
      return ctx.jobs.some((j) => j.kind === 'sync_plan' && j.status === 'failed' && String(j.created) >= String(last.created)) ? null : last;
    };

    function resultCard(job) {
      const r = job.result || {};
      const name = toolName(syncJobTool(job) || 'cvat');
      const rows = RESULT_KEYS.filter((k) => Number(r[k]) > 0).map((k) => el('li', { class: 'dir-row' },
        el('span', null, t(`sync.res_${k}`, { tool: name })), el('strong', null, num(r[k]))));
      return el('div', { class: 'card sync-result', id: 'sync-result' },
        el('div', { class: 'session-head' },
          el('span', { class: 'session-tool' }, icon('check', 22), el('strong', null, t('sync.result_title')))),
        rows.length ? el('ul', { class: 'plain dir-list res-list' }, ...rows) : el('p', { class: 'muted' }, t('sync.in_sync')),
        el('div', { class: 'session-actions' },
          el('button', { type: 'button', class: 'btn', onclick: () => { ctx.syncDismissed.add(job.id); draw(true); } }, t('common.close'))));
    }

    function diffBlock(title, n, diff) {
      if (!n || !diff) return null;
      return el('details', { class: 'sync-diff', open: true }, el('summary', null, title), diffView(projectId, diff));
    }

    function planCard(job) {
      const plan = job.result;
      const tool = plan.tool || syncJobTool(job);
      const name = toolName(tool);
      const remote = plan.remote || {};
      const conflicts = plan.conflicts || [];
      const only = plan.remote_only || {};
      const pid = plan.plan_id;
      if (!states.has(pid)) states.set(pid, { res: {}, def: null, error: null, del: false });
      const st = states.get(pid);
      const tCreate = plan.tasks_create || [];
      const tNew = plan.tasks_new || [];
      const tDelete = plan.tasks_delete || [];
      const nothing = planIsEmpty(plan);
      const radios = [];

      const sub = (items, text) => (items.length
        ? el('ul', { class: 'plain dir-sub' },
            ...items.slice(0, 8).map((x) => el('li', null, text(x))),
            items.length > 8 ? el('li', { class: 'muted' }, t('sync.only_more', { n: num(items.length - 8) })) : null)
        : null);
      const line = (label, n, extra) => (n
        ? el('li', { class: 'dir-line' },
            el('div', { class: 'dir-row' }, el('span', null, label), el('strong', null, num(n))),
            extra || null)
        : null);
      const taskText = (x) => [x.name, tn('sync.n_images', Number(x.images) || 0), x.job_size ? t('sync.per_job', { n: num(x.job_size) }) : null].filter(Boolean).join(' · ');
      const deleteBox = el('label', { class: 'dir-check' },
        el('input', { type: 'checkbox', id: 'sync-delete-tasks', onchange: (e) => { st.del = e.target.checked; } }),
        el('span', null, t('sync.delete_in_tool', { tool: name })),
        el('small', { class: 'muted' }, t('sync.delete_in_tool_help', { tool: name })));
      deleteBox.querySelector('input').checked = !!st.del;
      const dir = (title, lines) => {
        const shown = lines.filter(Boolean);
        return el('section', { class: 'sync-dir' },
          el('h4', null, title),
          shown.length ? el('ul', { class: 'plain dir-list' }, ...shown) : el('p', { class: 'muted' }, t('sync.dir_empty')));
      };
      const dirs = el('div', { class: 'sync-dirs' },
        dir(t('sync.push', { tool: name }), [
          line(t('sync.row_tasks_create'), tCreate.length, sub(tCreate, taskText)),
          line(t('sync.row_tasks_delete'), tDelete.length, el('div', null, sub(tDelete, (x) => x.name || `#${x.task}`), deleteBox)),
          line(t('sync.row_add'), plan.add),
          line(t('sync.row_new_annot'), plan.labels_new, el('p', { class: 'muted dir-sub' }, tn('sync.n_images', Number(plan.images_new) || 0))),
          line(t('sync.row_annot'), plan.push),
          line(t('sync.row_remove'), plan.remove_remote),
          line(t('sync.row_restore'), plan.restore_remote)]),
        dir(t('sync.pull', { tool: name }), [
          line(t('sync.row_tasks_new'), tNew.length, sub(tNew, taskText)),
          line(t('sync.row_annot'), plan.pull),
          line(t('sync.row_remove'), plan.remove_local),
          line(t('sync.row_restore'), plan.restore_local)]));

      const apply = el('button', { type: 'button', class: 'btn primary big', id: 'sync-apply' }, t('sync.apply'));
      const hint = el('p', { class: 'muted sync-hint', 'aria-live': 'polite' });
      const errorBox = el('div', { class: 'sync-error', role: 'alert' });
      const missing = () => conflicts.filter((c) => !st.res[c.image_id] && !st.def).length;
      const update = () => {
        const m = missing();
        apply.disabled = m > 0 || apply.dataset.busy === '1';
        hint.textContent = m > 0 ? tn('sync.choose_left', m) : '';
        for (const r of radios) {
          const v = st.res[r.id] || st.def;
          r.local.checked = v === 'local';
          r.remote.checked = v === 'remote';
        }
        fill(errorBox, ...(st.error === 'stale'
          ? [el('p', { class: 'report-line' }, icon('warning'), t('sync.stale')),
             el('button', { type: 'button', class: 'btn', onclick: async () => {
               ctx.syncDismissed.add(pid);
               const sc = plan.scope || {};
               const again = sc.scope && sc.scope !== 'project' ? { scope: sc.scope, task: sc.task, job: sc.job } : {};
               if (plan.direction) again.direction = plan.direction;
               try { await startSync(tool, again); } catch (err) { toastError(err); }
               draw(true);
             } }, t('sync.again'))]
          : []));
      };
      const setAll = (v) => { st.def = v; for (const c of conflicts) st.res[c.image_id] = v; update(); };

      const items = conflicts.map((c) => ({ id: c.image_id, name: c.name }));
      const rows = conflicts.map((c, i) => {
        const gid = `sync-c-${i}`;
        const seg = (val, label) => {
          const input = el('input', { type: 'radio', name: gid, id: `${gid}-${val}`, value: val, class: 'sr-only' });
          input.addEventListener('change', () => { st.res[c.image_id] = val; update(); });
          return { input, label: el('label', { for: `${gid}-${val}`, class: 'seg-opt' }, input, label) };
        };
        const l = seg('local', t('sync.keep_local'));
        const r = seg('remote', t('sync.keep_remote', { tool: name }));
        radios.push({ id: c.image_id, local: l.input, remote: r.input });
        return el('li', { class: 'conflict' },
          el('button', { type: 'button', class: 'conflict-thumb', 'aria-label': c.name, onclick: () => openLightbox(projectId, items, i) },
            el('img', { src: `/api/projects/${projectId}/images/${encodeURIComponent(c.image_id)}/thumb`, alt: '', loading: 'lazy' })),
          el('div', { class: 'conflict-main' },
            el('strong', { class: 'conflict-name', title: c.name }, c.name),
            el('p', { class: 'conflict-kind' }, t(`sync.kind_${c.kind in { removed_local: 1, removed_remote: 1 } ? c.kind : 'labels'}`, { tool: name })),
            el('div', { class: 'conflict-sides' },
              el('p', null, el('span', { class: 'side-tag' }, t('sync.side_local')), perClassText(c.local)),
              el('p', null, el('span', { class: 'side-tag' }, name), perClassText(c.remote)))),
          el('div', { class: 'seg', role: 'radiogroup', 'aria-label': c.name }, l.label, r.label));
      });

      const conflictBox = conflicts.length
        ? el('div', { class: 'sync-conflicts' },
            el('div', { class: 'conflicts-head' },
              el('h4', null, `${t('sync.conflicts')} (${num(conflicts.length)})`),
              el('div', { class: 'conflicts-all' },
                el('button', { type: 'button', class: 'btn small', onclick: () => setAll('local') }, t('sync.all_local')),
                el('button', { type: 'button', class: 'btn small', onclick: () => setAll('remote') }, t('sync.all_remote', { tool: name })))),
            el('p', { class: 'muted' }, t('sync.conflicts_help')),
            el('ul', { class: 'plain conflict-list' }, ...rows))
        : null;

      const onlyNames = (only.examples || []).map((x) => (typeof x === 'string' ? x : (x && x.name) || ''));
      const onlyBox = only.count
        ? el('details', { class: 'sync-diff' },
            el('summary', null, `${t('sync.only', { tool: name })}: ${num(only.count)}`),
            el('p', { class: 'muted' }, t('sync.only_help', { tool: name })),
            onlyNames.length ? el('ul', { class: 'plain only-list' }, ...onlyNames.map((n) => el('li', null, n))) : null,
            only.count > onlyNames.length ? el('p', { class: 'muted' }, t('sync.only_more', { n: num(only.count - onlyNames.length) })) : null)
        : null;

      /* "the tool wins": what exists only in this project is left alone */
      const here = plan.only_here || {};
      const hereTasks = here.tasks || [];
      const hereImages = Number(here.images) || 0;
      const hereBox = hereTasks.length || hereImages
        ? el('details', { class: 'sync-diff' },
            el('summary', null, t('sync.only_here')),
            el('p', { class: 'muted' }, t('sync.only_here_help', { tool: name })),
            el('ul', { class: 'plain only-list' },
              hereImages ? el('li', null, tn('sync.n_images', hereImages)) : null,
              ...hereTasks.map((x) => el('li', null, taskText(x)))))
        : null;
      const direction = plan.direction || 'both';

      apply.addEventListener('click', async () => {
        apply.dataset.busy = '1';
        apply.disabled = true;
        st.error = null;
        try {
          const job2 = await api('POST', `/projects/${projectId}/sync/${tool}/${encodeURIComponent(pid)}/apply`, { json: { resolutions: st.res, default: st.def, delete_tasks: !!(st.del && tDelete.length) } });
          ctx.syncDismissed.add(pid);
          if (job2 && job2.id) ctx.syncTool[job2.id] = tool;
          toast(t('sync.apply_started'), 'info', 3000);
          await jobStarted(job2);
          draw(true);
        } catch (err) {
          apply.dataset.busy = '0';
          if (err.status === 404) st.error = 'stale'; else toastError(err);  // 404: the comparison is gone; 409 only means a job is running
          update();
        }
      });

      /* the comparison is deleted on the server too: another tab cannot apply it afterwards */
      const discard = async () => {
        try {
          await api('DELETE', `/projects/${projectId}/sync/${tool}/${encodeURIComponent(pid)}`);
        } catch (err) {
          if (err.status !== 404) { toastError(err); return; }  // 404: it is gone already
        }
        ctx.syncDismissed.add(pid);
        draw(true);
      };

      const card = el('div', { class: 'card sync-plan', id: 'sync-plan', 'data-plan': pid },
        el('div', { class: 'session-head' },
          el('span', { class: 'session-tool' }, icon(TOOLS.includes(tool) ? tool : 'info', 22), el('strong', null, t('sync.preview_title', { tool: name }))),
          el('span', { class: 'sbadge sbadge-pulled' }, t('sync.review'))),
        el('p', { class: 'session-meta' },
          isHttp(remote.url) ? el('a', { href: remote.url, target: '_blank', rel: 'noopener' }, remote.name || name, ' ', icon('external', 14)) : el('span', null, remote.name || name),
          remote.images != null && !plan.create ? el('span', { class: 'sep', 'aria-hidden': 'true' }, '·') : null,
          remote.images != null && !plan.create ? el('span', null, tn('sync.remote_images', Number(remote.images) || 0, { tool: name })) : null,
          el('span', { class: 'sep', 'aria-hidden': 'true' }, '·'),
          el('strong', { class: 'sync-scope' }, scopeText(plan.scope)),
          el('span', { class: 'sep', 'aria-hidden': 'true' }, '·'),
          el('span', { class: 'sync-direction' }, t(`sync.dir_${direction}`, { tool: name }))),
        dirs,
        nothing ? el('p', { class: 'report-line ok' }, icon('check'), t('sync.in_sync')) : null,
        diffBlock(t('sync.diff_pull', { tool: name }), plan.pull, plan.diff_pull),
        diffBlock(t('sync.diff_push', { tool: name }), plan.push, plan.diff_push),
        hereBox,
        onlyBox,
        conflictBox,
        errorBox,
        el('div', { class: 'session-actions' },
          nothing ? null : apply,
          el('button', { type: 'button', class: 'btn', onclick: discard }, nothing ? t('common.close') : t('sync.discard')),
          hint));
      update();
      return card;
    }

    /* a failed comparison or apply: say why in the interface language and what to do next */
    function failedCard(job) {
      const tool = syncJobTool(job) || (ctx.links[0] && ctx.links[0].tool) || 'cvat';
      const close = () => { ctx.syncDismissed.add(job.id); draw(true); };
      let actions;
      if (isStale(job)) {
        actions = [el('button', { type: 'button', class: 'btn primary', onclick: async () => {
          ctx.syncDismissed.add(job.id);
          try { await startSync(tool); } catch (err) { toastError(err); }
          draw(true);
        } }, t('sync.again'))];
      } else if (isRemoteGone(job)) {
        actions = [el('button', { type: 'button', class: 'btn danger-outline', onclick: async () => {
          try {
            await api('DELETE', `/projects/${projectId}/links/${tool}`);
            ctx.syncDismissed.add(job.id);
            toast(t('sync.unlinked', { tool: toolName(tool) }), 'success', 4000);
            await refreshProject();
            draw(true);
          } catch (err) { toastError(err); }
        } }, t('sync.unlink')), el('button', { type: 'button', class: 'btn', onclick: close }, t('common.close'))];
      } else {
        actions = [el('button', { type: 'button', class: 'btn', onclick: close }, t('common.close'))];
      }
      return el('div', { class: 'sync-error card', role: 'alert' },
        el('p', { class: 'report-line' }, icon('warning'), jobError(job)), ...actions);
    }

    function draw(force) {
      const job = planJob();
      const plan = job ? job.result : null;
      const st = plan && states.get(plan.plan_id);
      const applying = ctx.jobs.some((j) => j.kind === 'sync_apply' && isActive(j));
      const latest = [...ctx.jobs].sort((a, b) => String(b.created).localeCompare(String(a.created))).find((j) => /^sync_/.test(j.kind));
      const failedJob = latest && latest.status === 'failed' && !ctx.syncDismissed.has(latest.id) && !plan ? latest : null;
      const done = !plan && !applying ? resultJob() : null;
      const sig = JSON.stringify([ctx.links, plan && plan.plan_id, planningTools(), applying, failedJob && failedJob.id, done && done.id, (ctx.tasksList || []).length]);
      if (!force && sig === lastSig && !(st && st.error)) return;
      lastSig = sig;
      fill(box, ...[
        /* CVAT has its own card above; a link of a tool that is not synchronised any more can only be forgotten here */
        ctx.links.some((l) => l.tool !== 'cvat')
          ? el('ul', { class: 'plain link-list' }, ...ctx.links.filter((l) => l.tool !== 'cvat').map(linkCard))
          : null,
        planningTools().length ? el('p', { class: 'muted', 'aria-live': 'polite' }, t('sync.comparing')) : null,
        applying ? el('p', { class: 'muted', 'aria-live': 'polite' }, t('sync.apply_started')) : null,
        failedJob ? failedCard(failedJob) : null,
        done ? resultCard(done) : null,
        job ? planCard(job) : null].filter(Boolean));
    }

    ctx.syncDraw = () => {
      if (ctx.panels && ctx.panels.annotations && ctx.panels.annotations.mounted) { draw(false); if (ctx.cvatDraw) ctx.cvatDraw(false); }
    };
    const reveal = () => {
      node.scrollIntoView({ behavior: 'smooth', block: 'start' });
      node.focus({ preventScroll: true });
      const card = node.querySelector('.sync-plan');
      if (card) { card.classList.add('flash'); setTimeout(() => card.classList.remove('flash'), 2500); }
    };
    return { node, draw: () => draw(true), reveal };
  }

  return { loadLinks, startSync, syncBlock };
}
