/* activelabel web interface: an ES module, vanilla JS, no dependencies. Talks to the JSON API under /api. */
import { setMessages, fmt, t, tr, tn, num, fmtBytes, lower } from './shell/i18n.js';
import { add, el, esc, fill, icon } from './shell/dom.js';
import { toast, toastError } from './shell/toast.js';
import { choiceDialog, closeOpenDialogs, confirmDialog, openModal } from './shell/dialogs.js';
import { initTheme } from './shell/theme.js';
import { chipsInput, classLine, problemRow } from './shell/widgets.js';
import { issueName, partName, taskBadge, taskName, taskPicker } from './cv/names.js';
import { api, errorMessage, errorParams, nameErrorParams, validationText } from './shell/api.js';
import { isHttp, linkBadge, originBadge, statusLabel, toolName } from './integrations/tools.js';
import { initProjects } from './pages/projects.js';
import { initNew } from './pages/new.js';
import { initImport } from './pages/import.js';
import { openLightbox } from './cv/grid.js';
import { createActivities } from './cv/activities.js';
import { createSync } from './integrations/sync.js';
import { annotationsPanel } from './cv/annotations.js';
import { curationPanel } from './cv/curation.js';
import { modelsPanel } from './cv/models.js';
import { datasetsPanel } from './cv/datasets.js';
import { imagesPanel } from './cv/images.js';
import { createStepper } from './cv/stepper.js';
import { trainingPanel } from './cv/training.js';

  const VL = window.VL || {};
  setMessages(window.I18N || {}, VL.locale || 'en');
  /* error params named with the task, tool and status names (the API module cannot import them) */
  nameErrorParams({
    parts: (p) => lower(partName(p)),
    tool: (tool) => toolName(tool),
    task: (task) => lower(taskName(task)),
    type: (task) => lower(taskName(task)),
    kind: (kind) => lower(taskName(kind)),
    from_task: (task) => lower(taskName(task)),
    to_task: (task) => lower(taskName(task)),
    status: (status) => lower(statusLabel(status)),
  });

  // ---------------------------------------------------------------- annotation sessions (shared bits)

  /** What changed when annotations came back: counts, per class, unknown classes, problems, examples. */
  function diffView(projectId, diff) {
    const d = diff || {};
    const sign = (n, ch) => `${ch}${num(n)}`;
    const stat = (cls, n, label) => el('div', { class: `stat ${cls}` }, el('strong', null, num(n)), el('span', null, label));
    const rows = Object.entries(d.per_class || {}).filter(([, c]) => (c.added || 0) + (c.removed || 0) + (c.edited || 0) > 0);
    const examples = (d.examples || []).slice(0, 20);
    const items = examples.map((x) => ({ id: x.image_id, name: x.name }));
    const unknown = d.unknown_classes || [];
    const problems = d.problems || [];

    const table = rows.length
      ? el('div', { class: 'table-wrap' },
          el('table', { class: 'diff-table', 'aria-label': t('annotate.diff_per_class') },
            el('thead', null, el('tr', null,
              el('th', { scope: 'col' }, t('annotate.diff_class')),
              el('th', { scope: 'col', class: 'num' }, t('annotate.diff_added')),
              el('th', { scope: 'col', class: 'num' }, t('annotate.diff_edited')),
              el('th', { scope: 'col', class: 'num' }, t('annotate.diff_removed')))),
            el('tbody', null, ...rows.map(([name, c]) => el('tr', null,
              el('th', { scope: 'row' }, name),
              el('td', { class: 'num add' }, num(c.added || 0)),
              el('td', { class: 'num edit' }, num(c.edited || 0)),
              el('td', { class: 'num del' }, num(c.removed || 0)))))))
      : null;

    return el('div', { class: 'diff' },
      el('div', { class: 'diff-stats' },
        stat('images', d.images_changed || 0, t('annotate.diff_images')),
        stat('add', d.added || 0, t('annotate.diff_added')),
        stat('edit', d.edited || 0, t('annotate.diff_edited')),
        stat('del', d.removed || 0, t('annotate.diff_removed'))),
      !(d.images_changed || 0) ? el('p', { class: 'report-line ok' }, icon('check'), t('annotate.diff_none')) : null,
      el('div', { class: 'diff-cols' },
        table ? el('div', null, el('h4', null, t('annotate.diff_per_class')), table) : null,
        examples.length
          ? el('div', null,
              el('h4', null, t('annotate.diff_examples')),
              el('ul', { class: 'plain diff-examples' }, ...examples.map((x, i) => el('li', null,
                el('button', { type: 'button', class: 'example-btn', title: x.name, onclick: () => openLightbox(projectId, items, i) },
                  el('span', { class: 'example-name' }, x.name),
                  el('span', { class: 'example-counts' },
                    el('span', { class: 'add', title: t('annotate.diff_added') }, sign(x.added || 0, '+')),
                    el('span', { class: 'edit', title: t('annotate.diff_edited') }, sign(x.edited || 0, '~')),
                    el('span', { class: 'del', title: t('annotate.diff_removed') }, sign(x.removed || 0, '−'))))))))
          : null),
      unknown.length
        ? el('div', null,
            el('h4', null, t('annotate.diff_unknown')),
            el('p', { class: 'muted' }, t('annotate.diff_unknown_help')),
            el('ul', { class: 'chips static' }, ...unknown.map((c) => el('li', { class: 'chip static is-new' }, el('span', null, c), el('span', { class: 'new-mark' }, t('ann.new'))))))
        : null,
      problems.length
        ? el('div', null, el('h4', null, t('ann.problems')), el('ul', { class: 'plain problems' }, ...problems.map(problemRow)))
        : null);
  }

  // ---------------------------------------------------------------- page: project
  function initProject() {
    const root = document.getElementById('project-root');
    const projectId = VL.projectId;
    const ctx = {
      id: projectId,
      project: null,
      tab: 'images',
      jobs: [],
      busy: new Set(),  // work that leaving the page would lose (an upload in progress)
      jobStatus: new Map(),
      openWhenDone: new Set(),
      pollTimer: null,
      panels: {},
      weightless: new Set(),  // runs whose weights just went with a deleted model (until the list says has_weights: false)
      links: [],
      syncTool: {},
      syncDismissed: new Set(),
      jobsSeen: false,
      watchers: [],  // called after every redraw of the activity strip (the buttons that depend on running jobs)
    };
    const STEPS = ['images', 'annotations', 'curation', 'datasets', 'training', 'models'];

    // ---- header, stepper, panels skeleton
    const headerBox = el('header', { class: 'project-head card' });
    const stepper = el('nav', { class: 'stepper', 'aria-label': t('project.steps') });
    const panelsBox = el('div', { class: 'panels' });
    const jobsBox = el('section', { class: 'jobs-bar', 'aria-label': t('jobs.title') });

    async function refreshProject() {
      try {
        ctx.project = await api('GET', `/projects/${projectId}`);
      } catch (err) {
        if (err.status === 404) { showMissing(); return false; }
        toastError(err);
        return false;
      }
      document.title = `${ctx.project.name} · ${t('app.name')}`;
      await loadLinks();
      renderHeader();
      showTab(ctx.tab);  // also opens a gated step once images arrive
      return true;
    }

    /** Reads the project again and redraws the step summaries, without reopening a step (a step bar click can call it). */
    async function refreshCounts() {
      try {
        ctx.project = await api('GET', `/projects/${projectId}`);
      } catch (err) { return; }
      renderStepper();
    }

    function showMissing() {
      clearTimeout(ctx.pollTimer);
      ctx.pollTimer = null;
      ctx.gone = true;
      fill(root, el('div', { class: 'empty card' },
        el('h1', null, t('project.gone_title')),
        el('p', null, t('project.gone_text')),
        el('a', { class: 'btn primary', href: '/' }, t('notfound.back'))));
    }

    function renderHeader() {
      const p = ctx.project;
      fill(headerBox,
        el('div', { class: 'project-head-main' },
          el('div', { class: 'project-title-row' },
            el('h1', null, p.name),
            el('span', { class: 'project-id' }, `#${p.id}`),
            taskBadge(p.task),
            ctx.links.length ? ctx.links.map(linkBadge) : originBadge(p.origin)),
          p.description ? el('p', { class: 'muted' }, p.description) : null,
          (p.classes || []).length ? classLine(p.classes) : null),
        el('div', { class: 'project-head-actions' },
          el('button', { type: 'button', class: 'btn', onclick: openEdit }, t('project.edit')),
          el('button', { type: 'button', class: 'btn danger-outline', onclick: openDelete }, t('project.delete'))));
    }

    const { gateDatasetNode, gateNode, hasImages, renderStepper, showTab, tabsView } = createStepper({ ctx, projectId, STEPS, stepper });

    // ---- edit & delete
    function openEdit() {
      const p = ctx.project;
      const nameInput = el('input', { id: 'e-name', type: 'text', value: p.name, maxlength: '100', required: true });
      const descInput = el('textarea', { id: 'e-desc', rows: '3', maxlength: '2000' }, p.description || '');
      const picker = taskPicker('e-task', p.task);
      const chips = chipsInput('e-classes', p.classes || []);
      const save = el('button', { type: 'submit', class: 'btn primary' }, t('common.save'));
      const dlg = el('dialog', { class: 'dialog wide', 'aria-label': t('project.edit') },
        el('form', { class: 'dialog-body form', method: 'dialog' },
          el('h2', null, t('project.edit')),
          el('div', { class: 'form-cols' },
            el('div', null,
              el('div', { class: 'field' }, el('label', { for: 'e-name' }, t('field.name')), nameInput),
              el('div', { class: 'field' }, el('label', { for: 'e-desc' }, t('field.description')), descInput),
              el('div', { class: 'field' }, el('label', { for: 'e-classes' }, t('field.classes')), chips.node)),
            el('fieldset', { class: 'field' }, el('legend', null, t('field.task')), picker.node)),
          el('div', { class: 'form-actions' }, save, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel')))));
      dlg.querySelector('form').noValidate = true;  // the app's own message, in the app's language, not the browser's bubble
      dlg.querySelector('form').addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!nameInput.value.trim()) { toastError(new Error(t('new.name_required'))); nameInput.focus(); return; }
        save.disabled = true;
        const body = { name: nameInput.value.trim(), description: descInput.value.trim(), task: picker.get(), classes: chips.get() };
        try {
          /* the server may ask for decisions (task change, classes still in use); ask them one by one */
          let updated = null;
          const decisions = {};
          for (let round = 0; round < 3 && !updated; round += 1) {
            try {
              updated = await api('PATCH', `/projects/${projectId}`, { json: { ...body, ...decisions } });
            } catch (err) {
              const d = err.detail;
              if (err.status !== 409 || !d || !d.code) throw err;
              if (d.code === 'task_change_annotations') {
                const choice = await choiceDialog({
                  title: t('taskchange.title'),
                  text: t('taskchange.text', { from: taskName(d.from_task), to: taskName(d.to_task), images: num(d.images), labels: num(d.labels) }),
                  choices: [
                    ...(d.options.includes('convert') ? [{ value: 'convert', primary: true, label: t('taskchange.convert', { to: taskName(d.to_task) }), help: t(`taskchange.convert_help.${d.to_task}`) }] : []),
                    { value: 'discard', danger: true, label: t('taskchange.discard'), help: t('taskchange.discard_help') },
                  ],
                });
                if (!choice) { save.disabled = false; return; }
                decisions.annotations = choice;
              } else if (d.code === 'classes_in_use') {
                const list = (d.classes || []).map((c) => t('classes.in_use_item', { name: c.name, labels: num(c.labels), images: num(c.images) })).join('; ');
                const ok = await confirmDialog({
                  title: t('classes.in_use_title'),
                  text: t('classes.in_use_text', { list }),
                  confirmLabel: t('classes.in_use_remove'),
                });
                if (!ok) { save.disabled = false; return; }
                decisions.classes_action = 'remove_annotations';
              } else {
                throw err;
              }
            }
          }
          if (!updated) { save.disabled = false; return; }
          ctx.project = { ...ctx.project, ...updated };
          dlg.close();
          toast(t('project.saved'), 'success', 3000);
          await refreshProject();
          Object.values(ctx.panels).forEach((panel) => panel.refresh && panel.refresh());
        } catch (err) {
          save.disabled = false;
          toastError(err);
        }
      });
      dlg.addEventListener('close', () => dlg.remove());
      document.body.appendChild(dlg);
      openModal(dlg);
      nameInput.focus();
    }

    async function openDelete() {
      const p = ctx.project;
      const ok = await confirmDialog({
        title: t('delete.title'),
        text: t('delete.text', { name: p.name }),
        confirmLabel: t('delete.confirm'),
        requireText: p.name,
      });
      if (!ok) return;
      try {
        await api('DELETE', `/projects/${projectId}`, { json: { confirm_name: p.name } });
        location.href = '/';
      } catch (err) {
        toastError(err);
      }
    }

    // ---- what the steps share (the activity strip and the sync are made here; the steps at the end)
    /* whether a comparison found nothing to review: nothing to bring, send, rename or decide */
    const planIsEmpty = (plan) => {
      const n = (x) => (Array.isArray(x) ? x.length : Number(x) || 0);
      return !['pull', 'push', 'add', 'conflicts', 'remove_remote', 'remove_local', 'restore_remote', 'restore_local',
        'tasks_create', 'tasks_new', 'tasks_delete'].some((k) => n(plan[k]));
    };

    /* what the steps share with the page: the panels take this one object (their own modules, #155); the parts
       made by the modules below are added to it as they are created */
    const page = {
      projectId, ctx, jobsBox, refreshProject, refreshCounts, renderStepper, showMissing, showTab, tabsView,
      hasImages, planIsEmpty, diffView, openLightbox, issueName,
    };
    const { isActive, isRemoteGone, isStale, jobError, jobStarted, kindActive, pollJobs, renderJobs, syncJobTool } = createActivities(page);
    Object.assign(page, { isActive, isRemoteGone, isStale, jobError, jobStarted, kindActive, syncJobTool });
    const { loadLinks, startSync, syncBlock } = createSync(page);
    Object.assign(page, { startSync, syncBlock });

    // ---- assemble
    ctx.panels = { images: imagesPanel(page), annotations: annotationsPanel(page), curation: curationPanel(page), datasets: datasetsPanel(page), training: trainingPanel(page), models: modelsPanel(page) };
    add(panelsBox, gateNode, gateDatasetNode, ...STEPS.map((s) => ctx.panels[s].node));
    fill(root, headerBox, el('div', { class: 'project-body' }, stepper, panelsBox), jobsBox);
    renderJobs();
    const startHash = location.hash.replace('#', '');  // read before anything rewrites the address
    ctx.tab = STEPS.includes(startHash) ? startHash : 'images';
    refreshProject().then((ok) => {
      if (!ok) return;
      pollJobs();
      ctx.panels.datasets.count().then(() => { if (ctx.tab === 'training') showTab('training'); });
      ctx.panels.training.count();
      ctx.panels.models.count();
    });
    /* back on this tab after a while: the project may have been deleted elsewhere (nothing polls while idle) */
    document.addEventListener('visibilitychange', () => { if (!document.hidden && !ctx.gone) refreshProject(); });
    /* leaving (a link, the language switch, closing the tab) in the middle of an upload, or with an import report still waiting for a decision, loses it: ask first */
    window.addEventListener('beforeunload', (e) => {
      if (ctx.busy.size || document.querySelector('.report-card .options [data-action]:not([data-action="other"])')) { e.preventDefault(); e.returnValue = ''; }
    });
    /* a link to another step of this page (#annotations...) changes the step without reloading */
    window.addEventListener('hashchange', () => {
      const step = location.hash.replace('#', '');
      if (!STEPS.includes(step) || step === ctx.tab) return;
      closeOpenDialogs();  // a window of the step left must not stay on top of the new one
      showTab(step);
    });
  }

  // ---------------------------------------------------------------- start
  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    const page = document.body.dataset.page;
    if (page === 'projects') initProjects();
    else if (page === 'new') initNew();
    else if (page === 'import') initImport();
    else if (page === 'project') initProject();
  });

// for the unit tests in tests/js until the helpers move to their own modules (#155)
export { fmt, t, tr, tn, num, fmtBytes, esc, errorMessage, errorParams, validationText, isHttp };
