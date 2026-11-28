/* The project page: its header (open in CVAT and FiftyOne, edit, delete), the step bar, the activity strip, the six steps and the page
 * object they share, and the address (#step) that follows the current step. */
import { createActivities } from './activities.js';
import { annotationsPanel } from './annotations.js';
import { curationPanel } from './curation.js';
import { datasetsPanel } from './datasets.js';
import { openLightbox } from './grid.js';
import { imagesPanel } from './images.js';
import { modelsPanel } from './models.js';
import { issueName, taskBadge, taskName, taskPicker } from './names.js';
import { createStepper } from './stepper.js';
import { trainingPanel } from './training.js';
import { loadFeatures } from '../shell/features.js';
import { editedNotice, editedSession, refreshEdited, watchEdited } from '../integrations/edited.js';
import { createSync } from '../integrations/sync.js';
import { linkBadge, originBadge } from '../integrations/tools.js';
import { api } from '../shell/api.js';
import { choiceDialog, closeOpenDialogs, confirmDialog, openModal } from '../shell/dialogs.js';
import { add, el, fill, icon } from '../shell/dom.js';
import { num, t } from '../shell/i18n.js';
import { toast, toastError } from '../shell/toast.js';
import { chipsInput, classLine, problemRow } from '../shell/widgets.js';

/** What changed when annotations came back: counts, per class, unknown classes, problems, examples. */
export function diffView(projectId, diff) {
  const d = diff || {};
  const sign = (n, ch) => `${ch}${num(n)}`;
  const stat = (cls, n, label) => el('div', { class: `stat ${cls}` }, el('strong', null, num(n)), el('span', null, label));
  const rows = Object.entries(d.per_class || {}).filter(([, c]) => (c.added || 0) + (c.removed || 0) + (c.edited || 0) > 0);
  const examples = (d.examples || []).slice(0, 20);
  const items = examples.map((x) => ({ id: x.image_id, name: x.name }));
  const unknown = d.unknown_classes || [];
  const problems = d.problems || [];
  const tags = d.tags || {};
  const tagsChanged = (tags.added || 0) + (tags.removed || 0) > 0;
  const tagChip = (x) => ((x.tags_added || 0) + (x.tags_removed || 0) > 0
    ? el('span', { class: 'tags-chip', title: t('annotate.diff_tags') }, `${t('annotate.diff_tags')} ${sign(x.tags_added || 0, '+')} ${sign(x.tags_removed || 0, '−')}`)
    : null);

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
      stat('del', d.removed || 0, t('annotate.diff_removed')),
      tagsChanged ? el('div', { class: 'stat tags' }, el('strong', null, `${sign(tags.added || 0, '+')} ${sign(tags.removed || 0, '−')}`), el('span', null, t('annotate.diff_tags'))) : null),
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
                  tagChip(x),
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

export function initProject() {
  const root = document.getElementById('project-root');
  const projectId = (window.VL || {}).projectId;
  const ctx = {
    id: projectId,
    project: null,
    tab: 'images',
    jobs: [],
    busy: new Set(),  // work that leaving the page would lose (an upload in progress)
    jobStatus: new Map(),
    openWhenDone: new Map(),  // job id -> the tab reserved for its App (or null)
    pollTimer: null,
    panels: {},
    weightless: new Set(),  // runs whose weights just went with a deleted model (until the list says has_weights: false)
    links: [],
    syncTool: {},
    syncDismissed: new Set(),
    jobsSeen: false,
    watchers: [],  // called after every redraw of the activity strip (the buttons that depend on running jobs)
    fiftyoneEdited: null,  // answer of GET /annotate: the sessions and how many images were edited in the App (#700)
    editedWatchers: [],  // called with it each time it is read
  };
  const STEPS = ['images', 'annotations', 'curation', 'datasets', 'training', 'models'];

  // ---- header, stepper, panels skeleton
  const headerBox = el('header', { class: 'project-head card' });
  const stepper = el('nav', { class: 'stepper', 'aria-label': t('project.steps') });
  const panelsBox = el('div', { class: 'panels' });
  const jobsBox = el('section', { class: 'jobs-bar', 'aria-label': t('jobs.title') });

  /** The Annotations step on the session that holds the edits made in the FiftyOne App. */
  function showEdited() {
    const session = editedSession(ctx.fiftyoneEdited);
    showTab('annotations', true);
    setTimeout(() => ctx.panels.annotations.reveal(session ? session.id : null), 80);
  }

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
    renderHeader();  // its Open in FiftyOne button depends on the image count
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
    /* the page takes the colour of the task type of the project (app.css: data-task, #621) */
    document.documentElement.dataset.task = p.task;
    fill(headerBox,
      el('div', { class: 'project-head-main' },
        el('div', { class: 'project-title-row' },
          el('h1', null, p.name),
          el('span', { class: 'project-id' }, `#${p.id}`),
          taskBadge(p.task),
          ctx.links.length ? ctx.links.map(linkBadge) : originBadge(p.origin)),
        p.description ? el('p', { class: 'muted' }, p.description) : null,
        (p.classes || []).length ? classLine(p.classes, p.task) : null,
        /* edits made in the FiftyOne App that are not in the project yet (#700): the line leads to the session */
        editedNotice(ctx.fiftyoneEdited, { id: 'head-edited', onReview: showEdited })),
      el('div', { class: 'project-head-actions' },
        el('button', { type: 'button', class: 'btn', onclick: openEdit }, t('project.edit')),
        el('button', { type: 'button', class: 'btn danger-outline', onclick: openDelete }, t('project.delete'))));
  }

  const { gateDatasetNode, gateNode, hasImages, renderStepper, showTab, tabsView } = createStepper({ ctx, projectId, STEPS, stepper });

  // ---- edit & delete
  function openEdit() {
    const p = ctx.project;
    const nameInput = el('input', { id: 'e-name', type: 'text', value: p.name, maxlength: '100', required: true,
      'aria-describedby': ctx.links.some((l) => l.tool === 'cvat') ? 'e-name-hint' : null });
    const descInput = el('textarea', { id: 'e-desc', rows: '3', maxlength: '2000' }, p.description || '');
    const picker = taskPicker('e-task', p.task);
    const chips = chipsInput('e-classes', p.classes || []);
    /* a regression project names its classes targets (#619); the label follows the type chosen here */
    const classesLabel = el('label', { for: 'e-classes' }, '');
    const wording = () => {
      const targets = picker.get() === 'regression';
      classesLabel.textContent = t(targets ? 'field.targets' : 'field.classes');
      const input = chips.node.querySelector('input');
      if (input) input.placeholder = t(targets ? 'targets.placeholder' : 'classes.placeholder');
    };
    picker.node.addEventListener('change', wording);
    wording();
    const save = el('button', { type: 'submit', class: 'btn primary' }, t('common.save'));
    const dlg = el('dialog', { class: 'dialog wide', 'aria-label': t('project.edit') },
      el('form', { class: 'dialog-body form', method: 'dialog' },
        el('h2', null, t('project.edit')),
        el('div', { class: 'form-cols' },
          el('div', null,
            /* a project linked to CVAT: the new name goes there on the next comparison, like a task name (#587) */
            el('div', { class: 'field' }, el('label', { for: 'e-name' }, t('field.name')), nameInput,
              ctx.links.some((l) => l.tool === 'cvat') ? el('p', { class: 'hint', id: 'e-name-hint' }, t('project.rename_cvat')) : null),
            el('div', { class: 'field' }, el('label', { for: 'e-desc' }, t('field.description')), descInput),
            el('div', { class: 'field' }, classesLabel, chips.node)),
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

  // ---- what the steps share
  /* whether a comparison found nothing to review: nothing to bring, send, rename or decide */
  /* a plan with only renames (a task, a class or the project renamed on one side) is not empty: Apply sends
     or takes the names (#587) */
  const planIsEmpty = (plan) => {
    const n = (x) => (Array.isArray(x) ? x.length : x && typeof x === 'object' ? 1 : Number(x) || 0);
    return !['pull', 'push', 'add', 'conflicts', 'remove_remote', 'remove_local', 'restore_remote', 'restore_local',
      'tasks_create', 'tasks_new', 'tasks_delete', 'tasks_rename', 'renamed_classes', 'project_rename'].some((k) => n(plan[k]));
  };

  /* The page object, built here and only here: the activity strip and the sync read it when they are made and
     add their helpers to it; then each step takes it whole. */
  const page = {
    projectId, ctx, jobsBox, refreshProject, refreshCounts, renderStepper, showMissing, showTab, tabsView,
    hasImages, planIsEmpty, diffView, openLightbox, issueName,
  };
  const { pollJobs, renderJobs, ...activities } = createActivities(page);
  Object.assign(page, activities);
  const { loadLinks, ...sync } = createSync(page);
  Object.assign(page, sync);

  // ---- assemble
  ctx.panels = { images: imagesPanel(page), annotations: annotationsPanel(page), curation: curationPanel(page), datasets: datasetsPanel(page), training: trainingPanel(page), models: modelsPanel(page) };
  add(panelsBox, gateNode, gateDatasetNode, ...STEPS.map((s) => ctx.panels[s].node));
  fill(root, headerBox, el('div', { class: 'project-body' }, stepper, panelsBox), jobsBox);
  renderJobs();
  const startHash = location.hash.replace('#', '');  // read before anything rewrites the address
  ctx.tab = STEPS.includes(startHash) ? startHash : 'images';
  /* FiftyOne may not be installed (#218): its buttons say so once the server answered (#658) */
  ctx.editedWatchers.push(() => { if (ctx.project) renderHeader(); });
  refreshEdited(page).catch(() => { /* the count is a hint: the Annotations step asks again */ });
  watchEdited(page);   // and again each time the person comes back from the App
  loadFeatures();   // the buttons registered with fiftyoneButton are marked when it answers
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
