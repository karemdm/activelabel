/* activelabel web interface: an ES module, vanilla JS, no dependencies. Talks to the JSON API under /api. */
import {
  LOCALE, nf1, setMessages, fmt, t, tr, tn, num, fmtDate, fmtDateTime, fmtBytes, lower,
} from './shell/i18n.js';
import { add, el, esc, fill, icon, redraw } from './shell/dom.js';
import { toast, toastError } from './shell/toast.js';
import {
  choiceDialog, closeOpenDialogs, confirmDialog, menuButton, once, openModal,
} from './shell/dialogs.js';
import { initTheme } from './shell/theme.js';
import { chipsInput, classLine, emptyState, problemRow, progressBar } from './shell/widgets.js';
import {
  CHUNK, SKIP_FILES, annotationStatus, issueName, partName, taskBadge, taskName, taskPicker,
} from './cv/names.js';
import { api, errorMessage, errorParams, nameErrorParams, validationText } from './shell/api.js';
import { isHttp, jobMessage, linkBadge, originBadge, statusLabel, toolName } from './integrations/tools.js';
import { initProjects } from './pages/projects.js';
import { initNew } from './pages/new.js';
import { initImport } from './pages/import.js';
import { imageGrid, openLightbox } from './cv/grid.js';
import { createActivities } from './cv/activities.js';
import { createSync } from './integrations/sync.js';
import { annotationsPanel } from './cv/annotations.js';
import { curationPanel } from './cv/curation.js';
import { modelsPanel } from './cv/models.js';
import { datasetsPanel, staleCount } from './cv/datasets.js';

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

    function stepStatus(step) {
      const c = (ctx.project && ctx.project.counts) || {};
      if (step === 'images') {
        if (!(c.active || c.excluded)) return t('status.images_none');
        return `${tn('status.images_active', c.active || 0)}${c.excluded ? ` · ${tn('status.excluded', c.excluded)}` : ''}`;
      }
      if (step === 'annotations') {
        const waiting = c.preannotated ? ` · ${tn('status.to_review', c.preannotated)}` : '';
        const none = c.empty ? ` · ${tn('status.empty', c.empty)}` : '';
        return c.annotated > 0 || c.empty > 0
          ? `${annotationStatus(c)}${c.annotated > 0 ? none : ''}${waiting} · ${tn('status.classes', (ctx.project.classes || []).length)}`
          : t('status.annotations_none');
      }
      if (step === 'training') {
        if (!ctx.runCount) return t('status.runs_none');
        return `${tn('status.runs', ctx.runCount)}${ctx.runActive ? ` · ${tn('status.runs_active', ctx.runActive)}` : ''}`;
      }
      if (step === 'datasets') return ctx.datasetCount ? tn('status.versions', ctx.datasetCount) : t('status.versions_none');
      if (step === 'models') {
        if (!ctx.modelCount) return t('status.models_none');
        return `${tn('status.models', ctx.modelCount)}${ctx.currentModel ? ` · ${t('status.current_model', { n: ctx.currentModel })}` : ''}`;
      }
      return c.analyzed > 0 ? tn('status.issues', c.with_issues || 0) : t('status.not_analyzed');
    }

    /* where a step stands, for the marker on the bar: done, needs attention, busy or still to do */
    function stepState(step) {
      const c = (ctx.project && ctx.project.counts) || {};
      if (step === 'images') return c.active > 0 ? 'done' : 'todo';
      if (step === 'annotations') return c.preannotated > 0 ? 'attention' : (c.annotated > 0 || c.empty > 0) ? 'done' : 'todo';
      if (step === 'curation') return c.analyzed > 0 ? (c.with_issues > 0 ? 'attention' : 'done') : 'todo';
      if (step === 'datasets') return ctx.datasetCount ? 'done' : 'todo';
      if (step === 'training') return ctx.runActive ? 'busy' : ctx.runCount ? 'done' : 'todo';
      return ctx.modelCount ? 'done' : 'todo';
    }

    /* the step bar groups the steps: the data (1-3) and the model made from it (4-6) */
    const STEP_GROUPS = [['data', ['images', 'annotations', 'curation']], ['model', ['datasets', 'training', 'models']]];
    function renderStepper() {
      const item = (step) => {
        const i = STEPS.indexOf(step);
        const state = stepState(step);
        return el('button', {
          type: 'button',
          class: `step is-${state}${ctx.tab === step ? ' current' : ''}`,
          'aria-current': ctx.tab === step ? 'step' : null,
          'data-step': step,
          'data-act': step,
          onclick: () => showTab(step),
        },
          el('span', { class: 'step-num', 'aria-hidden': 'true' }, state === 'done' ? icon('check', 16) : state === 'attention' ? '!' : String(i + 1)),
          el('span', { class: 'step-text' },
            el('span', { class: 'step-title' }, t(`step.${step}`), state === 'todo' ? null : el('span', { class: 'sr-only' }, ` (${t(`step.state_${state}`)})`), step === 'annotations' ? el('span', { class: 'optional step-tag' }, t('step.optional')) : null),
            el('span', { class: 'step-status', title: stepStatus(step) }, stepStatus(step))));
      };
      /* the page takes the colour of the area of the current step (app.css: data-area) */
      document.documentElement.dataset.area = STEP_GROUPS.find(([, steps]) => steps.includes(ctx.tab))[0];
      redraw(stepper, ...STEP_GROUPS.map(([group, steps]) => el('div', { class: 'step-group', role: 'group', 'aria-labelledby': `sg-${group}`, 'data-group': group },
        el('p', { class: 'step-group-title', id: `sg-${group}` }, t(`project.group_${group}`)),
        ...steps.map(item))));
      /* on a narrow screen the bar scrolls sideways: keep the current step in view */
      requestAnimationFrame(() => {  // after the bar is in the page and has its width
        const here = stepper.querySelector('.step.current');
        if (here && stepper.scrollWidth > stepper.clientWidth) {
          const box = stepper.getBoundingClientRect();
          const at = here.getBoundingClientRect();
          stepper.scrollLeft += at.left - box.left - (box.width - at.width) / 2;
        }
      });
    }

    /* Tabs inside a step: one subject per tab. The last tab is remembered per project and step. */
    function tabsView(key, label, tabs, initial) {
      const store = `al.tab.${projectId}.${key}`;
      let saved = null;
      try { saved = localStorage.getItem(store); } catch (err) { /* not remembered */ }
      let current = null;
      const parts = new Map();
      const list = el('div', { class: 'step-tabs', role: 'tablist', 'aria-label': label });
      function select(id, focus = false) {
        if (!parts.has(id)) return;
        current = id;
        for (const [tid, part] of parts) {
          const on = tid === id;
          part.button.setAttribute('aria-selected', String(on));
          part.button.tabIndex = on ? 0 : -1;
          part.pane.hidden = !on;
        }
        try { localStorage.setItem(store, id); } catch (err) { /* not remembered */ }
        if (focus) parts.get(id).button.focus();
        const tab = tabs.find((x) => x.id === id);
        if (tab && tab.onShow) tab.onShow();
      }
      function keys(e) {
        const ids = [...parts.keys()];
        const at = ids.indexOf(current);
        const next = { ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: ids.length - 1 }[e.key];
        if (next == null) return;
        e.preventDefault();
        select(ids[(next + ids.length) % ids.length], true);
      }
      const panes = tabs.map((tab) => {
        const id = `tab-${key}-${tab.id}`;
        const count = el('span', { class: 'step-tab-count', hidden: true });
        const button = el('button', { type: 'button', role: 'tab', id, class: 'step-tab', 'aria-controls': `${id}-pane`, onclick: () => select(tab.id), onkeydown: keys }, tab.label, count);
        const pane = el('div', { role: 'tabpanel', id: `${id}-pane`, class: 'step-tab-pane', 'aria-labelledby': id }, ...tab.nodes);
        parts.set(tab.id, { button, pane, count });
        add(list, button);
        return pane;
      });
      select(parts.has(saved) ? saved : (typeof initial === 'function' ? initial() : initial) || tabs[0].id);
      return {
        node: el('div', { class: 'tabs-view' }, list, ...panes),
        select,
        current: () => current,
        /** true when the tab came from the person's last visit */
        remembered: parts.has(saved),
        /** a number next to a tab's name (hidden for null) */
        setCount: (id, n) => { const part = parts.get(id); if (!part) return; part.count.hidden = n == null; part.count.textContent = n == null ? '' : num(n); },
      };
    }

    /* steps 2 and 3 work on the project's images: without images they only say so */
    const hasImages = () => {
      const c = (ctx.project && ctx.project.counts) || {};
      return (c.active || 0) + (c.excluded || 0) > 0;  // excluded images can be restored in step 3
    };
    const gateNode = el('section', { class: 'card gate', hidden: true },
      el('p', { class: 'report-line' }, icon('info'), t('gate.need_images')),
      el('button', { type: 'button', class: 'btn primary', onclick: () => showTab('images') }, t('gate.go_images')));

    /* step 5 needs a dataset version: without one it sends the person to step 4 */
    const gateDatasetNode = el('section', { class: 'card gate', hidden: true },
      el('p', { class: 'report-line' }, icon('info'), t('gate.need_dataset')),
      el('button', { type: 'button', class: 'btn primary', onclick: () => showTab('datasets') }, t('gate.go_datasets')));

    function showTab(step, scrollToPanel = false) {
      ctx.tab = step;
      /* a step the person opens is a history entry (Back goes to the previous step); the first one and redraws are not */
      if (location.hash !== `#${step}`) history[ctx.shown ? 'pushState' : 'replaceState'](null, '', `#${step}`);
      ctx.shown = true;
      const noImages = step !== 'images' && step !== 'models' && !hasImages();
      const noDataset = step === 'training' && !noImages && !ctx.datasetCount;
      const gated = noImages || noDataset;
      for (const s of STEPS) ctx.panels[s].node.hidden = gated || s !== step;
      gateNode.hidden = !noImages;
      gateDatasetNode.hidden = !noDataset;
      renderStepper();
      if (gated) return;
      const panel = ctx.panels[step];
      if (!panel.mounted) { panel.mounted = true; panel.mount(); } else if (panel.show) panel.show();
      if (scrollToPanel) panel.node.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

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

    // ================================================================ synchronisation with CVAT

    // ================================================================ step 1: images
    function imagesPanel(page) {
      const node = el('section', { class: 'panel', id: 'panel-images', 'aria-labelledby': 'h-images' });
      const panel = { node, mounted: false };
      let uploading = false;
      let controller = null;

      const fileInput = el('input', { type: 'file', multiple: true, class: 'sr-only', id: 'file-input', accept: 'image/*,.zip,application/zip', tabindex: '-1' });
      const folderInput = el('input', { type: 'file', multiple: true, class: 'sr-only', id: 'folder-input', tabindex: '-1' });
      folderInput.setAttribute('webkitdirectory', '');
      folderInput.setAttribute('directory', '');
      const progressBox = el('div', { class: 'upload-progress', hidden: true });
      const reportBox = el('div', { class: 'report', 'aria-live': 'polite' });
      let staged = null; // files dropped on the empty state, waiting for "Create task"
      const usable = (entries) => entries.filter((e) => !SKIP_FILES.has(e.name.split('/').pop().toLowerCase()));

      const drop = el('div', { class: 'dropzone', id: 'dropzone' },
        icon('upload', 36),
        el('p', { class: 'drop-title' }, t('images.drop_title')),
        el('p', { class: 'muted' }, t('images.drop_help')),
        el('div', { class: 'drop-actions' },
          el('button', { type: 'button', class: 'btn primary', onclick: () => fileInput.click() }, t('images.choose_files')),
          el('button', { type: 'button', class: 'btn', onclick: () => folderInput.click() }, t('images.choose_folder'))));

      /** Makes `zone` accept dropped files and folders; calls onEntries(entries). */
      function wireDrop(zone, onEntries) {
        ['dragenter', 'dragover'].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add('over'); }));
        ['dragleave', 'drop'].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove('over'); }));
        zone.addEventListener('drop', async (e) => onEntries(await filesFromDrop(e.dataTransfer)));
      }
      wireDrop(drop, (entries) => startUpload(entries));
      fileInput.addEventListener('change', () => { startUpload([...fileInput.files].map((f) => ({ file: f, name: f.webkitRelativePath || f.name }))); fileInput.value = ''; });
      folderInput.addEventListener('change', () => { startUpload([...folderInput.files].map((f) => ({ file: f, name: f.webkitRelativePath || f.name }))); folderInput.value = ''; });

      async function filesFromDrop(dt) {
        const out = [];
        const items = dt.items ? [...dt.items] : [];
        if (items.length && items[0].webkitGetAsEntry) {
          const roots = items.map((i) => i.webkitGetAsEntry()).filter(Boolean); // before any await
          const walk = async (entry, prefix) => {
            if (entry.isFile) {
              const file = await new Promise((res, rej) => entry.file(res, rej));
              out.push({ file, name: prefix + entry.name });
            } else if (entry.isDirectory) {
              const reader = entry.createReader();
              for (;;) {
                const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
                if (!batch.length) break;
                for (const child of batch) await walk(child, `${prefix}${entry.name}/`);
              }
            }
          };
          for (const r of roots) await walk(r, '');
          return out;
        }
        return [...dt.files].map((f) => ({ file: f, name: f.name }));
      }

      const emptyReport = () => ({ added: 0, duplicates: [], duplicates_count: 0, rejected: [], rejected_count: 0, other_files: [], other_files_count: 0 });
      const mergeReport = (acc, r) => {
        acc.added += Number(r.added) || 0;
        for (const key of ['duplicates', 'rejected', 'other_files']) {
          const list = r[key] || [];
          acc[key].push(...list);
          acc[`${key}_count`] += Number(r[`${key}_count`]) || list.length;
        }
      };

      async function startUpload(entries) {
        const files = usable(entries);
        if (!files.length || uploading) return;
        const jobSize = jobSizeInput.value.trim();
        if (jobSize && !/^[1-9]\d*$/.test(jobSize)) {
          jobSizeInput.setAttribute('aria-invalid', 'true');
          jobSizeInput.focus();
          toast(t('tasks.job_size_invalid'), 'error');
          return;
        }
        jobSizeInput.removeAttribute('aria-invalid');
        uploading = true;
        ctx.busy.add('upload');
        controller = new AbortController();
        fill(reportBox);
        if (!newDlg.open) newDlg.showModal();
        setStage('uploading');
        const merged = emptyReport();
        let sent = 0;
        let cancelled = false;
        let created = null; // the task made by the first chunk
        let failure = null; // why the upload stopped (a rejected name, a server error)
        const draw = () => {
          progressBox.hidden = false;
          fill(progressBox,
            el('div', { class: 'upload-line' },
              el('strong', null, t('images.uploading', { done: num(sent), total: num(files.length) })),
              el('button', { type: 'button', class: 'btn small', id: 'upload-cancel', onclick: () => controller.abort() }, t('common.cancel'))),
            progressBar(sent, files.length, t('images.upload_progress')));
        };
        draw();
        for (let i = 0; i < files.length; i += CHUNK) {
          const chunk = files.slice(i, i + CHUNK);
          const form = new FormData();
          chunk.forEach((e) => form.append('files', e.file, e.name));
          try {
            if (!created) {
              form.append('name', nameInput.value.trim());
              form.append('job_size', jobSize);
              form.append('open', 'true');
              const res = await api('POST', `/projects/${projectId}/tasks`, { body: form, signal: controller.signal });
              created = res.task;
              mergeReport(merged, res.report || {});
            } else {
              mergeReport(merged, await api('POST', `/projects/${projectId}/tasks/${created.id}/images`, { body: form, signal: controller.signal }));
            }
          } catch (err) {
            if (err.name === 'AbortError') cancelled = true; else failure = err;  // the reason is shown in the result card
            break;
          }
          sent += chunk.length;
          draw();
        }
        if (created) {
          // close the task (its images become fixed), also after a cancel or an error: best effort
          let removed = false;
          try {
            const closed = await api('PATCH', `/projects/${projectId}/tasks/${created.id}`, { json: { open: false } });
            removed = !!(closed && closed.removed); // an upload that added nothing leaves no task
          } catch (e) { /* the next task closes it */ }
          if (removed) {
            created = null;
          } else {
            nameInput.value = '';
            jobSizeInput.value = '';
            toast(t('tasks.created', { name: created.name }), 'success', 4000);
          }
        }
        uploading = false;
        ctx.busy.delete('upload');
        staged = null;
        drawStaged();
        setStage(created ? 'done' : 'form'); // nothing was created: back to the form, the report says why
        if (cancelled) toast(t('images.cancelled', { done: num(sent) }), 'info');
        renderReport(merged, sent < files.length, failure);
        await refreshProject();
        await loadTasks();
        grid.reload();
        if (ctx.panels.curation.mounted) ctx.panels.curation.refresh();
      }

      /* the server's rejection reasons are short English phrases; "too many entries (more than N)" carries a number */
      const reasonText = (reason) => {
        const text = String(reason);
        const key = /^too many entries/i.test(text) ? 'too_many_entries' : text.toLowerCase().replace(/[^a-z]+/g, '_');
        return tr(`reason.${key}`, text);
      };
      const nameOf = (x) => (typeof x === 'string' ? x : (x && x.name) || '');

      function expandable(title, rows) {
        return el('details', { class: 'expand' }, el('summary', null, title), el('ul', { class: 'plain scroll-box' }, ...rows));
      }

      function renderReport(r, incomplete, failure = null) {
        const rows = [];
        if (r.added) rows.push(el('p', { class: 'report-line ok' }, icon('check'), tn('images.added', r.added)));
        else rows.push(el('p', { class: 'report-line bad' }, icon('warning'), t('images.none_added')));  // no green mark, no task
        if (failure) rows.push(el('p', { class: 'report-line bad', role: 'alert' }, icon('warning'), String(failure.message || failure)));
        if (r.duplicates_count) {
          rows.push(el('div', { class: 'report-line' }, icon('info'),
            el('div', null, tn('images.duplicates', r.duplicates_count), ' ', el('span', { class: 'muted' }, t('images.duplicates_note')),
              expandable(t('images.show_list'), r.duplicates.map((d) => el('li', null, nameOf(d)))))));
        }
        if (r.rejected_count) {
          rows.push(el('div', { class: 'report-line bad' }, icon('warning'),
            el('div', null, tn('images.rejected', r.rejected_count),
              expandable(t('images.show_list'), r.rejected.map((x) => el('li', null, el('span', null, nameOf(x)), ' ', el('span', { class: 'muted' }, `— ${reasonText(x.reason || '')}`)))))));
        }
        if (incomplete) rows.push(el('p', { class: 'muted' }, t('images.incomplete')));
        const annotationLike = (r.other_files || []).some((f) => /\.(json|txt|xml|ya?ml|csv)$/i.test(nameOf(f)));
        if (r.other_files_count && annotationLike) {
          rows.push(el('div', { class: 'callout' }, icon('info'),
            el('div', null,
              el('strong', null, t('images.annotations_found')),
              el('p', { class: 'muted' }, tn('images.annotations_found_n', r.other_files_count)),
              el('button', { type: 'button', class: 'btn primary', onclick: () => { newDlg.close(); showTab('annotations', true); } }, t('images.go_annotations')))));
        }
        fill(reportBox, el('div', { class: 'card report-card' }, el('h3', null, t('images.result')), ...rows));
      }

      // ---- tasks (cards), and the filter of the grid
      let tasks = [];
      let filter = { task: null, job: null, dataset: null }; // dataset: {n, part|fold} set from step 4
      let renaming = null; // id of the task being renamed
      let renameDraft = null; // text typed so far (kept if the cards are redrawn)
      const expanded = new Set();
      let focusAfter = null; // selector to focus once the cards are drawn

      const tasksBox = el('div', { class: 'tasks-wrap' });
      const filterBox = el('div', { class: 'task-filter' });
      const taskById = (id) => tasks.find((x) => x.id === id);

      const grid = imageGrid({
        projectId,
        query: () => ({
          status: 'active',
          task: filter.task,
          job: filter.job,
          dataset: filter.dataset,
          emptyText: filter.task != null || filter.dataset ? t('tasks.filter_empty') : t('grid.no_images'),
        }),
        onOpen: (items, i) => openLightbox(projectId, items, i),
      });
      const gridTitle = el('h3', { class: 'section-title', id: 'h-images-grid', tabindex: '-1' }, t('images.grid_title'));

      /** Reads the tasks; `refresh` makes the server ask CVAT for the job states again instead of reusing a read of the last minute. */
      async function loadTasks(refresh = false) {
        try {
          tasks = (await api('GET', `/projects/${projectId}/tasks${refresh ? '?refresh=true' : ''}`)).tasks || [];
        } catch (err) {
          toastError(err);
          return;
        }
        const current = taskById(filter.task);
        if (!current) filter = { task: null, job: null, dataset: filter.dataset };
        else if (filter.job != null && !current.jobs.some((j) => j.n === filter.job)) filter.job = null;
        if (!tasks.some((x) => x.id === renaming)) renaming = null;
        nameInput.placeholder = t('tasks.name_placeholder', { id: tasks.reduce((m, x) => Math.max(m, x.id), 0) + 1 });
        drawTasks();
        drawFilter();
        imageTabs.setCount('tasks', tasks.length);
      }

      function setFilter(task, job, { reveal = false, dataset = null } = {}) {
        filter = { task, job, dataset };
        drawFilter();
        grid.reload();
        if (reveal) {
          imageTabs.select('gallery');
          gridTitle.scrollIntoView({ behavior: 'smooth', block: 'start' });
          gridTitle.focus({ preventScroll: true });
        }
      }

      function drawFilter() {
        const task = taskById(filter.task);
        const focused = document.activeElement && document.activeElement.closest && document.activeElement.closest('.task-filter') ? document.activeElement.id : null;
        const taskSelect = el('select', {
          id: 'filter-task',
          onchange: (e) => setFilter(e.target.value === '' ? null : Number(e.target.value), null, { dataset: filter.dataset }),
        },
        el('option', { value: '' }, t('tasks.filter_all_tasks')),
        ...tasks.map((x) => el('option', { value: String(x.id), selected: x.id === filter.task }, x.name)));
        const jobSelect = task && task.jobs.length > 1
          ? el('select', {
            id: 'filter-job',
            onchange: (e) => setFilter(filter.task, e.target.value === '' ? null : Number(e.target.value), { dataset: filter.dataset }),
          },
          el('option', { value: '' }, t('tasks.filter_all_jobs')),
          ...task.jobs.map((j) => el('option', { value: String(j.n), selected: j.n === filter.job }, t('tasks.job_option', { n: j.n, first: num(j.first), last: num(j.last) }))))
          : null;
        const chipText = task
          ? (filter.job != null ? t('tasks.filter_chip_job', { name: task.name, n: filter.job }) : t('tasks.filter_chip', { name: task.name }))
          : null;
        fill(filterBox,
          el('div', { class: 'task-filter-row' },
            el('div', { class: 'field inline' }, el('label', { for: 'filter-task', class: 'sr-only' }, t('tasks.filter_task')), taskSelect),
            jobSelect ? el('div', { class: 'field inline' }, el('label', { for: 'filter-job', class: 'sr-only' }, t('tasks.filter_job')), jobSelect) : null,
            chipText
              ? el('span', { class: 'chip filter-chip', role: 'status' }, chipText,
                el('button', { type: 'button', class: 'chip-x', id: 'filter-clear', 'aria-label': t('tasks.filter_clear'), title: t('tasks.filter_clear'), onclick: () => { setFilter(null, null, { dataset: filter.dataset }); const s = document.getElementById('filter-task'); if (s) s.focus(); } }, icon('close', 14)))
              : null,
            filter.dataset
              ? el('span', { class: 'chip filter-chip', role: 'status', id: 'filter-dataset-chip' },
                t('ds.filter_chip', { n: filter.dataset.n, part: filter.dataset.fold != null ? t('ds.fold', { n: filter.dataset.fold + 1 }) : partName(filter.dataset.part) }),
                el('button', { type: 'button', class: 'chip-x', id: 'filter-dataset-clear', 'aria-label': t('ds.filter_clear'), title: t('ds.filter_clear'), onclick: () => { setFilter(filter.task, filter.job); const s = document.getElementById('filter-task'); if (s) s.focus(); } }, icon('close', 14)))
              : null));
        if (focused) { const f = document.getElementById(focused); if (f) f.focus(); }
      }

      function taskCard(task) {
        const cid = `task-${task.id}`;
        const src = task.source && task.source.video ? task.source : null;
        const isRenaming = renaming === task.id;
        const filtered = filter.task === task.id;
        const manyJobs = task.jobs.length > 1;
        const open = manyJobs && expanded.has(task.id);
        const thinBar = (done, total, label) => { const b = progressBar(done, total, label); b.classList.add('thin'); return b; };

        let title;
        if (isRenaming) {
          const input = el('input', { type: 'text', id: `${cid}-name`, value: renameDraft == null ? task.name : renameDraft, 'aria-label': t('tasks.rename_label', { name: task.name }), autocomplete: 'off', maxlength: '100' });
          input.addEventListener('input', () => { renameDraft = input.value; });
          const cancel = () => { renaming = null; renameDraft = null; focusAfter = `#${cid}-rename`; drawTasks(); };
          const form = el('form', { class: 'task-rename', novalidate: true,
            onsubmit: async (e) => {
              e.preventDefault();
              const name = input.value.trim();
              if (!name) { toastError(new Error(t('error.task_name_empty'))); input.focus(); return; }
              if (name === task.name) { cancel(); return; }
              try {
                await api('PATCH', `/projects/${projectId}/tasks/${task.id}`, { json: { name } });
                toast(t('tasks.renamed'), 'success', 3000);
                renaming = null;
                renameDraft = null;
                focusAfter = `#${cid}-rename`;
                await loadTasks();
              } catch (err) { toastError(err); if (!(await reloadIfChanged(err))) input.focus(); }
            } },
          input,
          el('button', { type: 'submit', class: 'btn small primary' }, t('common.save')),
          el('button', { type: 'button', class: 'btn small', onclick: cancel }, t('common.cancel')));
          input.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.preventDefault(); cancel(); } });
          title = form;
          focusAfter = `#${cid}-name`;
        } else {
          title = el('h4', { class: 'task-name' },
            el('button', { type: 'button', class: 'task-title', title: t('tasks.see_images_task', { name: task.name }), 'aria-pressed': filtered ? 'true' : 'false', onclick: () => taskSelectSync(task.id, null) },
              src ? el('span', { class: 'task-video-icon', title: t('video.mode_video') }, icon('video', 16)) : null, task.name));
        }

        /* the open icon of a task or of one of its jobs: a menu with the tools it can be opened in */
        const openMenu = (job) => menuButton('external',
          job ? t('tasks.open_menu_job', { n: job, name: task.name }) : t('tasks.open_menu', { name: task.name }),
          () => openItems(task, job));
        const iconButton = (name, label, extra, onclick, id) => el('button', { type: 'button', class: `icon-btn small${extra ? ` ${extra}` : ''}`, id, 'aria-label': label, title: label, onclick }, icon(name, 16));
        let everyText = null;
        if (src && src.step) {
          everyText = src.fps
            ? t('video.meta_every_seconds', { s: nf1.format(Math.round((src.step / src.fps) * 100) / 100) })
            : tn('video.meta_every_frames', src.step);
        } else if (src && src.every_seconds) {
          everyText = t('video.meta_every_seconds', { s: nf1.format(src.every_seconds) });
        }
        const meta = [
          tn('tasks.meta_images', task.active),
          task.images > task.active ? tn('tasks.meta_excluded', task.images - task.active) : null,
          everyText,
          task.job_size ? t('tasks.meta_job_size', { n: num(task.job_size) }) : null,
          tn('tasks.meta_jobs', task.jobs.length),
        ].filter(Boolean).join(' · ');
        const progressText = t('tasks.progress', { done: num(task.annotated), total: num(task.active) });

        const jobsList = manyJobs
          ? el('ul', { class: 'plain task-jobs', id: `${cid}-jobs`, hidden: !open },
            ...task.jobs.map((j) => {
              const on = filtered && filter.job === j.n;
              /* state of the job in the linked CVAT project (new, in progress, completed, rejected) */
              const rj = task.remote && (task.remote.jobs || []).find((x) => x.n === j.n);
              const state = rj && rj.state ? String(rj.state).replace(/\s+/g, '_') : '';
              const badge = state
                ? el('span', { class: `job-state state-${state}` }, tr(`tasks.job_state.${state}`, rj.state))
                : null;
              /* who works on it and at which stage, as text under the row */
              const who = [rj && rj.stage ? tr(`tasks.job_stage.${rj.stage}`, rj.stage) : null,
                rj && rj.assignee ? t('tasks.job_assignee', { name: rj.assignee }) : null].filter(Boolean).join(' · ');
              return el('li', { class: 'job-li' },
                el('button', { type: 'button', class: `job-row${on ? ' is-filtered' : ''}${badge ? ' has-state' : ''}`, 'aria-pressed': on ? 'true' : 'false',
                  'aria-label': t('tasks.see_images_job', { n: j.n, name: task.name }), onclick: () => taskSelectSync(task.id, j.n) },
                el('span', { class: 'job-name' }, t('tasks.job_name', { n: j.n })),
                el('span', { class: 'job-range muted' }, `${num(j.first)}–${num(j.last)}`),
                thinBar(j.annotated, j.active, t('tasks.progress', { done: num(j.annotated), total: num(j.active) })),
                el('span', { class: 'job-count' }, `${num(j.annotated)}/${num(j.active)}`),
                badge,
                j.images > j.active ? el('span', { class: 'job-excluded muted' }, tn('tasks.job_excluded', j.images - j.active)) : null,
                who ? el('span', { class: 'job-who muted small' }, who) : null),
                openMenu(j.n));
            }))
          : null;

        return el('li', { class: `task-item card${filtered ? ' is-filtered' : ''}`, id: cid, 'aria-label': task.name },
          el('div', { class: 'task-head' },
            title,
            isRenaming ? null : el('div', { class: 'task-actions' },
              openMenu(null),
              iconButton('edit', `${t('tasks.rename')}: ${task.name}`, '', () => { renaming = task.id; renameDraft = null; drawTasks(); }, `${cid}-rename`),
              iconButton('trash', `${t('tasks.delete')}: ${task.name}`, 'danger', () => deleteTask(task)),
              manyJobs
                ? el('button', { type: 'button', class: 'icon-btn small task-toggle', 'aria-expanded': open ? 'true' : 'false', 'aria-controls': `${cid}-jobs`,
                  'aria-label': `${open ? t('tasks.hide_jobs') : t('tasks.show_jobs')}: ${task.name}`, title: open ? t('tasks.hide_jobs') : t('tasks.show_jobs'),
                  onclick: () => {
                    if (expanded.has(task.id)) expanded.delete(task.id); else expanded.add(task.id);
                    focusAfter = `#${cid} .task-toggle`;
                    drawTasks();
                  } }, icon('chevron', 16))
                : null)),
          el('p', { class: 'task-meta muted' }, meta),
          /* an upload cut off midway: the app closed the task, some images may be missing (#434) */
          task.incomplete ? el('p', { class: 'task-incomplete', title: t('tasks.incomplete_help') }, icon('error', 14), t('tasks.incomplete')) : null,
          el('div', { class: 'task-progress' },
            thinBar(task.annotated, task.active, progressText),
            el('span', { class: 'task-progress-text' }, progressText)),
          jobsList);
      }

      const taskSelectSync = (task, job) => setFilter(task, job, { reveal: true });

      /** The Update button: reads the tasks again, CVAT job states included, and keeps the focus on it. */
      async function refreshTasks() {
        focusAfter = '#tasks-refresh';
        await loadTasks(true);
        toast(t('tasks.refreshed'), 'success', 2500);   // a visible answer even when nothing changed
      }

      function drawTasks() {
        const head = el('div', { class: 'tasks-head' },
          el('h3', { class: 'tasks-title sr-only' }, t('tasks.title'), tasks.length ? el('span', { class: 'count-badge' }, num(tasks.length)) : null),   // the tab already names and counts them
          tasks.length ? el('div', { class: 'tasks-head-actions' },
            tasks.some((x) => x.remote)   // only tasks linked to CVAT have job states to read again
              ? el('button', { type: 'button', class: 'btn', id: 'tasks-refresh', title: t('tasks.refresh_help'), onclick: refreshTasks }, icon('refresh', 16), t('tasks.refresh'))
              : null,
            el('button', { type: 'button', class: 'btn primary', id: 'new-task-btn', onclick: () => openNewTask() }, icon('plus', 16), t('tasks.new_title'))) : null);
        let body;
        if (tasks.length) {
          body = el('ul', { class: 'plain task-list' }, ...tasks.map(taskCard));
        } else {
          body = el('div', { class: 'dropzone empty-tasks' },
            icon('images', 40),
            el('p', { class: 'drop-title' }, t('tasks.none')),
            el('button', { type: 'button', class: 'btn primary', id: 'new-task-btn', onclick: () => openNewTask() }, icon('plus', 16), t('tasks.new_title')));
          wireDrop(body, (entries) => openNewTask(entries));
        }
        fill(tasksBox, head, body);
        if (focusAfter) {
          const f = tasksBox.querySelector(focusAfter);
          focusAfter = null;
          if (f) f.focus();
        }
      }

      /** What a task (or one of its jobs) can be opened in. A disabled item says why. */
      function openItems(task, job) {
        const keypoints = ctx.project && ctx.project.task === 'keypoints';
        const remote = task.remote || {};
        const remoteJob = job ? (remote.jobs || []).find((x) => x.n === job) : null;
        const cvatUrl = (remoteJob && remoteJob.url) || remote.url;
        const cvatOk = task.cvat === 'ready' && isHttp(cvatUrl);
        const cvatReasons = { not_configured: 'tasks.cvat_not_configured', not_linked: 'tasks.cvat_not_linked', unreachable: 'tasks.cvat_unreachable' };
        const items = [keypoints
          ? { label: t('tasks.menu_edit_fo'), disabled: true, reason: t('tasks.fo_no_keypoints') }
          : { label: t('tasks.menu_edit_fo'), onclick: () => editInFiftyOne(task, job) }];
        if (keypoints) items.push({ label: t('tasks.menu_view_fo'), onclick: () => openTaskInFiftyOne(task, job) });
        items.push(cvatOk
          ? { label: t('tasks.menu_cvat'), href: cvatUrl }
          : { label: t('tasks.menu_cvat'), disabled: true, reason: t(cvatReasons[task.cvat] || 'tasks.cvat_not_synced') });
        return items;
      }

      /** The FiftyOne view of a task or job, to look at (a video task shows each video as an ordered sequence). */
      async function openTaskInFiftyOne(task, job) {
        try {
          const started = await api('POST', `/projects/${projectId}/fiftyone`, { json: job ? { task: task.id, job } : { task: task.id } });
          if (started && started.id) ctx.openWhenDone.add(started.id);
          toast(t('cur.fo_starting'), 'info', 4000);
          await jobStarted(started);
        } catch (err) { toastError(err); }
      }

      /** An edit session on exactly the active images of the task or job (its card is in Annotations). */
      async function editInFiftyOne(task, job) {
        try {
          const started = await api('POST', `/projects/${projectId}/annotate`, { json: { tool: 'fiftyone', scope: 'task', task: task.id, ...(job ? { job } : {}) } });
          toast(t('annotate.send_started', { tool: toolName('fiftyone') }), 'info', 4000);
          await jobStarted(started);
        } catch (err) { toastError(err); }
      }

      /** A 404/409 means the task changed elsewhere (another tab): show the real state instead of a dead card. */
      async function reloadIfChanged(err) {
        if (err.status !== 404 && err.status !== 409) return false;
        renaming = null;
        renameDraft = null;
        await refreshProject();
        await loadTasks();
        grid.reload();
        return true;
      }

      async function deleteTask(task) {
        const ok = await confirmDialog({
          title: t('tasks.delete_title', { name: task.name }),
          text: tn('tasks.delete_text', task.images),
          confirmLabel: t('tasks.delete_confirm'),
        });
        if (!ok) return;
        try {
          const r = await api('DELETE', `/projects/${projectId}/tasks/${task.id}`, { json: { confirm: true } });
          toast(tn('tasks.deleted', Number(r && r.deleted) || 0), 'success', 5000);
        } catch (err) { toastError(err); await reloadIfChanged(err); return; }
        expanded.delete(task.id);
        await refreshProject();
        await loadTasks();
        grid.reload();
        if (ctx.panels.curation.mounted) ctx.panels.curation.refresh();
      }

      // ---- "new task" dialog (name, images per job, then the files; upload progress and report)
      const nameInput = el('input', { type: 'text', id: 'task-name', autocomplete: 'off', maxlength: '100' });  // the server's limit
      const jobSizeInput = el('input', { type: 'number', id: 'task-job-size', min: '1', step: '1', inputmode: 'numeric', 'aria-describedby': 'task-job-size-hint' });
      jobSizeInput.addEventListener('input', () => jobSizeInput.removeAttribute('aria-invalid'));
      const stagedBox = el('div', { class: 'staged', hidden: true });
      const imagesBox = el('div', { class: 'nt-images' }, drop, stagedBox);
      const videoBox = el('div', { class: 'nt-video', hidden: true });
      const modeBtns = {};
      const modeTabs = el('div', { class: 'tabs nt-modes', role: 'group', 'aria-label': t('tasks.new_title') },
        ...['images', 'video'].map((m) => (modeBtns[m] = el('button', { type: 'button', class: `tab${m === 'images' ? ' active' : ''}`, 'aria-pressed': m === 'images' ? 'true' : 'false', id: `nt-mode-${m}`, onclick: () => setMode(m) }, t(`video.mode_${m}`)))));
      const fieldsBox = el('div', { class: 'new-task-fields' },
        el('div', { class: 'field' }, el('label', { for: 'task-name' }, t('tasks.name')), nameInput),
        el('div', { class: 'field' },
          el('label', { for: 'task-job-size' }, t('tasks.job_size')), jobSizeInput,
          el('p', { class: 'hint', id: 'task-job-size-hint' }, t('tasks.job_size_hint'))));
      const formBox = el('div', { class: 'nt-form' }, modeTabs, fieldsBox, imagesBox, videoBox);
      let mode = 'images';
      function setMode(m) {
        if (uploading || vid.busy) return;
        mode = m;
        for (const k of Object.keys(modeBtns)) {
          modeBtns[k].classList.toggle('active', k === m);
          modeBtns[k].setAttribute('aria-pressed', k === m ? 'true' : 'false');
        }
        imagesBox.hidden = m !== 'images';
        videoBox.hidden = m !== 'video';
      }

      // ---- video tab
      const POLL_LOST = 3;  // failed polls in a row before the person is told and can leave
      const vid = { busy: false, xhr: null, video: null, taskMade: false, plan: null, seq: 0, timer: null, unit: 'frames', polling: null, jobId: null, pollFails: 0 };
      const VIDEO_EXT = /\.(mp4|avi|mov|mkv|webm|m4v)$/i;
      const vFile = el('input', { type: 'file', class: 'sr-only', id: 'video-input', accept: '.mp4,.avi,.mov,.mkv,.webm,.m4v,video/*', tabindex: '-1' });
      const vDrop = el('div', { class: 'dropzone', id: 'video-dropzone' },
        icon('video', 36),
        el('p', { class: 'drop-title' }, t('video.drop_title')),
        el('p', { class: 'muted' }, t('video.drop_help')),
        el('div', { class: 'drop-actions' }, el('button', { type: 'button', class: 'btn primary', onclick: () => vFile.click() }, t('video.choose'))));
      const vUp = el('div', { class: 'upload-progress', hidden: true });
      const vSel = el('div', { class: 'video-sel', hidden: true });
      const vRun = el('div', { class: 'video-run', hidden: true, 'aria-live': 'polite' });
      add(videoBox, vFile, vDrop, vUp, vSel, vRun);
      wireDrop(vDrop, (entries) => pickVideo(entries.length ? entries[0].file : null));
      vFile.addEventListener('change', () => { pickVideo(vFile.files[0] || null); vFile.value = ''; });

      const pad2 = (n) => String(n).padStart(2, '0');
      const fmtClock = (sec) => {
        const s = Math.max(0, Math.round(Number(sec) || 0));
        const h = Math.floor(s / 3600);
        return h ? `${h}:${pad2(Math.floor((s % 3600) / 60))}:${pad2(s % 60)}` : `${pad2(Math.floor(s / 60))}:${pad2(s % 60)}`;
      };
      /** "120" -> frame 120; "1:30" or "01:02:03" -> frame at that time. null when invalid. */
      function parseFrame(text, fps) {
        const s = String(text).trim();
        if (/^\d+$/.test(s)) return Number(s);
        const m = /^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$/.exec(s);
        if (!m) return null;
        const sec = Number(m[1] || 0) * 3600 + Number(m[2]) * 60 + Number(m[3]);
        return Math.round(sec * fps);
      }
      const vStart = el('input', { type: 'text', id: 'video-start', autocomplete: 'off', inputmode: 'text', 'aria-describedby': 'video-start-alt' });
      const vStop = el('input', { type: 'text', id: 'video-stop', autocomplete: 'off', inputmode: 'text', 'aria-describedby': 'video-stop-alt' });
      const vEvery = el('input', { type: 'number', id: 'video-every', min: '0', step: 'any', inputmode: 'decimal', value: '1' });
      const vStartAlt = el('span', { class: 'muted alt', id: 'video-start-alt' });
      const vStopAlt = el('span', { class: 'muted alt', id: 'video-stop-alt' });
      const unitBtns = {};
      const unitSwitch = el('div', { class: 'tabs', role: 'group', 'aria-label': t('video.every') },
        ...['frames', 'seconds'].map((u) => (unitBtns[u] = el('button', { type: 'button', class: `tab${u === 'frames' ? ' active' : ''}`, 'aria-pressed': u === 'frames' ? 'true' : 'false', onclick: () => setUnit(u) }, t(`video.unit_${u}`)))));
      const vInfo = el('p', { class: 'video-info muted' });
      const vSummary = el('p', { class: 'video-summary', 'aria-live': 'polite' });
      const vStrip = el('ul', { class: 'plain video-strip' });
      const vCreate = el('button', { type: 'button', class: 'btn primary', id: 'video-create', onclick: () => createFromVideo() }, t('tasks.create'));
      const vAnother = el('button', { type: 'button', class: 'btn', onclick: () => discardVideo() }, t('video.change'));
      add(vSel,
        vInfo,
        el('div', { class: 'new-task-fields' },
          el('div', { class: 'field' }, el('label', { for: 'video-start' }, t('video.start')), vStart, vStartAlt),
          el('div', { class: 'field' }, el('label', { for: 'video-stop' }, t('video.end')), vStop, vStopAlt)),
        el('div', { class: 'field video-every' },
          el('label', { for: 'video-every' }, t('video.every')),
          el('div', { class: 'video-every-row' }, vEvery, unitSwitch)),
        vSummary, vStrip,
        el('div', { class: 'drop-actions' }, vCreate, vAnother));
      for (const i of [vStart, vStop, vEvery]) i.addEventListener('input', () => { i.removeAttribute('aria-invalid'); schedulePlan(); });

      function setUnit(u) {
        vid.unit = u;
        for (const k of Object.keys(unitBtns)) {
          unitBtns[k].classList.toggle('active', k === u);
          unitBtns[k].setAttribute('aria-pressed', k === u ? 'true' : 'false');
        }
        const v = vid.video;
        if (v && v.fps) {
          const cur = Number(vEvery.value);
          if (cur > 0) vEvery.value = String(u === 'seconds' ? Math.round((cur / v.fps) * 100) / 100 : Math.max(1, Math.round(cur * v.fps)));
        }
        schedulePlan();
      }

      /** Reads the form; returns the request body for the plan/create calls, or null when invalid. */
      function selection(mark) {
        const v = vid.video;
        if (!v) return null;
        const start = parseFrame(vStart.value, v.fps);
        const stop = parseFrame(vStop.value, v.fps);
        const every = Number(vEvery.value);
        const bad = [];
        if (start == null || start >= v.frames) bad.push(vStart);
        if (stop == null || stop >= v.frames || (start != null && stop < start)) bad.push(vStop);
        if (!(every > 0) || (vid.unit === 'frames' && !Number.isInteger(every))) bad.push(vEvery);
        if (mark) bad.forEach((i) => i.setAttribute('aria-invalid', 'true'));
        if (bad.length) return null;
        return vid.unit === 'seconds' ? { start, stop, every_seconds: every } : { start, stop, step: every };
      }

      function updateAlt() {
        const v = vid.video;
        const alt = (input, out) => {
          const f = parseFrame(input.value, v.fps);
          const isTime = /:/.test(input.value);
          out.textContent = f == null ? '' : isTime ? t('video.as_frame', { n: num(f) }) : t('video.as_time', { time: fmtClock(f / v.fps) });
        };
        alt(vStart, vStartAlt);
        alt(vStop, vStopAlt);
      }

      function schedulePlan() {
        if (!vid.video) return;
        updateAlt();
        clearTimeout(vid.timer);
        vid.plan = null;
        vCreate.disabled = true;
        vid.timer = setTimeout(runPlan, 300);
      }

      async function runPlan() {
        const body = selection(true);
        const my = ++vid.seq;
        if (!body) { vSummary.textContent = t('video.invalid'); fill(vStrip); return; }
        try {
          const p = await api('POST', `/projects/${projectId}/videos/${vid.video.id}/plan`, { json: body });
          if (my !== vid.seq || !vid.video) return;
          vid.plan = p;
          vCreate.disabled = !(p.frames > 0);
          const stepText = vid.unit === 'seconds'
            ? t('video.every_seconds_text', { s: num(Number(vEvery.value)), n: num(p.step) })
            : tn('video.every_frames_text', p.step);
          vSummary.textContent = p.frames > 0 ? tn('video.summary', p.frames, { step: stepText }) : t('video.invalid');
          drawStrip(p);
        } catch (err) {
          if (my !== vid.seq) return;
          vSummary.textContent = err.message;
          fill(vStrip);
        }
      }

      function drawStrip(p) {
        const v = vid.video;
        const k = Math.min(8, p.frames);
        const items = [];
        for (let j = 0; j < k; j++) {
          const idx = k === 1 ? p.first : p.first + p.step * Math.round((j * (p.frames - 1)) / (k - 1));
          items.push(el('li', null,
            el('img', { src: `/api/projects/${projectId}/videos/${v.id}/frame?n=${idx}&w=160`, alt: t('video.frame_alt', { n: num(idx) }), loading: 'lazy' }),
            el('span', { class: 'muted' }, fmtClock(idx / v.fps))));
        }
        fill(vStrip, ...items);
      }

      function showVideo(v) {
        vid.video = v;
        vid.taskMade = false;
        vDrop.hidden = true;
        vUp.hidden = true;
        vSel.hidden = false;
        vInfo.textContent = [fmtClock(v.duration), t('video.fps_n', { n: nf1.format(v.fps) }), `${v.width}×${v.height}`, tn('video.frames_n', v.frames)].join(' · ');
        vStart.value = '0';
        vStop.value = String(Math.max(0, v.frames - 1));
        vEvery.value = '1';
        vid.unit = 'frames';
        setUnit('frames');
        updateAlt();
        runPlan();
      }

      function pickVideo(file) {
        if (!file || vid.busy) return;
        if (!VIDEO_EXT.test(file.name)) { toast(t('video.bad_type'), 'error'); return; }
        vid.busy = true;
        vDrop.hidden = true;
        const form = new FormData();
        form.append('file', file, file.name);
        const xhr = new XMLHttpRequest();
        vid.xhr = xhr;
        const draw = (done, total) => {
          vUp.hidden = false;
          fill(vUp,
            el('div', { class: 'upload-line' },
              el('strong', null, t('video.uploading', { name: file.name })),
              el('button', { type: 'button', class: 'btn small', onclick: () => xhr.abort() }, t('common.cancel'))),
            progressBar(done, total, t('images.upload_progress')));
        };
        draw(0, 1);
        const back = () => { vid.busy = false; vid.xhr = null; vUp.hidden = true; vDrop.hidden = !!vid.video; };
        xhr.upload.addEventListener('progress', (e) => { if (e.lengthComputable) draw(e.loaded, e.total); });
        xhr.addEventListener('abort', back);
        xhr.addEventListener('error', () => { back(); toast(t('common.network_error'), 'error', 10000); });
        xhr.addEventListener('load', () => {
          let data = null;
          try { data = JSON.parse(xhr.responseText); } catch (e) { data = null; }
          back();
          if (xhr.status >= 200 && xhr.status < 300 && data) showVideo(data);
          else toastError(new Error(errorMessage(data, xhr.status)));
        });
        xhr.open('POST', `/api/projects/${projectId}/videos`);
        xhr.setRequestHeader('Accept', 'application/json');
        xhr.send(form);
      }

      /** Drops the uploaded video that no task uses (best effort; 409 = a task uses it). */
      async function discardVideo(quiet) {
        const v = vid.video;
        clearTimeout(vid.timer);
        vid.seq += 1;
        vid.video = null;
        vSel.hidden = true;
        vSummary.textContent = '';
        fill(vStrip);
        if (!quiet) vDrop.hidden = false;
        if (v && !vid.taskMade) {
          try { await api('DELETE', `/projects/${projectId}/videos/${v.id}`); } catch (e) { /* in use or already gone */ }
        }
      }

      function resetVideo() {
        if (vid.xhr) vid.xhr.abort();
        clearTimeout(vid.polling);
        vid.polling = null;
        vid.busy = false;
        vid.jobId = null;
        vid.pollFails = 0;
        vRun.hidden = true;
        fill(vRun);
        modeTabs.hidden = false;
        fieldsBox.hidden = false;
        vUp.hidden = true;
        vDrop.hidden = false;
        vSel.hidden = true;
      }

      async function createFromVideo() {
        if (vid.busy || !vid.video) return;
        const sel = selection(true);
        if (!sel) return;
        const jobSize = jobSizeInput.value.trim();
        if (jobSize && !/^[1-9]\d*$/.test(jobSize)) {
          jobSizeInput.setAttribute('aria-invalid', 'true');
          jobSizeInput.focus();
          toast(t('tasks.job_size_invalid'), 'error');
          return;
        }
        const body = { video: vid.video.id, name: nameInput.value.trim(), ...sel };
        if (jobSize) body.job_size = Number(jobSize);
        vid.busy = true;
        vid.taskMade = true; // from now on the server owns the video
        modeTabs.hidden = true;
        fieldsBox.hidden = true;
        vSel.hidden = true;
        vRun.hidden = false;
        closeBtn.hidden = true;
        closeX.hidden = true;
        fill(vRun, el('p', { class: 'muted' }, t('video.starting')));
        try {
          const job = await api('POST', `/projects/${projectId}/tasks/from-video`, { json: body });
          vid.jobId = job.id;
          vid.pollFails = 0;
          drawRun(job);  // the Cancel button is there from the start, not after the first poll
          pollJob(job.id);
        } catch (err) {
          vid.taskMade = false;
          showRunEnd(false, err.message);
        }
      }

      function drawRun(job) {
        fill(vRun,
          el('div', { class: 'upload-line' },
            el('strong', null, t('video.creating')),
            el('button', { type: 'button', class: 'btn small', onclick: async () => { try { await api('POST', `/jobs/${job.id}/cancel`); } catch (e) { toastError(e); } } }, t('jobs.stop'))),
          progressBar(job.done, job.total, t('video.creating')),
          job.message ? el('p', { class: 'muted job-msg' }, jobMessage(job)) : null);
      }

      async function pollJob(id) {
        let job;
        try {
          job = await api('GET', `/jobs/${id}`);
        } catch (err) {
          vid.pollFails += 1;
          if (vid.pollFails >= POLL_LOST && !vRun.querySelector('.poll-lost')) {
            /* the server does not answer: say so, and let the person leave (the job goes on and shows in the activity bar) */
            add(vRun, el('p', { class: 'report-line bad poll-lost', role: 'alert' }, icon('warning'), t('video.poll_lost')));
            closeBtn.textContent = t('common.close');
            closeBtn.hidden = false;
            closeX.hidden = false;
          }
          vid.polling = setTimeout(() => pollJob(id), 1500);
          return;
        }
        if (vid.jobId !== id) return;
        if (job.status === 'queued' || job.status === 'running') {
          if (vid.pollFails >= POLL_LOST) { closeBtn.hidden = true; closeX.hidden = true; }
          vid.pollFails = 0;
          drawRun(job);
          vid.polling = setTimeout(() => pollJob(id), 1500);
          return;
        }
        vid.polling = null;
        const frames = (job.result && (job.result.frames || job.result.images)) || (vid.plan && vid.plan.frames) || 0;
        if (job.status === 'done') {
          nameInput.value = '';
          jobSizeInput.value = '';
          showRunEnd(true, tn('video.created', frames));
        } else {
          vid.taskMade = !!(job.result && job.result.task);
          showRunEnd(false, job.status === 'cancelled' ? t('video.cancelled') : job.error || tr(`job.status.${job.status}`, job.status));
        }
        await refreshProject();
        await loadTasks();
        grid.reload();
        if (ctx.panels.curation.mounted) ctx.panels.curation.refresh();
      }

      function showRunEnd(ok, text) {
        vid.busy = false;
        fill(vRun,
          el('p', { class: `report-line ${ok ? 'ok' : 'bad'}` }, icon(ok ? 'check' : 'warning'), text),
          ok ? null : el('div', { class: 'drop-actions' }, el('button', { type: 'button', class: 'btn', onclick: () => {
            vRun.hidden = true; modeTabs.hidden = false; fieldsBox.hidden = false;
            if (vid.video) { vSel.hidden = false; schedulePlan(); } else vDrop.hidden = false;
          } }, t('video.back'))));
        if (ok) { vid.video = null; vid.taskMade = true; }
        closeBtn.textContent = t(ok ? 'common.close' : 'common.cancel');
        closeBtn.hidden = false;
        closeX.hidden = false;
      }
      const closeBtn = el('button', { type: 'button', class: 'btn', id: 'new-task-close', onclick: () => newDlg.close() }, t('common.cancel'));
      const closeX = el('button', { type: 'button', class: 'icon-btn', 'aria-label': t('common.close'), title: t('common.close'), onclick: () => newDlg.close() }, icon('close', 18));
      const newDlg = el('dialog', { class: 'dialog wide new-task-dialog', 'aria-label': t('tasks.new_title') },
        el('div', { class: 'dialog-body' },
          el('div', { class: 'dlg-head' }, el('h2', null, t('tasks.new_title')), closeX),
          formBox, progressBox, reportBox,
          el('div', { class: 'form-actions' }, closeBtn)));
      newDlg.addEventListener('cancel', (e) => { // Esc must not hide a running upload: say how to stop it
        if (uploading) {
          e.preventDefault();
          const stop = document.getElementById('upload-cancel');
          if (stop) stop.focus();
          toast(t('images.upload_running'), 'info', 4000);
        } else if (vid.jobId && vid.polling && vid.pollFails < POLL_LOST) e.preventDefault();
      });
      newDlg.addEventListener('close', () => {
        if (uploading) { newDlg.showModal(); return; } // a second Esc closes a dialog whose first one was refused
        // a video uploaded but never used by a task is removed again
        if (vid.jobId && vid.polling) return;
        resetVideo();
        if (vid.video) discardVideo(true);
        setMode('images');
      });

      function setStage(stage) {
        formBox.hidden = stage !== 'form';
        progressBox.hidden = stage !== 'uploading';
        closeBtn.hidden = stage === 'uploading';
        closeBtn.textContent = t(stage === 'done' ? 'common.close' : 'common.cancel');  // "Fechar" only once something was done
        closeX.hidden = stage === 'uploading';
      }

      function drawStaged() {
        stagedBox.hidden = !staged;
        drop.hidden = !!staged;
        if (!staged) { fill(stagedBox); return; }
        fill(stagedBox,
          icon('images', 28),
          el('strong', null, tn('images.staged', staged.length)),
          el('div', { class: 'drop-actions' },
            el('button', { type: 'button', class: 'btn primary', id: 'new-task-create', onclick: () => startUpload(staged) }, t('tasks.create')),
            el('button', { type: 'button', class: 'btn', onclick: () => { staged = null; drawStaged(); } }, t('images.change_files'))));
      }

      function openNewTask(entries) {
        if (!uploading && !(vid.jobId && vid.polling)) {
          resetVideo();
          if (entries && entries.length) setMode('images');
          fill(reportBox);
          const files = entries ? usable(entries) : [];
          staged = files.length ? files : null;
          drawStaged();
          setStage('form');
          jobSizeInput.removeAttribute('aria-invalid');
        }
        if (!newDlg.open) newDlg.showModal();
        nameInput.focus();
      }

      const imageTabs = tabsView('images', t('step.images'), [
        { id: 'gallery', label: t('images.tab_gallery'), nodes: [el('div', { class: 'grid-head' }, gridTitle, filterBox), grid.node] },
        { id: 'tasks', label: t('images.tab_tasks'), nodes: [tasksBox] },
      ], 'gallery');
      add(node,
        el('h2', { id: 'h-images' }, t('step.images')),
        imageTabs.node,
        newDlg, fileInput, folderInput);

      const reload = () => { loadTasks(); grid.reload(); };
      /* a project without images opens on its tasks, where images are added (unless the person chose a tab before) */
      panel.mount = () => { if (!imageTabs.remembered && !hasImages()) imageTabs.select('tasks'); reload(); };
      panel.refresh = reload;
      panel.show = () => { refreshCounts(); loadTasks(); };  // coming back from another step (an import, an apply, a sync changed the counts)
      /** Step 4 sends the person here to see one part of a dataset version. */
      panel.showDataset = (n, part, fold = null) => {
        filter = { task: null, job: null, dataset: { n, part, fold } };
        imageTabs.select('gallery');
        drawFilter();
        grid.reload();
      };
      /** A deleted dataset version can no longer filter the gallery (every load would answer 404). */
      panel.forgetDataset = (n) => {
        if (!filter.dataset || filter.dataset.n !== n) return;
        filter = { ...filter, dataset: null };
        drawFilter();
        if (panel.mounted) grid.reload();
      };
      return panel;
    }

    // ================================================================ step 2: annotate in an external tool

    // ================================================================ step 2: pre-annotation (a tab of Anotações)
    /* A saved model suggests labels for new images; a person reviews them in CVAT or FiftyOne.
       Three blocks, in the order of the work: pre-annotate, to review, history. */

    // ================================================================ step 2: annotations

    // ================================================================ step 3: curation

    /* whether a comparison found nothing to review: nothing to bring, send, rename or decide */
    const planIsEmpty = (plan) => {
      const n = (x) => (Array.isArray(x) ? x.length : Number(x) || 0);
      return !['pull', 'push', 'add', 'conflicts', 'remove_remote', 'remove_local', 'restore_remote', 'restore_local',
        'tasks_create', 'tasks_new', 'tasks_delete'].some((k) => n(plan[k]));
    };

    // ================================================================ jobs bar
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


    // ---------------------------------------------------------------- step 5: training runs
    function trainingPanel(page) {
      const node = el('section', { class: 'panel', id: 'panel-training', 'aria-labelledby': 'h-training' });
      const panel = { node, mounted: false };
      const listBox = el('div', { class: 'tr-list' });
      const announcer = el('p', { class: 'sr-only', 'aria-live': 'polite' });
      const nf3 = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 3 });
      const MAX_GRID = 50;
      const ACTIVE = (r) => r.status === 'queued' || r.status === 'running';
      let runs = [];
      let catalog = null;
      let timer = null;
      let loaded = false;
      const cards = new Map();   // run number -> {sig, node}
      const lastText = new Map();   // run number -> last announced status text

      // ---- small helpers
      async function ensureCatalog() {
        if (!catalog) catalog = await api('GET', `/training/catalog?type=${encodeURIComponent(ctx.project.task)}`);
        return catalog;
      }
      const modelOf = (id) => ((catalog && catalog.models) || []).find((m) => m.id === id);
      const metricName = (m) => tr(`tr.metric.${m}`, tr(`tr.col.${m}`, String(m).replace(/_/g, ' ')));
      const startName = (s) => tr(`tr.start_${s}`, s);
      const paramName = (n) => tr(`tr.p.${n}`, String(n).replace(/_/g, ' '));
      const choiceName = (c) => tr(`tr.opt.${c}`, String(c));
      const fmtVal = (v) => (typeof v === 'number' ? nf3.format(v) : String(v));
      const plain = (v) => String(v);   // hyperparameters are shown exactly (no rounding of 0.0001)
      const foldCount = (v) => Number((v.options || {}).kfold) || (Array.isArray(v.folds) ? v.folds.length : 0) || (v.sizes && Array.isArray(v.sizes.folds) ? v.sizes.folds.length : 0);
      const foldText = (f) => (f == null ? '' : f === 'all' ? t('tr.fold_all') : t('ds.fold', { n: Number(f) + 1 }));
      const datasetText = (r) => (r.dataset == null ? '' : [t('tr.dataset_short', { n: r.dataset }), foldText(r.fold)].filter(Boolean).join(' · '));
      /* a run whose configuration file is damaged comes without model, dataset or start, with `config_error` instead */
      const configError = (r) => (r.config_error ? el('p', { class: 'tr-fail', title: r.config_error }, icon('error', 16), el('span', null, t('tr.config_error'))) : null);  // the technical detail (a file path) stays in the tooltip

      function fmtDuration(ms) {
        const s = Math.max(0, Math.round(ms / 1000));
        if (s < 60) return `${s}s`;
        const m = Math.floor(s / 60);
        if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
        return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
      }
      function durationOf(r) {
        if (!r.started) return '';
        const end = r.finished ? Date.parse(r.finished) : Date.now();
        return fmtDuration(end - Date.parse(r.started));
      }
      function deviceOf(r) {
        const d = r.device || {};
        /* hardware names (GPU, Apple GPU, CPU) are the same in every language: they stay as they are */
        if (d.cuda) return d.device_name || 'GPU';
        if (d.mps) return 'Apple GPU';
        return d.torch ? 'CPU' : '';
      }
      const childIndex = (r) => (r.current ? (r.children || []).findIndex((c) => c.id === r.current) : -1);
      function phaseText(r) {
        if (r.status === 'queued') return r.queue_position ? t('tr.st_queued_n', { n: r.queue_position }) : t('tr.st_queued');
        const i = childIndex(r);
        if (r.status === 'running' && i >= 0 && r.trainings) {
          const own = r.epoch && r.epochs ? t('tr.ph_epoch', { e: r.epoch, n: r.epochs }) : t('tr.ph_training');
          return `${t('tr.ph_child', { i: i + 1, n: r.trainings })} · ${own}`;
        }
        if (r.status === 'running') {
          if (r.phase === 'environment') return t('tr.ph_environment');
          if (r.phase === 'preparing') return t('tr.ph_preparing');
          if (r.phase === 'evaluating') return t('tr.ph_evaluating');
          return r.epoch && r.epochs ? t('tr.ph_epoch', { e: r.epoch, n: r.epochs }) : t('tr.ph_training');
        }
        const names = { done: t('tr.st_done'), failed: t('tr.st_failed'), cancelled: t('tr.st_cancelled') };
        return names[r.status] || String(r.status);
      }
      function percentOf(r) {
        if (r.status !== 'running' || r.phase !== 'training' || !r.epochs) return null;
        const inEpoch = r.step && r.steps ? r.step / r.steps : 0;
        const done = r.step ? Math.max(0, (r.epoch || 1) - 1) + inEpoch : r.epoch || 0;
        const i = childIndex(r);
        const share = i >= 0 && r.trainings ? (i + done / r.epochs) / r.trainings : done / r.epochs;
        return Math.max(0, Math.min(100, Math.round(share * 100)));
      }
      function badge(r) {
        const glyph = r.status === 'done' ? icon('check', 14) : r.status === 'failed' ? icon('error', 14) : r.status === 'cancelled' ? icon('close', 14) : el('span', { class: 'tr-dot', 'aria-hidden': 'true' });
        return el('span', { class: `tr-badge ${r.status}` }, glyph, phaseText(r));
      }
      /* keeps the keyboard focus on the same button when a card or the detail is redrawn */
      function keepFocus(container, redraw) {
        const a = document.activeElement;
        const act = a && container.contains(a) && a.dataset ? a.dataset.act : null;
        redraw();
        if (act) {
          const again = container.querySelector(`[data-act="${act}"]`);
          if (again) again.focus();
        }
      }
      function announce(r) {
        const text = phaseText(r);
        if (lastText.get(r.number) === text) return;
        const first = !lastText.has(r.number);
        lastText.set(r.number, text);
        if (!first || ACTIVE(r)) announcer.textContent = `${t('tr.run_n', { n: r.number })}: ${text}`;
      }

      // ---- list
      function runCard(r) {
        const m = modelOf(r.model);
        const metric = (m && m.default_metric) || '';
        const pct = percentOf(r);
        const id = `run-${r.number}`;
        const best = r.best && r.best.value != null ? `${metricName(metric)} ${fmtVal(r.best.value)}` : '';
        const meta = r.config_error ? [] : [datasetText(r), startName(r.start), r.mode === 'grid' ? t('tr.mode_grid') : null, r.trainings ? tn('tr.trainings', r.trainings) : null].filter(Boolean);
        const facts = [durationOf(r), deviceOf(r)].filter(Boolean);
        const act = (name, label, fn, cls = '') => el('button', { type: 'button', class: `btn small ${cls}`, 'data-act': name, onclick: fn }, label);
        const noWeights = r.status === 'done' && !r.model_number && (r.has_weights === false || ctx.weightless.has(r.number));
        const saveButton = act('save', t('md.save'), () => saveModel(r), 'primary');
        if (noWeights) {
          saveButton.disabled = true;
          saveButton.setAttribute('aria-describedby', `${id}-noweights`);
        }
        return el('li', { class: `tr-run card st-${r.status}`, id, 'aria-labelledby': `${id}-t` },
          el('div', { class: 'tr-run-head' },
            el('h3', { id: `${id}-t` }, `#${r.number}`),
            el('span', { class: 'tr-model' }, m ? m.label : r.model),
            badge(r)),
          el('p', { class: 'muted tr-meta' }, meta.join(' · ')),
          pct != null ? el('progress', { class: 'tr-progress', max: '100', value: String(pct), 'aria-label': t('common.progress') }) : null,
          r.status === 'running' && r.message ? el('p', { class: 'muted tr-msg' }, r.message) : null,
          configError(r),
          r.status === 'failed' && r.error ? el('p', { class: 'tr-fail' }, icon('error', 16), el('span', null, r.error)) : null,
          noWeights ? el('p', { class: 'muted tr-msg tr-noweights', id: `${id}-noweights` }, icon('info', 14), t('md.no_weights')) : null,
          best || facts.length
            ? el('p', { class: 'tr-facts' }, best ? el('strong', null, best) : null, best && facts.length ? ' · ' : null, el('span', { class: 'muted' }, facts.join(' · ')))
            : null,
          el('div', { class: 'tr-actions' },
            act('detail', t('tr.details'), () => openDetail(r.number)),
            r.status === 'done' && r.model_number
              ? el('a', { class: 'btn small tr-saved', href: '#models', 'data-act': 'model' }, icon('star', 14), t('md.saved_as', { n: r.model_number }))
              : null,
            r.status === 'done' && !r.model_number ? saveButton : null,
            ACTIVE(r) ? act('cancel', t('tr.cancel'), () => cancelRun(r), 'danger-outline') : null,
            r.config_error ? null : act('again', t('tr.reproduce'), once(() => reproduceRun(r))),
            ACTIVE(r) ? null : el('button', { type: 'button', class: 'icon-btn danger tr-del', 'data-act': 'delete', 'aria-label': t('tr.delete_aria', { n: r.number }), title: t('tr.delete'), onclick: () => removeRun(r) }, icon('trash', 18))));
      }

      function draw() {
        const ul = listBox.querySelector('ul.tr-cards');
        newBtn.hidden = !runs.length;
        if (!runs.length) {
          cards.clear();
          fill(listBox, emptyState('chart', t('tr.empty'),
            el('button', { type: 'button', class: 'btn primary', onclick: once(() => openNew(newBtn)) }, icon('plus', 16), t('tr.new'))));
          return;
        }
        const list = ul || el('ul', { class: 'tr-cards' });
        const focused = document.activeElement;
        const act = focused && list.contains(focused) && focused.dataset ? focused.dataset.act : null;
        const owner = act ? [...list.children].findIndex((li) => li.contains(focused)) : -1;
        const ownerNumber = owner >= 0 ? Number(list.children[owner].id.replace('run-', '')) : null;
        const keep = new Set();
        const nodes = runs.map((r) => {
          keep.add(r.number);
          const sig = JSON.stringify(r) + (catalog ? '1' : '0') + (ctx.weightless.has(r.number) ? 'w' : '');
          const old = cards.get(r.number);
          if (old && old.sig === sig) return old.node;
          const fresh = runCard(r);
          cards.set(r.number, { sig, node: fresh });
          return fresh;
        });
        for (const k of [...cards.keys()]) if (!keep.has(k)) cards.delete(k);
        const same = nodes.length === list.children.length && nodes.every((n, i) => list.children[i] === n);
        if (!same) fill(list, ...nodes);
        if (!ul) fill(listBox, list);
        if (act && ownerNumber != null && cards.has(ownerNumber) && !list.contains(document.activeElement)) {
          const again = cards.get(ownerNumber).node.querySelector(`[data-act="${act}"]`);
          if (again) again.focus();
        }
      }

      function schedule() {
        clearTimeout(timer);
        if (runs.some(ACTIVE)) timer = setTimeout(load, 2000);
      }

      async function load(quiet = false) {
        try {
          await ensureCatalog().catch(() => null);
          const d = await api('GET', `/projects/${projectId}/runs`);
          runs = d.runs || [];
          loaded = true;
        } catch (err) {
          if (!quiet && !loaded) toastError(err);
          schedule();
          return;
        }
        ctx.runCount = runs.length;
        ctx.runActive = runs.filter(ACTIVE).length;
        runs.forEach(announce);
        if (node.isConnected) draw();
        renderStepper();
        schedule();
      }

      async function cancelRun(r) {
        const ok = await confirmDialog({ title: t('tr.cancel_title', { n: r.number }), text: t('tr.cancel_text'), confirmLabel: t('tr.cancel') });
        if (!ok) return;
        try {
          await api('POST', `/projects/${projectId}/runs/${r.number}/cancel`);
          toast(t('tr.cancelled', { n: r.number }), 'success', 3000);
        } catch (err) { toastError(err); }
        await load(true);
      }
      async function reproduceRun(r) {
        try {
          const made = await api('POST', `/projects/${projectId}/runs/${r.number}/reproduce`);
          toast(t('tr.created', { n: made.number }), 'success', 4000);
        } catch (err) { toastError(err); }
        await load(true);
      }
      function saveModel(r) {
        const spec = modelOf(r.model);
        const canOnnx = !!spec && ['classification', 'semantic'].includes(spec.task_kind);
        const onnx = el('input', { type: 'checkbox', id: 'md-onnx', disabled: !canOnnx });
        const formats = el('fieldset', { class: 'field md-formats' },
          el('legend', null, t('md.formats')),
          el('p', { class: 'md-always' }, icon('check', 14), el('span', null, t('md.format_pt'))),
          el('label', { class: 'ds-check', for: 'md-onnx' }, onnx, el('span', null, t('md.format_onnx'))),
          el('p', { class: 'hint' }, canOnnx ? t('md.format_onnx_help') : t('md.onnx_later')));
        const note = el('textarea', { id: 'md-note', rows: '2', maxlength: '500', placeholder: t('md.note_placeholder') });
        const save = el('button', { type: 'submit', class: 'btn primary' }, icon('star', 16), t('md.save'));
        const dlg = el('dialog', { class: 'dialog', 'aria-labelledby': 'md-save-t' },
          el('form', { class: 'dialog-body form', method: 'dialog' },
            el('h2', { id: 'md-save-t' }, t('md.save_title', { n: r.number })),
            el('p', { class: 'muted' }, t('md.save_help')),
            formats,
            el('div', { class: 'field' }, el('label', { for: 'md-note' }, t('md.note'), el('span', { class: 'optional' }, ` · ${t('step.optional')}`)), note),
            el('div', { class: 'form-actions' }, save, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel')))));
        dlg.querySelector('form').addEventListener('submit', async (e) => {
          e.preventDefault();
          save.disabled = true;
          try {
            const m = await api('POST', `/projects/${projectId}/models`, { json: { run: r.number, note: note.value } });
            let exportFailed = false;
            if (onnx.checked) {
              try { await api('POST', `/projects/${projectId}/models/${m.number}/export`, { json: { format: 'onnx' } }); } catch (err) { exportFailed = true; toastError(err); }
            }
            dlg.close();
            toast(t(onnx.checked && !exportFailed ? 'md.saved_onnx' : 'md.saved', { n: m.number }), 'success', 4000);
            await load(true);
            ctx.panels.models.count();
          } catch (err) {
            toastError(err);
            if (err.code !== 'run_no_weights') { save.disabled = false; return; }
            ctx.weightless.add(r.number);  // the weights are gone: the card says so instead of offering the save again
            dlg.close();
            await load(true);
          }
        });
        dlg.addEventListener('close', () => dlg.remove());
        document.body.appendChild(dlg);
        openModal(dlg);
        note.focus();
      }

      async function removeRun(r) {
        const ok = await confirmDialog({ title: t('tr.delete_title', { n: r.number }), text: t('tr.delete_text'), confirmLabel: t('tr.delete') });
        if (!ok) return;
        try {
          await api('DELETE', `/projects/${projectId}/runs/${r.number}`, { json: { confirm: true } });
          toast(t('tr.deleted', { n: r.number }), 'success', 3000);
        } catch (err) { toastError(err); }
        await load(true);
      }

      // ---------------------------------------------------------------- new run dialog
      async function openNew(trigger, version = null) {
        let cat;
        let versions;
        let augCat;
        try {
          cat = await ensureCatalog();
          augCat = await api('GET', `/training/augmentations?type=${encodeURIComponent(ctx.project.task)}`);
          versions = ((await api('GET', `/projects/${projectId}/datasets`)).versions || []).filter((v) => !v.incomplete).sort((a, b) => b.version - a.version);
        } catch (err) { toastError(err); return; }
        const models = (cat.models || []).filter((m) => m.family !== 'ultralytics');   // YOLO: not trainable yet
        if (!versions.length || !models.length) { toast(t('tr.no_options'), 'error'); return; }

        const st = { dataset: (versions.find((v) => v.version === version) || versions[0]).version, fold: 'all', model: models[0].id, start: 'transfer', mode: 'single', metric: '', vals: {}, sel: {}, aug: {}, augLevel: null, augBad: new Set() };
        const versionOf = () => versions.find((v) => v.version === st.dataset);
        const modelNow = () => models.find((m) => m.id === st.model);
        let active = [];       // parameter specs shown now
        let reads = {};        // name -> list of values
        const errEls = {};

        const sel = (id, ...opts) => el('select', { id, class: 'tr-select' }, ...opts);
        const create = el('button', { type: 'submit', class: 'btn primary' }, t('tr.start_run'));
        const formError = el('p', { class: 'tr-form-error', role: 'alert', hidden: true });

        // data
        const datasetSel = sel('tr-dataset', ...versions.map((v) => el('option', { value: String(v.version) },
          [`v${v.version}`, fmtDate(v.created), foldCount(v) ? t('ds.kfold_short', { k: foldCount(v) }) : null, v.note && String(v.note).trim().toLowerCase() !== `v${v.version}` ? String(v.note).slice(0, 40) : null].filter(Boolean).join(' · '))));
        datasetSel.value = String(st.dataset);
        const foldSel = sel('tr-fold');
        const foldField = el('div', { class: 'field', hidden: true }, el('label', { for: 'tr-fold' }, t('tr.fold')), foldSel);
        function drawFolds() {
          const k = foldCount(versionOf());
          foldField.hidden = !k;
          fill(foldSel, el('option', { value: 'all' }, t('tr.fold_all')), ...Array.from({ length: k }, (_, i) => el('option', { value: String(i) }, t('ds.fold', { n: i + 1 }))));
          st.fold = 'all';
        }
        /* a version freezes its labels: say so when the labels changed afterwards (training is not blocked) */
        const staleNote = el('p', { class: 'ds-stale-note', id: 'tr-dataset-stale', role: 'status', hidden: true });
        function drawStale() {
          const n = staleCount(versionOf());
          staleNote.hidden = !n;
          fill(staleNote, ...(n ? [icon('warning', 14), tn('ds.stale_train', n, { n: num(n), version: st.dataset })] : []));
        }
        datasetSel.setAttribute('aria-describedby', 'tr-dataset-stale');
        datasetSel.addEventListener('change', () => { st.dataset = Number(datasetSel.value); drawFolds(); drawStale(); changed(); });
        foldSel.addEventListener('change', () => { st.fold = foldSel.value; changed(); });

        // model
        const modelSel = sel('tr-model');
        const modelNote = el('p', { class: 'hint', id: 'tr-model-note' });
        /* what a change of model did to the hyperparameters, in one line; read out by screen readers */
        const modelChanges = el('p', { class: 'hint tr-model-changes', id: 'tr-model-changes', role: 'status' });
        let shownModel = null;   // the model the form holds values of
        const groups = new Map();
        const kinds = new Set(models.map((m) => m.task_kind));
        for (const m of models) {
          const key = `${m.task_kind}|${m.family}`;
          if (!groups.has(key)) groups.set(key, { label: [kinds.size > 1 ? tr(`tr.kind.${m.task_kind}`, m.task_kind) : null, tr(`tr.family.${m.family}`, m.family)].filter(Boolean).join(' · '), items: [] });
          groups.get(key).items.push(m);
        }
        for (const g of groups.values()) add(modelSel, el('optgroup', { label: g.label }, ...g.items.map((m) => el('option', { value: m.id }, m.label))));
        modelSel.setAttribute('aria-describedby', 'tr-model-note tr-model-changes');

        // start
        const startHelp = el('p', { class: 'hint', 'aria-live': 'polite' });
        const startInputs = {};
        const startLabels = {};
        const startBox = el('div', { class: 'seg tr-seg', role: 'radiogroup', 'aria-label': t('tr.start') },
          ...['transfer', 'finetune', 'scratch'].map((s) => {
            const input = el('input', { type: 'radio', name: 'tr-start', id: `tr-start-${s}`, value: s, class: 'sr-only', onchange: () => { st.start = s; drawStart(); drawParams(); } });
            startInputs[s] = input;
            startLabels[s] = el('label', { for: input.id, class: 'seg-opt' }, input, startName(s));
            return startLabels[s];
          }));
        function drawStart() {
          const m = modelNow();
          if (!m.supports_pretrained) st.start = 'scratch';
          for (const [s, input] of Object.entries(startInputs)) {
            input.checked = st.start === s;
            startLabels[s].hidden = s !== 'scratch' && !m.supports_pretrained;   // not offered, rather than greyed out
          }
          startHelp.textContent = m.supports_pretrained ? t(`tr.start_${st.start}_help`) : t('tr.start_scratch_only');
        }

        // mode
        const modeBox = el('div', { class: 'seg tr-seg', role: 'radiogroup', 'aria-label': t('tr.mode') },
          ...['single', 'grid'].map((m) => {
            const input = el('input', { type: 'radio', name: 'tr-mode', id: `tr-mode-${m}`, value: m, class: 'sr-only', checked: st.mode === m,
              onchange: () => { setMode(m); } });
            return el('label', { for: input.id, class: 'seg-opt' }, input, t(`tr.mode_${m}`));
          }));
        const modeHelp = el('p', { class: 'hint' });
        const summary = el('p', { class: 'tr-combos', role: 'status' });

        // selection metric
        const metricSel = sel('tr-metric');

        // device (#320): not part of the configuration; the list comes from the training environment
        const deviceName = (d) => (d.startsWith('cuda:') ? t('tr.device_gpu', { n: d.slice(5) }) : t(`tr.device_${d}`));
        const deviceSel = sel('tr-device', ...['auto', 'cpu'].map((d) => el('option', { value: d }, deviceName(d))));
        api('GET', '/training/environment').then((env) => fill(deviceSel, ...(env.devices || ['auto', 'cpu']).map((d) => el('option', { value: d }, deviceName(d))))).catch(() => {});

        // parameters
        const paramsBox = el('div', { class: 'tr-params' });
        /* the settings stay closed: the line on the bar says what differs from the defaults */
        const advLine = el('span', { class: 'muted tr-adv-line' });
        const advBox = el('details', { class: 'tr-adv' }, el('summary', null, el('strong', null, t('tr.params')), advLine));
        const moreBox = el('details', { class: 'tr-more' }, el('summary', null, t('tr.more_params')));
        const moreInner = el('div', { class: 'tr-params' });
        add(moreBox, moreInner);

        function readParam(spec) {
          const grid = st.mode === 'grid';
          const name = spec.name;
          if (spec.kind === 'bool') return { list: [!!st.vals[name]] };
          if (spec.kind === 'choice') {
            const list = grid ? st.sel[name] : st.sel[name].slice(0, 1);
            return list.length ? { list } : { error: t('tr.err_required') };
          }
          const tokens = String(st.vals[name] == null ? '' : st.vals[name]).split(grid ? ',' : '\u0000').map((s) => s.trim()).filter(Boolean);
          if (!tokens.length) return { error: t('tr.err_required') };
          const list = [];
          for (const tok of tokens) {
            const n = Number(tok);
            if (!Number.isFinite(n)) return { error: t('tr.err_number', { v: tok }) };
            if (spec.kind === 'int' && !Number.isInteger(n)) return { error: t('tr.err_int', { v: tok }) };
            if ((spec.min != null && n < spec.min) || (spec.max != null && n > spec.max)) return { error: t('tr.err_range', { min: plain(spec.min), max: plain(spec.max) }) };
            list.push(n);
          }
          if (new Set(list).size !== list.length) return { error: t('tr.err_dup') };
          return { list };
        }

        function paramField(spec) {
          const grid = st.mode === 'grid';
          const name = spec.name;
          const id = `tr-p-${name}`;
          const err = el('p', { class: 'tr-err', id: `${id}-err` });
          errEls[name] = err;
          const hint = spec.help ? el('p', { class: 'hint', id: `${id}-hint` }, tr(`tr.ph.${name}`, spec.help)) : null;
          const describedby = `${hint ? `${id}-hint ` : ''}${id}-err`;
          let control;
          if (spec.kind === 'bool') {
            control = el('label', { for: id, class: 'ds-check tr-switch' },
              el('input', { type: 'checkbox', id, role: 'switch', checked: !!st.vals[name], 'aria-describedby': describedby, onchange: (e) => { st.vals[name] = e.target.checked; changed(); } }),
              el('span', null, paramName(name)));
            return el('div', { class: 'field tr-param' }, control, hint, err);
          }
          if (spec.kind === 'choice') {
            if (grid) {
              control = el('fieldset', { class: 'field tr-param' }, el('legend', null, paramName(name)),
                el('div', { class: 'tr-checks' }, ...spec.choices.map((c, i) => el('label', { class: 'ds-check', for: `${id}-${i}` },
                  el('input', { type: 'checkbox', id: `${id}-${i}`, value: c, checked: st.sel[name].includes(c),
                    onchange: (e) => { st.sel[name] = spec.choices.filter((x) => (x === c ? e.target.checked : st.sel[name].includes(x))); changed(); } }),
                  el('span', null, choiceName(c))))), hint, err);
              return control;
            }
            const select = el('select', { id, class: 'tr-select', 'aria-describedby': describedby, onchange: (e) => { st.sel[name] = [e.target.value]; changed(); } },
              ...spec.choices.map((c) => el('option', { value: c, selected: st.sel[name][0] === c }, choiceName(c))));
            return el('div', { class: 'field tr-param' }, el('label', { for: id }, paramName(name)), select, hint, err);
          }
          const input = el('input', {
            id, type: grid ? 'text' : 'number', inputmode: spec.kind === 'int' ? 'numeric' : 'decimal', autocomplete: 'off', value: String(st.vals[name] == null ? '' : st.vals[name]),
            min: !grid && spec.min != null ? String(spec.min) : null, max: !grid && spec.max != null ? String(spec.max) : null, step: !grid ? (spec.kind === 'int' ? '1' : 'any') : null,
            placeholder: grid ? `${spec.default}, …` : null, 'aria-describedby': describedby,
          });
          input.addEventListener('input', () => { st.vals[name] = input.value; changed(); });
          return el('div', { class: 'field tr-param' }, el('label', { for: id }, paramName(name)), input, hint, err);
        }

        /* Augmentation techniques of the model's kind of task. A preset level fills the list; ticking a technique
           or changing a setting makes the level "custom", whose list goes in the request. */
        const augBox = el('fieldset', { class: 'field tr-aug', id: 'tr-aug' });
        const augKind = () => (augCat.kinds || {})[modelNow().task_kind] || { techniques: [], presets: {} };
        const augLevels = () => st.sel.augmentation || [];
        function loadPreset(level) {
          st.aug = {};
          st.augBad.clear();
          for (const entry of augKind().presets[level] || []) st.aug[entry.name] = { ...entry.params };
        }
        function augEntries() {
          return augKind().techniques.filter((tq) => st.aug[tq.name]).map((tq) => ({ name: tq.name, params: { ...st.aug[tq.name] } }));
        }
        function dropStaleErrors() {
          for (const key of [...st.augBad]) if (!st.aug[key.split('.')[0]]) st.augBad.delete(key);
        }
        function toCustom() {
          st.sel.augmentation = ['custom'];
          st.augLevel = 'custom';
          const select = document.getElementById('tr-p-augmentation');
          if (select && select.tagName === 'SELECT') select.value = 'custom';
        }
        function toggleAug(tq, on) {
          if (on) {
            st.aug[tq.name] = Object.fromEntries(tq.params.map((p) => [p.name, p.default]));
            if (tq.needs_flip_pairs) st.aug[tq.name].flip_pairs = [];
            for (const other of tq.conflicts) delete st.aug[other];
          } else {
            delete st.aug[tq.name];
          }
          dropStaleErrors();
          toCustom();
          drawAug();
          changed();
          const box = document.getElementById(`tr-aug-${tq.name}`);
          if (box) box.focus();   // the list was drawn again: keep the keyboard where it was
        }
        /* a number setting of a chosen technique; an invalid value blocks "Start" until fixed */
        function augParam(tq, spec) {
          const id = `tr-aug-${tq.name}-${spec.name}`;
          const key = `${tq.name}.${spec.name}`;
          const input = el('input', { id, type: 'number', inputmode: 'decimal', autocomplete: 'off', class: 'tr-aug-num', value: String(st.aug[tq.name][spec.name]),
            min: String(spec.min), max: String(spec.max), step: spec.kind === 'int' ? '1' : 'any', title: tr(`tr.augph.${spec.name}`, spec.help), 'aria-invalid': st.augBad.has(key) ? 'true' : null });
          input.addEventListener('input', () => {
            const n = Number(input.value);
            const ok = input.value.trim() !== '' && Number.isFinite(n) && n >= spec.min && n <= spec.max && (spec.kind !== 'int' || Number.isInteger(n));
            if (ok) { st.aug[tq.name][spec.name] = n; st.augBad.delete(key); input.removeAttribute('aria-invalid'); } else { st.augBad.add(key); input.setAttribute('aria-invalid', 'true'); }
            toCustom();
            changed();
          });
          return el('label', { for: id, class: 'tr-aug-set-item' }, el('span', null, tr(`tr.augp.${spec.name}`, spec.name)), input);
        }
        /* keypoints: the left/right pairs a horizontal flip swaps, written "0-1, 2-3" (points numbered from 0) */
        function augPairs(tq) {
          const id = `tr-aug-${tq.name}-pairs`;
          const key = `${tq.name}.flip_pairs`;
          const input = el('input', { id, type: 'text', autocomplete: 'off', class: 'tr-aug-pairs', placeholder: '0-1, 2-3', 'aria-describedby': `${id}-hint`,
            value: (st.aug[tq.name].flip_pairs || []).map((pair) => pair.join('-')).join(', '), 'aria-invalid': st.augBad.has(key) ? 'true' : null });
          input.addEventListener('input', () => {
            const pairs = input.value.split(',').map((s) => s.trim()).filter(Boolean).map((s) => s.split('-').map((x) => x.trim()));
            const flat = pairs.flat();
            const ok = pairs.every((pair) => pair.length === 2 && pair.every((x) => /^\d+$/.test(x)) && pair[0] !== pair[1]) && new Set(flat).size === flat.length;
            if (ok) { st.aug[tq.name].flip_pairs = pairs.map((pair) => pair.map(Number)); st.augBad.delete(key); input.removeAttribute('aria-invalid'); } else { st.augBad.add(key); input.setAttribute('aria-invalid', 'true'); }
            toCustom();
            changed();
          });
          return el('div', { class: 'tr-aug-set-item wide' }, el('label', { for: id }, t('tr.aug_pairs')), input, el('p', { class: 'hint', id: `${id}-hint` }, t('tr.aug_pairs_help')));
        }
        function augItem(tq) {
          const on = !!st.aug[tq.name];
          const id = `tr-aug-${tq.name}`;
          const settings = on ? tq.params.map((spec) => augParam(tq, spec)) : [];
          if (on && tq.needs_flip_pairs) settings.push(augPairs(tq));
          return el('div', { class: `tr-aug-item${on ? ' on' : ''}` },
            el('label', { for: id, class: 'ds-check' },
              el('input', { type: 'checkbox', id, checked: on, 'aria-describedby': `${id}-hint`, onchange: (e) => toggleAug(tq, e.target.checked) }),
              el('span', null, tr(`tr.aug.${tq.name}`, tq.name))),
            el('p', { class: 'hint', id: `${id}-hint` }, tr(`tr.augh.${tq.name}`, tq.help)),
            settings.length ? el('div', { class: 'tr-aug-set' }, ...settings) : null);
        }
        function drawAug() {
          const kind = augKind();
          augBox.hidden = !kind.techniques.length || augLevels().length !== 1;
          if (augBox.hidden) { fill(augBox); return; }
          fill(augBox, el('legend', null, t('tr.aug_title')),
            el('p', { class: 'hint' }, augEntries().length ? t('tr.aug_help') : t('tr.aug_none')),
            el('div', { class: 'tr-aug-list' }, ...kind.techniques.map(augItem)));
        }
        /* a preset level (re)fills the list for the model now shown; a custom list keeps what this model offers */
        function syncAug() {
          const level = augLevels().length === 1 ? augLevels()[0] : null;
          if (level && level !== 'custom') loadPreset(level);
          if (level === 'custom') {
            const offered = new Set(augKind().techniques.map((tq) => tq.name));
            for (const name of Object.keys(st.aug)) if (!offered.has(name)) delete st.aug[name];
            dropStaleErrors();
          }
          st.augLevel = level;
          drawAug();
        }

        function setDefault(spec) {
          if (spec.kind === 'choice') st.sel[spec.name] = [spec.default];
          else st.vals[spec.name] = spec.kind === 'bool' ? !!spec.default : String(spec.default);
        }
        /* a valid value that is not the default of `spec`: the person chose it */
        function edited(spec) {
          const r = readParam(spec);
          return !r.error && r.list.join() !== String([spec.default]);
        }
        /* Values for the model now shown, coming from `prev` (null on opening): a value the person chose is
           kept when the new model has the parameter and accepts it; the rest takes the new model's default.
           Returns the lines that say what changed. */
        function carryValues(prev) {
          const m = modelNow();
          const notes = [];
          const kept = [];
          for (const spec of m.params) {
            if (spec.name === 'start') continue;
            const old = prev && prev.params.find((p) => p.name === spec.name);
            const name = paramName(spec.name);
            if (!old) {
              setDefault(spec);
              if (prev) notes.push(t('tr.chg_new', { p: name }));
              continue;
            }
            if (edited(old)) {
              const before = readParam(old).list;
              if (spec.kind === 'choice') st.sel[spec.name] = st.sel[spec.name].filter((c) => spec.choices.includes(c));
              if (!readParam(spec).error) { kept.push(name); continue; }
              setDefault(spec);
              notes.push(t('tr.chg_reset', { p: name, v: valueText(spec, spec.default), old: before.map((v) => valueText(spec, v)).join(', ') }));
              continue;
            }
            setDefault(spec);
            if (String(old.default) !== String(spec.default)) notes.push(t('tr.chg_default', { p: name, v: valueText(spec, spec.default) }));
          }
          if (prev) {
            for (const old of prev.params) {
              if (old.name !== 'start' && !m.params.some((p) => p.name === old.name)) notes.push(t('tr.chg_gone', { p: paramName(old.name) }));
            }
          }
          if (kept.length) notes.push(t('tr.chg_kept', { list: kept.join(', ') }));
          return notes;
        }
        function drawParams() {
          const m = modelNow();
          active = m.params.filter((p) => p.name !== 'start' && (p.name !== 'backbone_lr_factor' || st.start === 'finetune'));
          for (const k of Object.keys(errEls)) delete errEls[k];
          const common = active.filter((p) => p.group !== 'model');
          const own = active.filter((p) => p.group === 'model');
          fill(paramsBox, ...common.flatMap((p) => (p.name === 'augmentation' ? [paramField(p), augBox] : [paramField(p)])));
          fill(moreInner, ...own.map(paramField));
          moreBox.hidden = !own.length;
          syncAug();
          changed();
        }
        function setMode(mode) {
          st.mode = mode;
          if (mode === 'single') {
            for (const spec of active) {
              if (spec.kind === 'int' || spec.kind === 'float') st.vals[spec.name] = String(st.vals[spec.name]).split(',')[0].trim();
              if (spec.kind === 'choice') st.sel[spec.name] = st.sel[spec.name].slice(0, 1);
            }
          }
          modeHelp.textContent = t(mode === 'grid' ? 'tr.mode_grid_help' : 'tr.mode_single_help');
          if (mode === 'grid') advBox.open = true;  // a grid search is about the values: show them
          drawParams();
        }
        function drawModel() {
          const m = modelNow();
          modelNote.textContent = [m.notes, m.packages && m.packages.length ? t('tr.packages_note', { list: m.packages.join(', ') }) : ''].filter(Boolean).join(' ');
          const notes = carryValues(shownModel);
          if (!m.metrics.includes(st.metric)) {   // a metric the person chose stays while the new model has it
            if (shownModel && st.metric) notes.push(t('tr.chg_default', { p: t('tr.select_metric'), v: metricName(m.default_metric) }));
            st.metric = m.default_metric;
          }
          fill(metricSel, ...m.metrics.map((x) => el('option', { value: x, selected: x === st.metric }, metricName(x))));
          const start = st.start;
          drawStart();
          if (shownModel && st.start !== start) notes.push(t('tr.chg_default', { p: t('tr.start'), v: startName(st.start) }));
          if (shownModel) {
            const shown = notes.length > 4 ? [...notes.slice(0, 4), t('tr.params_more', { n: notes.length - 4 })] : notes;
            modelChanges.textContent = notes.length ? `${t('tr.chg_title', { model: m.label })} ${shown.join(' · ')}` : t('tr.chg_none', { model: m.label });
          }
          shownModel = m;
          setMode(st.mode);
        }
        modelSel.addEventListener('change', () => { st.model = modelSel.value; drawModel(); });
        metricSel.addEventListener('change', () => { st.metric = metricSel.value; });

        function changed() {
          let combos = 1;
          let bad = 0;
          reads = {};
          for (const spec of active) {
            const r = readParam(spec);
            const box = errEls[spec.name];
            const input = document.getElementById(`tr-p-${spec.name}`);
            if (box) box.textContent = r.error || '';
            if (input) { if (r.error) input.setAttribute('aria-invalid', 'true'); else input.removeAttribute('aria-invalid'); }
            if (r.error) { bad += 1; continue; }
            reads[spec.name] = r.list;
            combos *= r.list.length;
          }
          const levels = reads.augmentation || [];
          if (levels.length > 1 && levels.includes('custom')) {   // a custom list cannot be one value of a grid
            if (errEls.augmentation) errEls.augmentation.textContent = t('tr.aug_custom_alone');
            bad += 1;
          }
          if ((levels.length === 1 ? levels[0] : null) !== st.augLevel) syncAug();
          if (levels.length === 1 && levels[0] === 'custom') bad += st.augBad.size;
          const grid = st.mode === 'grid';
          const folds = st.fold === 'all' ? Math.max(1, foldCount(versionOf())) : 1;   // a hold-out version has 0 folds: one run per combination
          const total = combos * folds;
          summary.hidden = !grid && folds < 2;
          summary.classList.remove('bad', 'ok');
          let blocked = bad > 0;
          if (grid && !bad) {
            if (combos < 2) { summary.textContent = t('tr.grid_need'); summary.classList.add('bad'); blocked = true; }
            else if (total > MAX_GRID) { summary.textContent = t('tr.trainings_max', { total, n: MAX_GRID }); summary.classList.add('bad'); blocked = true; }
            else { summary.textContent = folds > 1 ? t('tr.combos_folds', { c: tn('tr.combos', combos), k: folds, total }) : tn('tr.combos', combos); summary.classList.add('ok'); }
          } else if (grid) summary.textContent = '';
          else if (folds > 1) { summary.textContent = t('tr.folds_runs', { k: folds }); summary.classList.add('ok'); }
          create.disabled = blocked;
          formError.hidden = true;
          if (bad) advBox.open = true;  // the field with the problem must be in view
          advLine.textContent = changedLine();
        }

        /* one value of a parameter as the person reads it: exact, with the decimal sign of the fields (0,0001) */
        const exact = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 20, useGrouping: false });
        function valueText(spec, v) {
          if (spec.kind === 'choice') return choiceName(v);
          if (typeof v === 'boolean') return v ? '✓' : '✗';
          return exact.format(Number(v));
        }
        /* "epochs 20 · batch size 8": only what is not the default; the grid lists show as they are */
        function changedLine() {
          const diffs = active.filter((spec) => reads[spec.name] && reads[spec.name].join() !== String([spec.default]))
            .map((spec) => `${paramName(spec.name)} ${reads[spec.name].map((v) => valueText(spec, v)).join(', ')}`);
          if (!diffs.length) return t('tr.params_default');
          return diffs.length > 3 ? `${diffs.slice(0, 3).join(' · ')} · ${t('tr.params_more', { n: diffs.length - 3 })}` : diffs.join(' · ');
        }

        function requestBody() {
          const params = {};
          const grid = {};
          for (const spec of active) {
            const list = reads[spec.name];
            if (st.mode === 'grid' && list.length > 1) grid[spec.name] = list; else params[spec.name] = list[0];
          }
          const folds = foldCount(versionOf());
          const body = { dataset: st.dataset, fold: folds ? (st.fold === 'all' ? 'all' : Number(st.fold)) : null, model: st.model, start: st.start, mode: st.mode, metric: st.metric, params, grid, device: deviceSel.value };
          if (params.augmentation === 'custom') body.augmentations = augEntries();
          return body;
        }

        const form = el('form', { class: 'dialog-body form tr-form', method: 'dialog', novalidate: true },
          el('h2', null, t('tr.new_title')),
          el('div', { class: 'tr-form-grid' },
            el('div', { class: 'tr-col' },
              el('div', { class: 'field' }, el('label', { for: 'tr-dataset' }, t('tr.dataset')), datasetSel, staleNote),
              foldField,
              el('div', { class: 'field' }, el('label', { for: 'tr-model' }, t('tr.model')), modelSel, modelNote, modelChanges),
              el('fieldset', { class: 'field' }, el('legend', null, t('tr.start')), startBox, startHelp)),
            el('div', { class: 'tr-col' },
              el('fieldset', { class: 'field' }, el('legend', null, t('tr.mode')), modeBox, modeHelp, summary),
              el('div', { class: 'field' }, el('label', { for: 'tr-metric' }, t('tr.select_metric')), metricSel, el('p', { class: 'hint' }, t('tr.select_metric_help'))),
              el('div', { class: 'field' }, el('label', { for: 'tr-device' }, t('tr.device_label')), deviceSel, el('p', { class: 'hint' }, t('tr.device_help'))))),
          el('div', { class: 'tr-adv-wrap' }, advBox),
          formError,
          el('div', { class: 'form-actions' }, create, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel'))));
        add(advBox, paramsBox, moreBox);
        const dlg = el('dialog', { class: 'dialog wide tr-dialog', 'aria-label': t('tr.new_title') }, form);

        form.addEventListener('submit', async (e) => {
          e.preventDefault();
          if (create.disabled) return;
          create.disabled = true;
          formError.hidden = true;
          try {
            const made = await api('POST', `/projects/${projectId}/runs`, { json: requestBody() });
            dlg.close();
            toast(t('tr.created', { n: made.number }), 'success', 4000);
            await load(true);
          } catch (err) {
            formError.textContent = err.message || String(err);
            formError.hidden = false;
            create.disabled = false;
            formError.scrollIntoView({ block: 'nearest' });
          }
        });
        dlg.addEventListener('close', () => { dlg.remove(); if (trigger && trigger.isConnected) trigger.focus(); });
        drawFolds();
        drawStale();
        drawModel();
        document.body.appendChild(dlg);
        openModal(dlg);
        datasetSel.focus();
      }

      // ---------------------------------------------------------------- charts and tables (no libraries)
      const SVGNS = 'http://www.w3.org/2000/svg';
      function svg(tag, attrs, ...kids) {
        const n = document.createElementNS(SVGNS, tag);
        for (const [k, v] of Object.entries(attrs || {})) if (v != null) n.setAttribute(k, String(v));
        for (const k of kids) if (k) n.appendChild(typeof k === 'string' ? document.createTextNode(k) : k);
        return n;
      }
      /* series: [{name, cls, points: [[epoch, value], ...]}]; bestEpoch draws a vertical marker */
      function lineChart(title, series, bestEpoch) {
        const W = 440;
        const H = 210;
        const m = { l: 46, r: 14, t: 12, b: 28 };
        const all = series.flatMap((s) => s.points).filter(([, y]) => Number.isFinite(y));
        if (!all.length) return null;
        const xs = all.map((p) => p[0]);
        let x0 = Math.min(...xs);
        let x1 = Math.max(...xs);
        if (x0 === x1) { x0 -= 1; x1 += 1; }
        let y0 = Math.min(...all.map((p) => p[1]));
        let y1 = Math.max(...all.map((p) => p[1]));
        if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
        const pad = (y1 - y0) * 0.08;
        y0 = y0 >= 0 ? Math.max(0, y0 - pad) : y0 - pad;   // metrics and losses are never negative
        y1 += pad;
        const X = (x) => m.l + ((x - x0) / (x1 - x0)) * (W - m.l - m.r);
        const Y = (y) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);
        const kids = [];
        for (let i = 0; i <= 3; i++) {
          const v = y0 + ((y1 - y0) * i) / 3;
          kids.push(svg('line', { class: 'tr-grid', x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
          kids.push(svg('text', { class: 'tr-axis', x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, nf3.format(Math.round(v * 1000) / 1000)));
        }
        const step = Math.max(1, Math.ceil((x1 - x0) / 6));
        for (let x = Math.ceil(x0); x <= x1; x += step) kids.push(svg('text', { class: 'tr-axis', x: X(x), y: H - 8, 'text-anchor': 'middle' }, String(x)));
        if (bestEpoch != null && bestEpoch >= x0 && bestEpoch <= x1) kids.push(svg('line', { class: 'tr-best', x1: X(bestEpoch), x2: X(bestEpoch), y1: m.t, y2: H - m.b }));
        series.forEach((s, i) => {
          const pts = s.points.filter(([, y]) => Number.isFinite(y));
          if (!pts.length) return;
          kids.push(svg('polyline', { class: `tr-line s${i + 1}`, fill: 'none', points: pts.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(' ') }));
          if (pts.length <= 40) pts.forEach(([x, y]) => kids.push(svg('circle', { class: `tr-pt s${i + 1}`, cx: X(x), cy: Y(y), r: x === bestEpoch ? 4.5 : 2.8 }, svg('title', null, `${t('tr.epoch_n', { n: x })}: ${s.name} ${nf3.format(y)}`))));
        });
        const last = series.map((s) => { const p = s.points[s.points.length - 1]; return p ? `${s.name} ${nf3.format(p[1])}` : ''; }).filter(Boolean).join(', ');
        const chart = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'tr-svg', role: 'img', 'aria-label': `${title}: ${last}` }, ...kids);
        return el('figure', { class: 'tr-chart' },
          el('figcaption', null, el('strong', null, title)),
          chart,
          el('ul', { class: 'tr-legend' },
            ...series.map((s, i) => (s.points.some(([, y]) => Number.isFinite(y)) ? [s, i] : null)).filter(Boolean).map(([s, i]) => el('li', null, el('span', { class: `tr-swatch s${i + 1}`, 'aria-hidden': 'true' }), s.name)),
            bestEpoch != null ? el('li', null, el('span', { class: 'tr-swatch best', 'aria-hidden': 'true' }), t('tr.best_epoch_n', { n: bestEpoch })) : null));
      }

      function confusionTable(matrix, classes) {
        const n = classes.length;
        const rowSum = matrix.map((row) => row.reduce((a, b) => a + b, 0));
        const colSum = classes.map((_, j) => matrix.reduce((a, row) => a + (row[j] || 0), 0));
        let keep = classes.map((_, i) => i).filter((i) => rowSum[i] > 0 || colSum[i] > 0);
        if (!keep.length) return null;
        const LIMIT = 40;
        const total = keep.length;
        if (total > LIMIT) keep = keep.sort((a, b) => rowSum[b] + colSum[b] - rowSum[a] - colSum[a]).slice(0, LIMIT).sort((a, b) => a - b);
        const cell = (i, j) => {
          const v = (matrix[i] || [])[j] || 0;
          const share = rowSum[i] ? v / rowSum[i] : 0;
          return el('td', { class: `tr-cell${i === j ? ' diag' : ''}${v === 0 ? ' zero' : ''}`, style: v ? `--share:${Math.round(share * 55)}%` : null, title: `${classes[i]} → ${classes[j]}: ${v}` }, v ? num(v) : '·');
        };
        return el('div', { class: 'tr-block' },
          el('h4', null, t('tr.confusion')),
          el('p', { class: 'muted tr-note' }, t('tr.confusion_help'), total > LIMIT ? ` ${t('tr.shown_of', { n: LIMIT, total: num(total) })}` : ''),
          el('div', { class: 'tr-table-wrap' }, el('table', { class: 'tr-table tr-confusion' },
            el('caption', { class: 'sr-only' }, t('tr.confusion')),
            el('thead', null, el('tr', null, el('th', { scope: 'col', class: 'corner' }, t('tr.actual_vs_pred')), ...keep.map((j) => el('th', { scope: 'col', title: classes[j] }, classes[j])))),
            el('tbody', null, ...keep.map((i) => el('tr', null, el('th', { scope: 'row', title: classes[i] }, classes[i]), ...keep.map((j) => cell(i, j))))))),
          n > keep.length && total <= LIMIT ? el('p', { class: 'muted tr-note' }, tn('tr.hidden_classes', n - keep.length)) : null);
      }

      function perClassTable(per) {
        const rows = Object.entries(per || {}).filter(([, v]) => v && typeof v === 'object');
        if (!rows.length) return null;
        const order = ['precision', 'recall', 'f1', 'support'];
        const keys = [];
        rows.forEach(([, v]) => Object.keys(v).forEach((k) => { if (!keys.includes(k) && typeof v[k] === 'number') keys.push(k); }));
        keys.sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
        const shown = rows.filter(([, v]) => !('support' in v) || v.support > 0 || keys.some((k) => k !== 'support' && v[k] > 0));
        const fmtCell = (k, v) => (k === 'support' || (Number.isInteger(v) && v > 1) ? num(v) : nf3.format(v));
        return el('div', { class: 'tr-block' },
          el('h4', null, t('tr.per_class')),
          el('div', { class: 'tr-table-wrap' }, el('table', { class: 'tr-table' },
            el('thead', null, el('tr', null, el('th', { scope: 'col' }, t('tr.class')), ...keys.map((k) => el('th', { scope: 'col' }, k === 'support' ? t(keys.includes('iou') ? 'tr.support.pixels' : keys.includes('ap50') ? 'tr.support.objects' : 'tr.metric.support') : metricName(k))))),
            el('tbody', null, ...shown.map(([name, v]) => el('tr', null, el('th', { scope: 'row', title: name }, name), ...keys.map((k) => el('td', null, v[k] == null ? '' : fmtCell(k, v[k])))))))));
      }

      function partResults(part) {
        const scalars = Object.entries(part).filter(([, v]) => typeof v === 'number');
        const classes = part.classes || Object.keys(part.per_class || {});
        return el('div', { class: 'tr-part' },
          scalars.length ? el('dl', { class: 'tr-stats' }, ...scalars.map(([k, v]) => el('div', { class: 'tr-stat' }, el('dt', null, metricName(k)), el('dd', null, k === 'images' ? num(v) : nf3.format(v))))) : null,
          Array.isArray(part.confusion_matrix) ? confusionTable(part.confusion_matrix, classes) : null,
          perClassTable(part.per_class));
      }

      // ---------------------------------------------------------------- YAML (built here from the config object)
      function yamlScalar(v) {
        if (v === null || v === undefined) return 'null';
        if (typeof v === 'boolean' || typeof v === 'number') return String(v);
        const s = String(v);
        return /^[A-Za-z][\w.-]*$/.test(s) && !/^(true|false|null|yes|no|on|off|y|n)$/i.test(s) ? s : JSON.stringify(s);
      }
      function toYaml(obj, pad = '') {
        const lines = [];
        for (const [k, v] of Object.entries(obj)) {
          if (Array.isArray(v)) lines.push(`${pad}${k}: [${v.map(yamlScalar).join(', ')}]`);
          else if (v && typeof v === 'object') {
            if (Object.keys(v).length) { lines.push(`${pad}${k}:`); lines.push(toYaml(v, `${pad}  `)); } else lines.push(`${pad}${k}: {}`);
          } else lines.push(`${pad}${k}: ${yamlScalar(v)}`);
        }
        return lines.join('\n');
      }
      async function copyText(text) {
        try { await navigator.clipboard.writeText(text); return true; } catch (e) { /* falls back below */ }
        const area = el('textarea', { class: 'sr-only', 'aria-hidden': 'true', tabindex: '-1' }, text);
        document.body.appendChild(area);
        area.select();
        let ok = false;
        try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
        area.remove();
        return ok;
      }

      // ---------------------------------------------------------------- run detail
      function openDetail(number) {
        const body = el('div', { class: 'dialog-body tr-detail' });
        const live = el('p', { class: 'sr-only', 'aria-live': 'polite' });
        const dlg = el('dialog', { class: 'dialog wide tr-dialog', 'aria-label': t('tr.detail_title', { n: number }) }, body, live);
        let data = null;
        let sig = '';
        let part = null;
        let dtimer = null;
        let lastLive = '';

        function configBlock(r) {
          if (r.config_error) return el('section', { class: 'tr-section', 'aria-labelledby': 'trd-config' }, el('h3', { id: 'trd-config', class: 'section-title' }, t('tr.config')), configError(r));
          const c = r.config || {};
          const m = modelOf(c.model || r.model);
          const params = Object.entries(c.params || {});
          const grid = Object.entries(c.grid || {});
          const row = (k, v) => (v ? el('div', { class: 'tr-kv' }, el('dt', null, k), el('dd', null, v)) : null);
          const yaml = toYaml(c);
          const pkgs = Object.entries(r.packages || {}).map(([k, v]) => `${k} ${v}`).join(', ');
          return el('section', { class: 'tr-section', 'aria-labelledby': 'trd-config' },
            el('h3', { id: 'trd-config', class: 'section-title' }, t('tr.config')),
            el('dl', { class: 'tr-dl' },
              row(t('tr.model'), m ? m.label : c.model),
              row(t('tr.dataset'), datasetText(r)),
              row(t('tr.start'), startName(c.start || r.start)),
              row(t('tr.mode'), c.mode === 'grid' ? t('tr.mode_grid') : t('tr.mode_single')),
              row(t('tr.select_metric'), metricName(c.metric || (m && m.default_metric) || '')),
              row(t('tr.device'), deviceOf(r)),
              row(t('tr.duration'), durationOf(r)),
              row(t('tr.started'), r.started ? fmtDateTime(r.started) : ''),
              pkgs ? el('div', { class: 'tr-kv wide' }, el('dt', null, t('tr.packages')), el('dd', { class: 'tr-pkgs' }, pkgs)) : null),
            params.length || grid.length
              ? el('ul', { class: 'tr-chips', 'aria-label': t('tr.params') },
                ...params.map(([k, v]) => el('li', null, el('span', { class: 'muted' }, `${paramName(k)} `), typeof v === 'string' ? choiceName(v) : plain(v))),
                ...grid.map(([k, v]) => el('li', { class: 'grid' }, el('span', { class: 'muted' }, `${paramName(k)} `), `[${v.map((x) => (typeof x === 'string' ? choiceName(x) : plain(x))).join(', ')}]`)))
              : null,
            el('details', { class: 'tr-yaml' },
              el('summary', null, t('tr.yaml')),
              el('div', { class: 'tr-yaml-bar' },
                el('button', { type: 'button', class: 'btn small', 'data-act': 'copy', onclick: async () => { toast(t((await copyText(yaml)) ? 'tr.yaml_copied' : 'tr.yaml_failed'), 'info', 2500); } }, icon('copy', 14), t('tr.copy_yaml'))),
              el('pre', { class: 'tr-pre', tabindex: '0', 'aria-label': t('tr.yaml') }, yaml)));
        }

        function progressBlock(r) {
          if (!ACTIVE(r)) return r.status === 'failed' && r.error ? el('p', { class: 'tr-fail' }, icon('error', 16), el('span', null, r.error)) : null;
          const pct = percentOf(r);
          return el('div', { class: 'tr-live' },
            el('p', { class: 'tr-live-line' }, el('strong', null, phaseText(r)), r.message ? el('span', { class: 'muted' }, ` · ${r.message}`) : null),
            pct != null ? el('progress', { class: 'tr-progress', max: '100', value: String(pct), 'aria-label': t('common.progress') }) : null);
        }

        let childSel = null;   // training (child) whose curves and results are shown; null = the run's own
        let childData = null;
        const valueText = (k, v) => (k === 'start' ? startName(v) : typeof v === 'string' ? choiceName(v) : plain(v));
        const childLabel = (c) => [
          Object.entries(c.values || {}).map(([k, v]) => `${k === 'start' ? t('tr.start') : paramName(k)} ${valueText(k, v)}`).join(', '),
          c.fold != null ? t('ds.fold', { n: Number(c.fold) + 1 }) : ''].filter(Boolean).join(' · ') || c.id;

        async function showChild(id) {
          if (!id) { childSel = null; childData = null; sig = ''; render(); return; }
          try {
            childData = await api('GET', `/projects/${projectId}/runs/${number}/children/${id}`);
            childSel = id;
            sig = '';
            render();
          } catch (err) { toastError(err); }
        }

        function trainingsBlock(r) {
          const kids = r.children || [];
          if (!kids.length) return null;
          const res = r.results || {};
          const metric = (r.config && r.config.metric) || res.metric || '';
          const names = Object.keys(kids[0].values || {});
          const perCombo = kids.filter((c) => c.combo === kids[0].combo).length;
          const shown = childSel || (ACTIVE(r) ? r.current : r.best_child);
          const viewBtn = (id, label) => el('button', {
            type: 'button', class: `btn small${shown === id ? ' primary' : ''}`, 'aria-pressed': String(shown === id),
            'aria-label': t('tr.show_curves_of', { name: label }),
            onclick: () => showChild(id === (ACTIVE(r) ? r.current : r.best_child) ? null : id) }, t(shown === id ? 'tr.showing' : 'tr.show_curves'));
          const head = el('tr', null,
            el('th', { scope: 'col' }, '#'),
            ...names.map((k) => el('th', { scope: 'col' }, k === 'start' ? t('tr.start') : paramName(k))),
            perCombo > 1 ? el('th', { scope: 'col' }, t('tr.per_fold')) : null,
            el('th', { scope: 'col' }, res.ranking && perCombo > 1 ? t('tr.mean_std', { m: metricName(metric) }) : metricName(metric)),
            el('th', { scope: 'col' }, el('span', { class: 'sr-only' }, t('tr.curves'))));
          let rows;
          if (res.ranking) {
            rows = res.ranking.map((row) => {
              const ids = row.children || [];
              const scores = row.scores || {};
              const pick = ids.filter((id) => id in scores).sort((a, b) => scores[b] - scores[a])[0] || null;
              const entry = kids.find((c) => c.id === (pick || ids[0])) || { values: row.values };
              const failed = ids.map((id) => kids.find((c) => c.id === id)).filter((c) => c && c.status === 'failed');
              const mean = row.mean == null ? '—' : perCombo > 1 ? `${nf3.format(row.mean)} ± ${nf3.format(row.std || 0)}` : nf3.format(row.mean);
              return el('tr', { class: row.combo === res.best_combo ? 'best' : null },
                el('th', { scope: 'row' }, row.mean == null ? '—' : String(row.rank)),
                ...names.map((k) => el('td', null, valueText(k, (row.values || {})[k]))),
                perCombo > 1 ? el('td', { class: 'muted' }, ids.map((id) => (id in scores ? nf3.format(scores[id]) : '—')).join(' · ')) : null,
                el('td', { class: 'num', title: failed.length ? failed.map((c) => c.error).join('\n') : null },
                  el('strong', null, mean), failed.length ? el('span', { class: 'tr-fail-note' }, ` ${tn('tr.failed_n', failed.length)}`) : null),
                el('td', null, pick ? viewBtn(pick, childLabel(entry)) : null));
            });
          } else {
            rows = kids.map((c, i) => el('tr', { class: c.id === r.current ? 'current' : null },
              el('th', { scope: 'row' }, String(i + 1)),
              ...names.map((k) => el('td', null, valueText(k, (c.values || {})[k]))),
              perCombo > 1 ? el('td', null, t('ds.fold', { n: Number(c.fold) + 1 })) : null,
              el('td', { class: 'num', title: c.error || null }, c.best && c.best.value != null ? nf3.format(c.best.value) : el('span', { class: 'muted' }, phaseText(c))),
              el('td', null, c.status !== 'queued' ? viewBtn(c.id, childLabel(c)) : null)));
          }
          return el('section', { class: 'tr-section', 'aria-labelledby': 'trd-children' },
            el('h3', { id: 'trd-children', class: 'section-title' }, res.ranking ? t('tr.ranking') : t('tr.trainings_title')),
            el('p', { class: 'muted tr-note' }, res.ranking ? t(perCombo > 1 ? 'tr.ranking_help_folds' : 'tr.ranking_help', { k: perCombo }) : tn('tr.trainings', kids.length)),
            el('div', { class: 'tr-table-wrap' }, el('table', { class: 'tr-table tr-rank' },
              el('caption', { class: 'sr-only' }, res.ranking ? t('tr.ranking') : t('tr.trainings_title')),
              el('thead', null, head), el('tbody', null, ...rows))));
        }

        function viewingNote(r) {
          const kids = r.children || [];
          if (!kids.length) return null;
          const id = childSel || (ACTIVE(r) ? r.current : r.best_child);
          const c = kids.find((x) => x.id === id);
          if (!c) return null;
          const key = childSel ? 'tr.viewing' : ACTIVE(r) ? 'tr.viewing_current' : 'tr.viewing_best';
          return el('p', { class: 'tr-viewing' }, icon('chart', 16), el('span', null, t(key, { name: childLabel(c) })),
            childSel ? el('button', { type: 'button', class: 'btn small', onclick: () => showChild(null) }, t(ACTIVE(r) ? 'tr.view_current' : 'tr.view_best')) : null);
        }

        function chartsBlock(r) {
          const rows = r.metrics || [];
          if (!rows.length) return null;
          const c = r.config || {};
          const m = modelOf(c.model || r.model);
          const metric = c.metric || (r.results && r.results.metric) || (m && m.default_metric) || '';
          const keyOf = rows.some((x) => `val_${metric}` in x) ? `val_${metric}` : Object.keys(rows[rows.length - 1]).find((k) => k.startsWith('val_') && k !== 'val_loss');
          const bestRow = [...rows].reverse().find((x) => x.best);
          const best = (r.results && r.results.best_epoch) || (r.best && r.best.epoch) || (bestRow && bestRow.epoch) || null;
          const pts = (k) => rows.filter((x) => x[k] != null).map((x) => [x.epoch, x[k]]);
          const loss = lineChart(t('tr.chart_loss'), [{ name: t('tr.train'), points: pts('train_loss') }, { name: t('tr.val'), points: pts('val_loss') }], best);
          const score = keyOf ? lineChart(`${metricName(metric || keyOf.slice(4))} (${t('tr.val').toLowerCase()})`, [{ name: metricName(keyOf.slice(4)), points: pts(keyOf) }], best) : null;
          return el('section', { class: 'tr-section', 'aria-labelledby': 'trd-curves' },
            el('h3', { id: 'trd-curves', class: 'section-title' }, t('tr.curves')),
            el('div', { class: 'tr-charts' }, loss, score));
        }

        function resultsBlock(r) {
          const res = r.results;
          if (!res) return null;
          const parts = ['val', 'test'].filter((k) => res[k] && typeof res[k] === 'object');
          if (!parts.length) return null;
          if (!parts.includes(part)) part = parts[0];
          const pick = parts.length > 1
            ? el('div', { class: 'seg tr-seg', role: 'radiogroup', 'aria-label': t('tr.results') },
              ...parts.map((k) => {
                const input = el('input', { type: 'radio', name: 'trd-part', id: `trd-part-${k}`, value: k, class: 'sr-only', checked: part === k, onchange: () => { part = k; sig = ''; render(); } });
                return el('label', { for: input.id, class: 'seg-opt' }, input, k === 'val' ? t('tr.val') : t('tr.test'));
              }))
            : null;
          return el('section', { class: 'tr-section', 'aria-labelledby': 'trd-results' },
            el('h3', { id: 'trd-results', class: 'section-title' }, t('tr.results')),
            res.best_epoch != null ? el('p', { class: 'muted' }, t('tr.best_summary', { best: res.best_epoch, n: res.epochs_run != null ? res.epochs_run : res.best_epoch }), res.stopped_early ? ` ${t('tr.stopped_early')}` : '') : null,
            pick, partResults(res[part]));
        }

        function render() {
          const r = data;
          const view = childSel && childData ? { ...childData, number: r.number } : r;
          const s = JSON.stringify(r) + part + (childSel ? JSON.stringify(childData) : '');
          if (s === sig) return;
          sig = s;
          const top = body.scrollTop;
          keepFocus(body, () => fill(body,
            el('div', { class: 'tr-detail-head' },
              el('h2', null, t('tr.detail_title', { n: r.number })),
              badge(r),
              el('button', { type: 'button', class: 'icon-btn', 'data-act': 'close', 'aria-label': t('common.close'), onclick: () => dlg.close() }, icon('close', 18))),
            progressBlock(r), trainingsBlock(r), viewingNote(r), chartsBlock(view), resultsBlock(view), configBlock(r)));
          body.scrollTop = top;
          const text = phaseText(r);
          if (text !== lastLive) { lastLive = text; live.textContent = text; }
        }

        async function refresh() {
          clearTimeout(dtimer);
          try {
            data = await api('GET', `/projects/${projectId}/runs/${number}`);
            if (childSel) childData = await api('GET', `/projects/${projectId}/runs/${number}/children/${childSel}`);
          } catch (err) {
            if (err.status === 404) { dlg.close(); toastError(err); return; }
            if (!data) { dlg.close(); toastError(err); return; }
          }
          if (!dlg.isConnected) return;
          render();
          if (ACTIVE(data)) dtimer = setTimeout(refresh, 2000);
        }

        dlg.addEventListener('close', () => { clearTimeout(dtimer); dlg.remove(); load(true); });
        document.body.appendChild(dlg);
        openModal(dlg);
        add(body, el('p', { class: 'muted' }, t('common.loading')));
        refresh().then(() => { const close = dlg.querySelector('[data-act="close"]'); if (close) close.focus(); });
      }

      // ---------------------------------------------------------------- assemble
      const newBtn = el('button', { type: 'button', class: 'btn primary', id: 'tr-new', onclick: once(() => openNew(newBtn)) }, icon('plus', 16), t('tr.new'));
      add(node,
        el('div', { class: 'ds-head' },
          el('div', null, el('h2', { id: 'h-training' }, t('step.training')), el('p', { class: 'muted' }, t('tr.intro'))),
          newBtn),
        announcer,
        listBox);

      panel.mount = () => load();
      panel.show = () => load(true);
      /** Step 4 sends the person here with a dataset version to train on. */
      panel.startWith = (version) => openNew(newBtn, version);
      /** The project changed (its task type may have): the models on offer and their names are asked again. */
      panel.refresh = () => {
        catalog = null;
        cards.clear();
        if (panel.mounted) load(true);
      };
      /** Run counts for the step's status line, before the panel is opened. */
      panel.count = async () => {
        try {
          runs = (await api('GET', `/projects/${projectId}/runs`)).runs || [];
          ctx.runCount = runs.length;
          ctx.runActive = runs.filter(ACTIVE).length;
          renderStepper();
        } catch (err) { /* the status stays "no runs" */ }
      };
      return panel;
    }

    // ---- assemble
    page.staleCount = staleCount;
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
