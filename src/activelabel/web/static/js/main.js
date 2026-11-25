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
  CHUNK, ISSUES, SKIP_FILES, annotationStatus, issueName, partName, taskBadge, taskName, taskPicker,
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
    function curationPanel(page) {
      const node = el('section', { class: 'panel', id: 'panel-curation', hidden: true, 'aria-labelledby': 'h-curation' });
      const panel = { node, mounted: false };
      let summary = null;
      let issue = '';
      let view = 'active';
      let limitsOpen = false;
      let firstLoad = true;
      const ANY = '*';  // the quick filter "only with issues": the server filter issue=any
      const noticeBox = el('div', { class: 'cur-notice' });
      const chipsBox = el('div', { class: 'issue-chips', role: 'group', 'aria-label': t('cur.filter') });
      const histBox = el('div', { class: 'hists' });
      const advBox = el('div', { class: 'advanced', id: 'cur-limits', hidden: true });
      const actionsBar = el('div', { class: 'cur-actions' });
      const toolbar = el('div', { class: 'toolbar' });
      const tabs = el('div', { class: 'tabs', role: 'group', 'aria-label': t('cur.view') });
      const countText = el('span', { class: 'muted', 'aria-live': 'polite' });

      const grid = imageGrid({
        projectId,
        selectable: true,
        query: () => ({ status: view, issue: view === 'active' ? (issue === ANY ? 'any' : issue) : '', emptyText: view === 'excluded' ? t('cur.no_excluded') : t('cur.no_match') }),
        onSelect: (sel) => { countText.textContent = sel.size ? tn('cur.selected', sel.size) : ''; drawToolbar(); },
        onOpen: (items, i) => openLightbox(projectId, items, i),
      });

      ctx.curationSelected = () => [...grid.selected];

      const analyzeBtn = (recompute) =>
        el('button', { type: 'button', class: `btn${recompute ? '' : ' primary'}`, disabled: kindActive('analyze'), onclick: once(() => analyze(recompute), () => kindActive('analyze')) },
          recompute ? t('cur.reanalyze') : t('cur.analyze'));
      let analyzing = false;
      ctx.watchers.push(() => {
        if (panel.mounted && analyzing !== kindActive('analyze')) { analyzing = !analyzing; drawNotice(); }
      });

      async function analyze(recompute) {
        try {
          const job = await api('POST', `/projects/${projectId}/curation/analyze`, { json: { recompute: !!recompute } });
          toast(t('cur.started'), 'info', 3000);
          await jobStarted(job);
        } catch (err) {
          toastError(err);
        }
      }

      async function resetAnalysis() {
        const ok = await confirmDialog({ title: t('cur.reset_title'), text: t('cur.reset_text'), confirmLabel: t('cur.reset') });
        if (!ok) return;
        try {
          await api('DELETE', `/projects/${projectId}/curation`);
          toast(t('cur.reset_done'), 'success', 4000);
          await refreshProject();
          load();
        } catch (err) { toastError(err); }
      }

      async function load() {
        try {
          summary = await api('GET', `/projects/${projectId}/curation`);
        } catch (err) {
          summary = null;
          if (err.status !== 404) toastError(err);
        }
        if (summary && ctx.project) {
          const ids = new Set();
          Object.values(summary.issue_ids || {}).forEach((l) => l.forEach((x) => ids.add(x)));
                    ctx.project.counts = { ...(ctx.project.counts || {}), analyzed: (summary.totals || {}).measured || 0, with_issues: ids.size };
          renderStepper();
        }
        if (firstLoad && analyzed()) { firstLoad = false; if (flaggedIds()) issue = ANY; }  // problems first
        else if (issue === ANY && !flaggedIds()) issue = '';
        draw();
        grid.reload();
      }

      const analyzed = () => summary && ((summary.totals || {}).measured || 0) > 0;
      /* ids of every image with an issue (the union of the per-issue lists), or null when there is none */
      function flaggedIds() {
        const ids = new Set();
        Object.values((summary && summary.issue_ids) || {}).forEach((l) => l.forEach((x) => ids.add(x)));
        return ids.size ? [...ids] : null;
      }

      function draw() {
        drawNotice();
        drawActions();
        drawChips();
        drawHists();
        drawAdvanced();
        drawTabs();
        drawToolbar();
        histBox.hidden = !analyzed();
        chipsBox.hidden = !analyzed() || view !== 'active';
        advBox.hidden = !limitsOpen || !summary || !summary.thresholds;
      }

      function drawNotice() {
        if (analyzed()) {
          const totals = summary.totals || {};
          fill(noticeBox, el('p', { class: 'muted' }, t('cur.measured', { measured: num(totals.measured || 0), total: num(totals.active || 0) })));
        } else {
          fill(noticeBox, el('div', { class: 'callout' }, icon('info'),
            el('div', null, el('strong', null, t('cur.not_analyzed')), el('p', { class: 'muted' }, t('cur.intro')), analyzeBtn(false))));
        }
      }

      /* the actions of the step, at the top: the safe ones together, "reset" apart and quiet */
      function drawActions() {
        const limitsBtn = summary && summary.thresholds ? el('button', {
          type: 'button', class: 'btn', 'aria-expanded': String(limitsOpen), 'aria-controls': 'cur-limits',
          onclick: () => { limitsOpen = !limitsOpen; draw(); },
        }, t('cur.limits')) : null;
        redraw(actionsBar,
          analyzed() ? analyzeBtn(true) : null, openFoBtn, pullFoBtn, limitsBtn,
          analyzed() ? el('button', { type: 'button', class: 'btn quiet-danger cur-reset', onclick: resetAnalysis }, t('cur.reset')) : null);
      }

      function drawChips() {
        const totals = (summary && summary.totals) || {};
        const counts = (summary && summary.issues) || {};
        const mk = (key, label, n, help) =>
          el('button', {
            type: 'button', class: `chip-btn${issue === key ? ' active' : ''}`, 'aria-pressed': issue === key ? 'true' : 'false',
            title: help || null, 'data-issue': key, 'data-act': key,
            onclick: () => { issue = key; drawChips(); grid.reload(); },
          }, el('span', null, label), el('span', { class: 'count' }, num(n)));
        const flagged = flaggedIds();
        redraw(chipsBox,
          mk('', t('cur.all'), totals.active || 0, t('cur.all.help')),
          flagged ? mk(ANY, t('cur.only_issues'), flagged.length, t('cur.only_issues.help')) : null,
          ...ISSUES.concat(Object.keys(counts).filter((k) => !ISSUES.includes(k))).map((k) => mk(k, issueName(k), counts[k] || 0, tr(`issue.${k}.help`, ''))));
      }

      function drawHists() {
        const h = (summary && summary.histograms) || {};
        const shown = ['brightness', 'contrast', 'sharpness'].filter((m) => h[m] && (h[m].counts || []).length);
        fill(histBox, ...shown.map((m) => histogram(m, h[m])), shown.length ? el('p', { class: 'muted hist-legend' }, t('cur.hist_legend')) : null);
      }

      function histogram(metric, data) {
        const counts = data.counts;
        const edges = data.edges || [];
        const max = Math.max(1, ...counts);
        const w = 200;
        const hgt = 60;
        const bw = w / counts.length;
        const bars = counts.map((c, i) => {
          const bh = Math.round((c / max) * (hgt - 4));
          return `<rect x="${(i * bw + 0.5).toFixed(1)}" y="${hgt - bh}" width="${(bw - 1).toFixed(1)}" height="${bh}" rx="1"></rect>`;
        }).join('');
        const lo = edges.length ? edges[0] : null;
        const hi = edges.length ? edges[edges.length - 1] : null;
        const fmt = (x) => (x == null ? '' : nf1.format(x));
        const name = tr(`metric.${metric}`, metric);
        const label = `${name}: ${fmt(lo)} – ${fmt(hi)}; ${t('cur.hist_count', { n: num(max) })}`;
        return el('figure', { class: 'hist' },
          el('figcaption', null, el('strong', null, name), el('span', { class: 'muted' }, t('cur.hist_count', { n: num(max) }))),
          el('div', { html: `<svg viewBox="0 0 ${w} ${hgt}" width="100%" role="img" aria-label="${esc(label)}" preserveAspectRatio="none">${bars}</svg>` }),
          el('div', { class: 'hist-axis', 'aria-hidden': 'true' }, el('span', null, fmt(lo)), el('span', null, lo != null ? fmt((lo + hi) / 2) : ''), el('span', null, fmt(hi))));
      }

      function drawAdvanced() {
        const th = summary && summary.thresholds;
        if (!th) { fill(advBox); return; }
        const inputs = Object.entries(th).map(([k, v]) => {
          const id = `th-${k}`;
          return el('div', { class: 'field inline' }, el('label', { for: id }, tr(`threshold.${k}`, k)), el('input', { id, name: k, type: 'number', step: 'any', value: String(v) }));
        });
        const formError = el('p', { class: 'tr-form-error', role: 'alert', hidden: true });
        const form = el('form', { class: 'th-form', novalidate: true }, el('p', { class: 'muted' }, t('cur.thresholds_help')), el('div', { class: 'th-grid' }, ...inputs), formError,
          el('div', { class: 'form-actions' }, el('button', { type: 'submit', class: 'btn primary' }, t('cur.thresholds_apply'))));
        const fail = (message) => { formError.textContent = message; formError.hidden = false; };
        form.addEventListener('submit', async (e) => {
          e.preventDefault();
          formError.hidden = true;
          const body = {};
          for (const input of form.querySelectorAll('input')) {
            input.removeAttribute('aria-invalid');
            const value = input.value.trim();
            if (value === '' || !Number.isFinite(Number(value))) {  /* Number('') is 0: an emptied field must not become a threshold */
              input.setAttribute('aria-invalid', 'true');
              input.focus();
              fail(t('cur.threshold_number', { name: tr(`threshold.${input.name}`, input.name) }));
              return;
            }
            body[input.name] = Number(value);
          }
          try {
            summary = await api('PUT', `/projects/${projectId}/curation/thresholds`, { json: body });
            toast(t('cur.thresholds_saved'), 'success', 3000);
            await load();
          } catch (err) {
            fail(err.message);  /* the server explains a refused value (out of range) next to the fields */
          }
        });
        fill(advBox, form);
      }

      function drawTabs() {
        const excluded = ((ctx.project || {}).counts || {}).excluded || 0;
        redraw(tabs,
          ...['active', 'excluded'].map((v) =>
            el('button', {
              type: 'button', class: `tab${view === v ? ' active' : ''}`, 'aria-pressed': view === v ? 'true' : 'false', 'data-view': v, 'data-act': v,
              onclick: () => { view = v; draw(); grid.reload(); },
            }, v === 'active' ? t('cur.tab_active') : `${t('cur.tab_excluded')} (${num(excluded)})`)));
      }

      function drawToolbar() {
        const n = grid.selected.size;
        const buttons = [];
        /* the actions on a selection appear once something is selected (no greyed-out buttons) */
        if (view === 'active') {
          buttons.push(
            el('button', { type: 'button', class: 'btn', 'data-act': 'all', title: t('cur.select_all'), onclick: selectAll }, t('cur.select_all_short')),
            el('button', { type: 'button', class: 'btn', 'data-act': 'suggested', disabled: !analyzed(), 'aria-describedby': analyzed() ? null : 'cur-suggested-hint', title: t('cur.select_suggested'), onclick: selectSuggested }, t('cur.select_suggested_short')));
          if (n) {
            buttons.push(
              el('button', { type: 'button', class: 'btn', 'data-act': 'clear', onclick: () => grid.setSelected([]) }, t('cur.clear')),
              el('button', { type: 'button', class: 'btn primary', 'data-act': 'exclude', onclick: once(exclude) }, t('cur.exclude')));
          }
        } else {
          buttons.push(el('button', { type: 'button', class: 'btn', 'data-act': 'all', onclick: selectAll }, t('cur.select_all_excluded')));
          if (n) {
            buttons.push(
              el('button', { type: 'button', class: 'btn', 'data-act': 'clear', onclick: () => grid.setSelected([]) }, t('cur.clear')),
              el('button', { type: 'button', class: 'btn primary', 'data-act': 'restore', onclick: once(restore) }, t('cur.restore')),
              el('button', { type: 'button', class: 'btn danger-outline', 'data-act': 'delete', onclick: once(remove) }, t('cur.delete')));
          }
        }
        const needsAnalysis = view === 'active' && !analyzed();  // "suggested" comes from the analysis
        const hint = n && !needsAnalysis ? null : el('span', { class: 'muted cur-select-hint', id: needsAnalysis ? 'cur-suggested-hint' : null },
          t(needsAnalysis ? 'cur.suggested_needs_analysis' : view === 'active' ? 'cur.select_hint' : 'cur.select_hint_excluded'));
        redraw(toolbar, ...buttons, hint, countText);
      }

      async function selectAll() {
        try { grid.setSelected(await grid.allIds()); } catch (err) { toastError(err); }
      }
      async function selectSuggested() {
        try {
          const data = await api('GET', `/projects/${projectId}/curation/suggested`);
          const ids = Array.isArray(data) ? data : (data && data.ids) || [];
          grid.setSelected(ids);
          toast(ids.length ? tn('cur.suggested_n', ids.length) : t('cur.suggested_none'), 'info', 4000);
        } catch (err) { toastError(err); }
      }
      const after = async () => { await refreshProject(); await load(); };
      async function exclude() {
        const ids = [...grid.selected];
        try {
          const r = await api('POST', `/projects/${projectId}/images/exclude`, { json: { ids, reason: issue || 'manual' } });
          toast(tn('cur.excluded_n', (r && r.changed) ?? ids.length), 'success', 5000);
          await after();
        } catch (err) { toastError(err); }
      }
      async function restore() {
        const ids = [...grid.selected];
        try {
          const r = await api('POST', `/projects/${projectId}/images/restore`, { json: { ids } });
          toast(tn('cur.restored_n', (r && r.changed) ?? ids.length), 'success', 5000);
          await after();
        } catch (err) { toastError(err); }
      }
      async function remove() {
        const ids = [...grid.selected];
        const ok = await confirmDialog({
          title: t('cur.delete_title'), text: tn('cur.delete_text', ids.length), confirmLabel: t('cur.delete_confirm'),
        });
        if (!ok) return;
        try {
          const r = await api('POST', `/projects/${projectId}/images/delete`, { json: { ids, confirm: true } });
          toast(tn('cur.deleted_n', (r && r.deleted) ?? ids.length), 'success', 5000);
          await after();
        } catch (err) { toastError(err); }
      }

      async function openFiftyOne() {
        const view_ = issue ? `curation - ${issue}` : 'curation - quality issues';
        try {
          const job = await api('POST', `/projects/${projectId}/fiftyone`, { json: { view: view_ } });
          if (job && job.id) ctx.openWhenDone.add(job.id);
          toast(t('cur.fo_starting'), 'info', 4000);
          await jobStarted(job);
        } catch (err) { toastError(err); }
      }
      async function pullExclusions() {
        try {
          const r = await api('POST', `/projects/${projectId}/fiftyone/pull-exclusions`);
          toast(tn('cur.pulled_n', (r && r.excluded) || 0), 'success', 5000);
          await after();
        } catch (err) { toastError(err); }
      }

      const openFoBtn = el('button', { type: 'button', class: 'btn', onclick: once(openFiftyOne, () => kindActive('fiftyone')) }, t('cur.open_fo'));
      ctx.watchers.push(() => { openFoBtn.disabled = kindActive('fiftyone'); });
      const pullFoBtn = el('button', { type: 'button', class: 'btn', title: t('cur.fo_hint'), onclick: once(pullExclusions) }, t('cur.pull_fo'));

      add(node,
        el('h2', { id: 'h-curation' }, t('step.curation')),
        noticeBox,
        actionsBar,
        advBox,
        histBox,
        el('div', { class: 'cur-main' },
          tabs, chipsBox, toolbar, grid.node));
      panel.mount = () => load();
      panel.refresh = () => load();
      return panel;
    }

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


    // ---- step 4: dataset versions (frozen splits, see docs/DATASETS.md)
    /* images whose labels changed after the version was made (0 when none); the version keeps the old ones */
    const staleCount = (v) => { const s = v.stale || {}; return (s.changed_images || 0) + (s.added || 0) + (s.removed || 0); };

    function datasetsPanel(page) {
      const node = el('section', { class: 'panel', id: 'panel-datasets', 'aria-labelledby': 'h-datasets' });
      const panel = { node, mounted: false };
      const listBox = el('div', { class: 'ds-list' });
      let versions = [];
      const isClassification = () => ctx.project && ctx.project.task === 'classification';

      /* sizes: {train, val, test} or {test, folds: [n, ...]}; the preview may send counts or id lists */
      const count = (x) => (Array.isArray(x) ? x.length : Number(x) || 0);
      function normSizes(d) {
        if (d.sizes) return d.sizes;
        const src = d.parts || d;
        const out = {};
        ['train', 'val', 'test'].forEach((k) => { if (src[k] != null) out[k] = count(src[k]); });
        if (Array.isArray(d.folds)) out.folds = d.folds.map(count);
        return out;
      }
      /* one entry per part to draw: {key, label, n, cls, part, fold} */
      /* parts to show: the ones with images, and the ones that were asked for even if they ended up empty */
      function partItems(sizes, options) {
        const asked = ((options || {}).split) || {};
        const wanted = (p) => count(sizes[p]) > 0 || p === 'train' || Number(asked[p]) > 0;
        const items = [];
        if (Array.isArray(sizes.folds)) {
          sizes.folds.forEach((n, i) => items.push({ key: `fold${i}`, label: t('ds.fold', { n: i + 1 }), n: count(n), cls: `fold-${i % 2}`, fold: i }));
        } else {
          ['train', 'val'].forEach((p) => { if (sizes[p] != null && wanted(p)) items.push({ key: p, label: partName(p), n: count(sizes[p]), cls: p, part: p }); });
        }
        if (sizes.test != null && wanted('test')) items.push({ key: 'test', label: partName('test'), n: count(sizes.test), cls: 'test', part: 'test' });
        return items;
      }
      const totalOf = (items) => items.reduce((a, x) => a + x.n, 0);
      const pctOf = (n, total) => (total > 0 ? Math.round((n / total) * 100) : 0);

      function partsBar(items, label) {
        return el('div', { class: 'ds-bar', role: 'img', 'aria-label': label },
          ...items.filter((x) => x.n > 0).map((x) => el('span', { class: `ds-seg ds-${x.cls}`, style: `flex-grow:${x.n}`, title: `${x.label}: ${num(x.n)}` })));
      }
      const legend = (items) => el('ul', { class: 'ds-legend' }, ...items.map((x) => el('li', null,
        el('span', { class: `ds-dot ds-${x.cls}`, 'aria-hidden': 'true' }), `${x.label}: `, el('strong', null, num(x.n)),
        el('span', { class: 'muted' }, ` · ${pctOf(x.n, totalOf(items))}%`))));
      const barLabel = (items) => items.map((x) => `${x.label} ${num(x.n)}`).join(', ');

      /* background images of a stats object {part: {images, background, objects}} */
      const statsList = (stats) => (stats ? (Array.isArray(stats.folds) ? [...stats.folds, stats.test] : Object.values(stats)).filter((s) => s && typeof s === 'object') : []);
      const backgroundOf = (stats) => statsList(stats).reduce((a, s) => a + (Number(s.background) || 0), 0);
      function statFor(stats, item) {
        if (!stats) return {};
        if (item.fold != null) return (Array.isArray(stats.folds) ? stats.folds[item.fold] : stats[`fold${item.fold + 1}`] || stats[`fold_${item.fold + 1}`] || stats[`fold${item.fold}`]) || {};
        return stats[item.key] || {};
      }

      /* one warning per fact: an empty part says so once (not "class missing" for every class), and the
         proportion drift is said once */
      function cleanWarnings(warnings, items) {
        const empty = new Set(items.filter((x) => x.n === 0).map((x) => (x.fold != null ? `fold${x.fold}` : x.part)));
        const seen = new Set();
        const out = [];
        for (const w of warnings || []) {
          if (w.code === 'video_split') continue;
          const where = w.fold != null ? `fold${w.fold}` : w.part;
          let x = w;
          if (empty.has(where) && /class_missing|few_images/.test(String(w.code))) x = { code: 'part_empty', part: w.part, fold: w.fold };
          else if (/drift|group/.test(String(w.code))) x = { code: 'drift' };
          const key = JSON.stringify([x.code, x.part, x.fold, x.class]);
          if (!seen.has(key)) { seen.add(key); out.push(x); }
        }
        /* several classes missing from the same part: one warning listing them */
        const grouped = [];
        const byPart = new Map();
        for (const w of out) {
          if (w.code !== 'class_missing') { grouped.push(w); continue; }
          const where = w.fold != null ? `fold${w.fold}` : w.part;
          if (!byPart.has(where)) {
            byPart.set(where, { code: 'classes_missing', part: w.part, fold: w.fold, classes: [], available: [], avoidable: [] });
            grouped.push(byPart.get(where));
          }
          const group = byPart.get(where);
          group.classes.push(w.class);
          group.available.push(w.available);
          group.avoidable.push(w.avoidable);
        }
        return grouped.map((w) => (w.code === 'classes_missing' && w.classes.length === 1
          ? { code: 'class_missing', part: w.part, fold: w.fold, class: w.classes[0], available: w.available[0], avoidable: w.avoidable[0] } : w));
      }

      /* why a hold-out version cannot train a model (an empty training or validation part), or null */
      function trainBlockReason(sizes) {
        if (Array.isArray(sizes.folds)) return null;
        if (!count(sizes.train)) return t('ds.cannot_train_train');
        if (!count(sizes.val)) return t('ds.cannot_train_val');
        return null;
      }

      /* information about the videos that were cut into segments (not a warning) */
      const videoInfos = (warnings) => (warnings || []).filter((w) => w.code === 'video_split');
      /* parts of one video sit in training and in validation/test: the scores may be optimistic (shown prominently) */
      const sharedVideos = (warnings) => (warnings || []).find((w) => w.code === 'video_parts_shared') || null;
      const sharedAlert = (w, tag = 'div') => (w ? el(tag, { class: 'ds-shared', role: 'alert' }, icon('warning', 16), el('strong', null, t('ds.w_video_parts_shared'))) : null);
      /* the gap actually used for a video, from the version's or the preview's video list */
      const gapSecondsOf = (w, videos) => { const v = (videos || []).find((x) => x.video === w.video); return v && Number(v.gap_seconds) > 0 ? Number(v.gap_seconds) : 0; };
      function videoText(w, videos = null) {
        const parts = (w.parts || []).map((p) => (p === 'fold' ? t('ds.folds_word') : partName(p))).join(t('ds.and'));
        const vars = { name: w.name || w.video, n: num(w.segments), parts, gap: num(w.gap || 0) };
        const secs = gapSecondsOf(w, videos);
        const gapNote = secs ? ` ${t('ds.video_gap_s', { s: num(secs) })}` : '';
        return t(w.gap > 0 ? 'ds.w_video_split' : 'ds.w_video_split_nogap', vars) + gapNote;
      }

      /* whether a missing class could have been in every part (the API says; older versions do not) */
      function avoidText(w) {
        if (w.code === 'class_missing') {
          if (w.avoidable == null) return '';
          return w.avoidable ? tn('ds.w_avoidable', w.available || 0) : tn('ds.w_unavoidable', w.available || 0);
        }
        const flags = (w.avoidable || []).filter((x) => x != null);
        if (!flags.length) return '';
        const fixed = flags.filter((x) => x === false).length;
        if (fixed === 0) return t('ds.w_avoidable_all');
        return fixed === flags.length ? t('ds.w_unavoidable_all') : tn('ds.w_unavoidable_some', fixed, { total: num(flags.length) });
      }

      /* warnings in plain words; short = for the badges on a card */
      function warnText(w, items, short = false) {
        const part = w.fold != null ? t('ds.fold', { n: w.fold + 1 }) : partName(w.part);
        const code = String(w.code || '');
        if (code === 'class_missing') return short ? t('ds.b_class_missing', { class: w.class, part }) : `${t('ds.w_class_missing', { class: w.class, part })} ${avoidText(w)}`.trim();
        if (code === 'classes_missing') {
          const list = w.classes.slice(0, 6).join(', ') + (w.classes.length > 6 ? ` (+${w.classes.length - 6})` : '');
          return short ? tn('ds.b_classes_missing', w.classes.length, { part }) : `${tn('ds.w_classes_missing', w.classes.length, { part, list })} ${avoidText(w)}`.trim();
        }
        if (code === 'part_empty') return short ? t('ds.b_part_empty', { part }) : t('ds.w_part_empty', { part });
        if (code === 'few_images') return short ? t('ds.b_few_images', { part }) : t('ds.w_few_images', { part });
        if (code === 'labels_without_geometry' || code === 'multiple_classes' || code === 'keypoints_count_mismatch') {
          const vars = { n: num(w.count || 0), expected: num(w.expected || 0) };
          return short ? tn(`ds.b_${code}`, w.count || 0, vars) : tn(`ds.w_${code}`, w.count || 0, vars);
        }
        if (/drift|group/.test(code)) {
          const total = totalOf(items);
          const real = items.map((x) => pctOf(x.n, total)).join('/');
          const why = w.cause === 'video' ? '' : w.cause === 'duplicates' ? '_duplicates' : '_generic';  // only frames of a video talk about videos
          return short ? t('ds.b_group_drift', { real }) : t(`ds.w_group_drift${why}`, { real });
        }
        if (/dup|curation/.test(code)) return short ? t('ds.b_duplicates') : t('ds.w_duplicates');
        return tr(`ds.w_${code}`, code);
      }

      // ---------------------------------------------------------------- list
      function splitText(v) {
        const o = v.options || {};
        if (o.kfold) return t('ds.kfold_short', { k: o.kfold });
        const s = o.split || {};
        /* what was asked for; the real sizes are in the bar and the legend below (groups can make them differ) */
        return t('ds.split_asked', { split: [s.train, s.val, s.test].filter((x) => x != null).join(' / ') });
      }

      /* warnings go into a few groups: one line says how many, the list opens on demand */
      const WARN_GROUP = {
        class_missing: 'classes', classes_missing: 'classes', part_empty: 'size', few_images: 'size',
        labels_without_geometry: 'labels', multiple_classes: 'labels', keypoints_count_mismatch: 'labels',
      };
      const warnGroup = (w) => WARN_GROUP[w.code] || (/drift|group/.test(String(w.code)) ? 'balance' : 'other');
      function warnSummary(list) {
        const count = new Map();
        list.forEach((w) => count.set(warnGroup(w), (count.get(warnGroup(w)) || 0) + 1));
        const groups = [...count].map(([g, n]) => t(`ds.warn_g_${g}`, { n: num(n) })).join(' · ');
        return `${tn('ds.warn_count', list.length)}: ${groups}`;
      }

      function versionCard(v) {
        const sizes = normSizes(v);
        const items = partItems(sizes, v.options);
        const bg = backgroundOf(v.stats);
        const cleaned = cleanWarnings(v.warnings, items).filter((w) => w.code !== 'video_parts_shared');
        const warns = cleaned.map((w) => el('li', { class: 'ds-badge warn', title: warnText(w, items) }, icon('warning', 14), warnText(w, items, true)));
        const infoBadges = videoInfos(v.warnings).map((w) => el('li', { class: 'ds-badge', title: videoText(w, v.videos) }, icon('video', 14), t('ds.b_video_split', { n: num(w.segments) })));
        const id = `ds-v${v.version}`;
        const noTrain = trainBlockReason(sizes);
        const stale = staleCount(v);
        return el('li', { class: 'ds-card card', id, 'aria-labelledby': `${id}-title` },
          el('div', { class: 'ds-card-head' },
            el('h3', { id: `${id}-title` }, `v${v.version}`),
            el('span', { class: 'muted' }, fmtDateTime(v.created)),
            v.type ? taskBadge(v.type) : null,
            v.incomplete ? el('span', { class: 'ds-badge bad', title: t('ds.incomplete_help') }, icon('error', 14), t('ds.incomplete')) : null,
            el('button', { type: 'button', class: 'icon-btn danger ds-del', 'aria-label': t('ds.delete_aria', { n: v.version }), title: t('ds.delete'), onclick: () => remove(v) }, icon('trash', 18))),
          v.note ? el('p', { class: 'ds-note' }, v.note) : null,
          el('p', { class: 'muted ds-meta' }, splitText(v), bg > 0 ? ` · ${t('ds.background_count', { n: num(bg) })}` : ''),
          partsBar(items, barLabel(items)),
          legend(items),
          warns.length ? el('details', { class: 'ds-warn-group' },
            el('summary', null, icon('warning', 14), warnSummary(cleaned)),
            el('ul', { class: 'ds-badges', 'aria-label': t('ds.warnings') }, ...warns)) : null,
          sharedAlert(sharedVideos(v.warnings)),
          infoBadges.length ? el('ul', { class: 'ds-badges' }, ...infoBadges) : null,
          noTrain ? el('p', { class: 'ds-block', id: `${id}-block` }, icon('warning', 14), noTrain) : null,
          stale ? el('div', { class: 'ds-stale', role: 'status' },
            el('p', null, icon('warning', 14), el('strong', null, tn('ds.stale', stale, { n: num(stale) }))),
            el('button', { type: 'button', class: 'btn small ds-new-from', onclick: once(() => openNew(null, v)) }, icon('plus', 14), t('ds.new_from'))) : null,
          el('div', { class: 'ds-view' },
            el('span', { class: 'muted' }, t('ds.see_images')),
            ...items.filter((x) => x.n > 0).map((x) => el('button', { type: 'button', class: 'btn small', 'aria-label': t('ds.see_part', { n: v.version, part: x.label }),
              onclick: () => { ctx.panels.images.showDataset(v.version, x.part || null, x.fold != null ? x.fold : null); showTab('images', true); } }, x.label)),
            el('button', { type: 'button', class: 'btn small primary ds-train-with', disabled: !!v.incomplete || !!noTrain, 'aria-describedby': noTrain ? `${id}-block` : null,
              onclick: () => { showTab('training'); ctx.panels.training.startWith(v.version); } }, icon('chart', 14), t('ds.train_with'))));
      }

      function draw() {
        newBtn.hidden = !versions.length;
        fill(listBox, versions.length
          ? el('ul', { class: 'ds-cards' }, ...versions.map(versionCard))
          : emptyState('images', t('ds.empty'),
            el('button', { type: 'button', class: 'btn primary', onclick: once(() => openNew(newBtn)) }, icon('plus', 16), t('ds.new'))));
      }

      async function load() {
        try {
          const d = await api('GET', `/projects/${projectId}/datasets`);
          versions = (d.versions || []).slice().sort((a, b) => b.version - a.version);
        } catch (err) {
          toastError(err);
          return;
        }
        ctx.datasetCount = versions.length;
        draw();
        renderStepper();
      }

      async function remove(v) {
        const ok = await confirmDialog({ title: t('ds.delete_title', { n: v.version }), text: t('ds.delete_text'), confirmLabel: t('ds.delete') });
        if (!ok) return;
        const send = (extra) => api('DELETE', `/projects/${projectId}/datasets/${v.version}`, { json: { confirm: true, ...extra } });
        try {
          try {
            await send({});
          } catch (err) {
            /* finished runs used this version: say which, and delete only when the person accepts they cannot be repeated */
            if (err.code !== 'dataset_used_by_runs') throw err;
            const again = await confirmDialog({
              title: t('ds.delete_used_title', { n: v.version }),
              text: t('ds.delete_used_text', { runs: errorParams(err.params).runs }),
              confirmLabel: t('ds.delete_used_confirm'),
            });
            if (!again) return;
            await send({ confirm_runs: true });
          }
          toast(t('ds.deleted', { n: v.version }), 'success', 3000);
          ctx.panels.images.forgetDataset(v.version);
        } catch (err) {
          toastError(err);
        }
        await load();
      }

      // ---------------------------------------------------------------- new version form
      /* `from`: a version whose options fill the form ("new version from this one") */
      async function openNew(trigger, from = null) {
        let tasks;
        try {
          tasks = ((await api('GET', `/projects/${projectId}/tasks`)).tasks || []).filter((x) => x.active > 0);
        } catch (err) {
          toastError(err);
          return;
        }
        const cls = isClassification();
        const st = { tasks: new Set(tasks.map((x) => x.id)), mode: 'split', bg: false, maxK: null, blocked: false };
        let timer = null;
        let token = 0;

        const numInput = (id, value, min, max) => el('input', { type: 'number', id, inputmode: 'numeric', min: String(min), max: String(max), step: '1', value: String(value) });
        const train = numInput('ds-train', 70, 1, 100);
        const val = numInput('ds-val', 20, 0, 99);
        const test = numInput('ds-test', 10, 0, 99);
        const kInput = numInput('ds-k', 5, 2, 10);
        const kHelp = el('p', { class: 'muted ds-help', id: 'ds-k-help' });
        kInput.setAttribute('aria-describedby', 'ds-k-help');
        const allowBox = el('input', { type: 'checkbox', id: 'ds-allow-empty', onchange: () => syncCreate() });
        const ktestInput = numInput('ds-ktest', 10, 0, 90);
        const seed = el('input', { type: 'number', id: 'ds-seed', inputmode: 'numeric', step: '1', value: '42' });
        const note = el('input', { type: 'text', id: 'ds-note', maxlength: '500', autocomplete: 'off' });
        const bgBox = el('input', { type: 'checkbox', id: 'ds-bg', role: 'switch' });
        const videosSel = el('select', { id: 'ds-videos' }, el('option', { value: 'auto' }, t('ds.videos_auto')), el('option', { value: 'whole' }, t('ds.videos_whole')));
        const gapInput = el('input', { type: 'number', id: 'ds-gap', inputmode: 'decimal', min: '0', max: '3600', step: '0.1', value: '0.5' });
        /* the safety gap is a choice: on by default, off sends 0 seconds (frames of one video may then sit on both sides) */
        const gapBox = el('input', { type: 'checkbox', id: 'ds-gap-on', role: 'switch', checked: true });
        const gapField = el('div', { class: 'ds-gap-field' }, el('div', { class: 'field ds-num' }, el('label', { for: 'ds-gap' }, t('ds.gap')), gapInput),
          el('p', { class: 'muted ds-help' }, t('ds.gap_help')));
        const gapOff = el('p', { class: 'ds-gap-off', role: 'status', hidden: true }, icon('warning', 14), t('ds.gap_off_warn'));
        const syncGap = () => { gapField.hidden = !gapBox.checked; gapOff.hidden = gapBox.checked; };
        gapBox.addEventListener('change', () => { syncGap(); changed(); });
        const create = el('button', { type: 'submit', class: 'btn primary' }, t('ds.create'));
        if (from) {
          const o = from.options || {};
          const asked = new Set(o.tasks || []);
          if (asked.size) st.tasks = new Set(tasks.filter((x) => asked.has(x.id)).map((x) => x.id));
          st.bg = !!o.background && !cls;
          bgBox.checked = st.bg;
          seed.value = String(o.seed != null ? o.seed : 42);
          videosSel.value = o.videos || 'auto';
          const gapSeconds = o.gap_seconds != null ? Number(o.gap_seconds) : 0.5;
          gapBox.checked = gapSeconds > 0;
          gapInput.value = String(gapSeconds > 0 ? gapSeconds : 0.5);
          const sp = o.split || {};
          if (o.kfold) {
            st.mode = 'kfold';
            kInput.value = String(o.kfold);
            ktestInput.value = String(sp.test != null ? sp.test : 10);
          } else {
            [train.value, val.value, test.value] = [sp.train, sp.val, sp.test].map((x, i) => String(x != null ? x : [70, 20, 10][i]));
          }
        }
        /* a version whose training or validation part is empty is created only on an explicit "create anyway" */
        const syncCreate = () => { create.disabled = st.blocked && !allowBox.checked; };

        // tasks
        const taskBoxes = tasks.map((x) => {
          const box = el('input', { type: 'checkbox', id: `ds-task-${x.id}`, checked: st.tasks.has(x.id), onchange: (e) => { if (e.target.checked) st.tasks.add(x.id); else st.tasks.delete(x.id); changed(); } });
          return el('li', null, el('label', { for: box.id, class: 'ds-check' }, box, el('span', { class: 'ds-check-name' }, x.name), el('span', { class: 'muted' }, tn('tasks.meta_images', x.active))));
        });

        // ratios
        const ratioBar = el('div', { class: 'ds-bar', 'aria-hidden': 'true' });
        const sumText = el('p', { class: 'ds-sum', role: 'status' });
        const modeBox = el('div', { class: 'ds-ratios' });
        const raw = (inp) => parseInt(inp.value, 10);   // as typed: a negative or a too big value stays visible
        const int = (inp) => { const n = raw(inp); return Number.isFinite(n) ? Math.max(0, Math.min(100, n)) : NaN; };
        const validPct = (inp) => Number.isFinite(raw(inp)) && raw(inp) >= 0 && raw(inp) <= 100;
        const field = (label, input) => el('div', { class: 'field ds-num' }, el('label', { for: input.id }, label), input);

        function drawRatios() {
          const parts = st.mode === 'split'
            ? [['train', int(train)], ['val', int(val)], ['test', int(test)]]
            : [['fold-0', 100 - (int(ktestInput) || 0)], ['test', int(ktestInput) || 0]];
          fill(ratioBar, ...parts.map(([cl, n]) => el('span', { class: `ds-seg ds-${cl}`, style: `flex-grow:${Number.isFinite(n) ? n : 0}` })));
          if (st.mode === 'split') {
            const sum = [train, val, test].reduce((a, i) => a + (raw(i) || 0), 0);
            sumText.className = `ds-sum ${sum === 100 && [train, val, test].every(validPct) ? 'ok' : 'bad'}`;
            sumText.textContent = t('ds.sum', { n: sum });
          } else {
            sumText.className = 'ds-sum ok';
            sumText.textContent = t('ds.kfold_sum', { k: kInput.value || '?', train: 100 - (int(ktestInput) || 0), test: int(ktestInput) || 0 });
          }
        }
        function drawMode() {
          fill(modeBox,
            st.mode === 'split'
              ? el('div', { class: 'ds-nums' }, field(t('ds.pct_train'), train), field(t('ds.pct_val'), val), field(t('ds.pct_test'), test))
              : el('div', { class: 'ds-nums' }, field(t('ds.k'), kInput), field(t('ds.pct_test'), ktestInput)),
            st.mode === 'split' ? null : kHelp,
            ratioBar, sumText);
          drawRatios();
        }
        /* the three percentages stay linked: changing one adjusts the next so the sum is 100 */
        train.addEventListener('input', () => {
          const a = int(train);
          if (Number.isNaN(a)) return;
          const v = Math.min(int(val) || 0, 100 - a);
          val.value = String(v);
          test.value = String(100 - a - v);
        });
        val.addEventListener('input', () => {
          const v = int(val);
          if (Number.isNaN(v)) return;
          const m = Math.min(v, 100 - (int(train) || 0));
          if (m !== v) val.value = String(m);
          test.value = String(100 - (int(train) || 0) - m);
        });
        test.addEventListener('input', () => {
          const x = int(test);
          if (Number.isNaN(x)) return;
          const m = Math.min(x, 100 - (int(train) || 0));
          if (m !== x) test.value = String(m);
          val.value = String(100 - (int(train) || 0) - m);
        });

        /* the request body, or the reason it cannot be sent */
        function options() {
          if (!st.tasks.size) return { error: t('ds.err_tasks') };
          const sd = parseInt(seed.value, 10);
          if (!Number.isFinite(sd)) return { error: t('ds.err_seed') };
          const gap = gapBox.checked ? parseFloat(gapInput.value) : 0;
          if (!(gap >= 0 && gap <= 3600) || (gapBox.checked && gap === 0)) return { error: t('ds.err_gap') };
          const body = { tasks: [...st.tasks], background: cls ? false : st.bg, seed: sd, videos: videosSel.value, gap_seconds: gap };
          if (st.mode === 'split') {
            const s = { train: raw(train), val: raw(val), test: raw(test) };
            const bad = ['train', 'val', 'test'].find((k) => !validPct({ train, val, test }[k]));
            if (bad) return { error: t('ds.err_pct_range', { part: partName(bad) }) };
            if (s.train < 1) return { error: t('ds.err_train_min') };
            if (s.train + s.val + s.test !== 100) return { error: t('ds.err_sum_is', { n: s.train + s.val + s.test }) };
            body.split = s;
            body.kfold = null;
          } else {
            const k = parseInt(kInput.value, 10);
            const tp = int(ktestInput);
            if (st.maxK != null && st.maxK < 2) return { error: t('ds.k_none') };
            if (st.maxK != null && k > st.maxK) return { error: t('ds.err_k_max', { max: num(st.maxK) }) };
            if (!(k >= 2 && k <= 10)) return { error: t('ds.err_k') };
            if (!validPct(ktestInput) || tp > 90) return { error: t('ds.err_test_range') };
            body.split = { train: 100 - tp, val: 0, test: tp };
            body.kfold = k;
          }
          return { body };
        }

        // ---- live preview
        const previewBox = el('div', { class: 'ds-preview', 'aria-live': 'polite' });
        const showMsg = (text, cl = 'muted') => fill(previewBox, el('p', { class: cl, role: cl === 'ds-error' ? 'alert' : null }, text));

        /* the most folds the images allow (from the preview): limits K in the field and says why */
        function limitK(maxK) {
          st.maxK = Number.isFinite(maxK) ? maxK : null;
          if (st.maxK == null) return;
          kInput.max = String(Math.max(2, Math.min(10, st.maxK)));
          kHelp.textContent = st.maxK < 2 ? t('ds.k_none') : t('ds.k_max', { max: num(st.maxK) });
          kfoldRadio.disabled = st.maxK < 2;
        }

        function renderPreview(d) {
          const sizes = normSizes(d);
          const items = partItems(sizes, d.options);
          limitK(d.max_kfold);
          const blocked = (d.empty_parts || []).filter((p) => p === 'train' || p === 'val');
          st.blocked = blocked.length > 0;
          const reason = trainBlockReason(sizes);
          const emptyNote = blocked.length
            ? el('li', { class: 'report-line ds-warn ds-blocked', role: 'alert' }, icon('warning'),
              el('div', null,
                el('p', null, t('ds.empty_blocked', { parts: blocked.map((p) => lower(partName(p))).join(t('ds.and')) })),
                el('label', { for: allowBox.id, class: 'ds-check' }, allowBox, el('span', null, t('ds.allow_empty')))))
            : reason ? el('li', { class: 'report-line ds-warn ds-blocked' }, icon('warning'), reason) : null;
          const total = totalOf(items);
          const stats = d.stats || {};
          const per = items.map((x) => statFor(stats, x));
          const bgTotal = per.reduce((a, s) => a + (Number(s.background) || 0), 0);
          const names = [];
          per.forEach((s) => Object.keys(s.objects || {}).forEach((c) => { if (!names.includes(c)) names.push(c); }));
          const head = el('tr', null, el('th', { scope: 'col' }, ''), ...items.map((x) => el('th', { scope: 'col' }, x.label)));
          const row = (label, cells, cl = '') => el('tr', { class: cl }, el('th', { scope: 'row' }, label), ...cells.map((c) => el('td', null, num(c))));
          /* a class row: the count, the share of the class in each part, and a marked 0 in a part that has images */
          const classRow = (c, cells, zero) => {
            const sum = cells.reduce((a, n) => a + n, 0);
            return el('tr', { class: zero ? 'ds-rare' : '' }, el('th', { scope: 'row' }, c),
              ...cells.map((n, i) => (n === 0 && items[i].n > 0
                ? el('td', { class: 'ds-zero', title: t('ds.zero_help', { class: c, part: items[i].label }) }, icon('warning', 12), el('strong', null, '0'), el('span', { class: 'sr-only' }, ` ${t('ds.zero_sr')}`))
                : el('td', { title: sum ? t('ds.class_share', { pct: pctOf(n, sum) }) : null }, num(n), el('small', { class: 'ds-share' }, sum ? `${pctOf(n, sum)}%` : '')))));
          };
          /* the rarest classes first (those missing from a part before the others): they are the ones at risk */
          const classRows = names.map((c) => {
            const cells = per.map((s) => (s.objects || {})[c] || 0);
            const zero = cells.some((n, i) => n === 0 && items[i].n > 0);
            return { c, cells, zero, sum: cells.reduce((a, n) => a + n, 0) };
          }).sort((a, b) => (Number(b.zero) - Number(a.zero)) || (a.sum - b.sum) || a.c.localeCompare(b.c)).map((r) => classRow(r.c, r.cells, r.zero));
          const SHOWN = 8;
          let all = false;
          const body = el('tbody');
          const more = el('button', { type: 'button', class: 'btn small', 'aria-expanded': 'false', onclick: () => { all = !all; paint(); } });
          function paint() {
            fill(body,
              row(t('ds.images'), items.map((x, i) => (per[i].images != null ? per[i].images : x.n)), 'strong'),
              bgTotal > 0 ? row(t('ds.background'), per.map((s) => s.background || 0)) : null,
              classRows.length ? el('tr', { class: 'sub' }, el('th', { scope: 'colgroup', colspan: String(items.length + 1) }, cls ? t('ds.images_per_class') : t('ds.objects_per_class'))) : null,
              ...(all ? classRows : classRows.slice(0, SHOWN)));
            more.hidden = classRows.length <= SHOWN;
            more.textContent = all ? t('ds.show_less') : t('ds.show_all', { n: num(classRows.length) });
            more.setAttribute('aria-expanded', String(all));
          }
          paint();
          const warns = cleanWarnings(d.warnings, items).filter((w) => !(w.code === 'part_empty' && blocked.includes(w.part)))  // said above, with the choice
            .filter((w) => w.code !== 'video_parts_shared')
            .map((w) => el('li', { class: 'report-line ds-warn' }, icon('warning'), warnText(w, items)));
          const infos = videoInfos(d.warnings).map((w) => el('li', { class: 'report-line ds-info' }, icon('info'), videoText(w, d.videos)));
          fill(previewBox,
            el('h3', { class: 'section-title' }, t('ds.preview'), el('span', { class: 'muted' }, ` · ${tn('status.images_active', total)}`)),
            partsBar(items, barLabel(items)),
            legend(items),
            el('div', { class: 'ds-table-wrap' }, el('table', { class: 'ds-table' }, el('thead', null, head), body)),
            more,
            sharedAlert(sharedVideos(d.warnings)),
            warns.length || emptyNote ? el('ul', { class: 'ds-warns', 'aria-label': t('ds.warnings') }, emptyNote, ...warns) : el('p', { class: 'report-line ds-ok' }, icon('check'), t('ds.no_warnings')),
            infos.length ? el('ul', { class: 'ds-warns ds-infos' }, ...infos) : null);
        }

        async function runPreview() {
          const opts = options();
          if (opts.error) return;
          const mine = ++token;
          try {
            const d = await api('POST', `/projects/${projectId}/datasets/preview`, { json: opts.body });
            if (mine !== token) return;
            renderPreview(d);
            syncCreate();
          } catch (err) {
            if (mine !== token) return;
            create.disabled = true;  // a version the server refuses to preview cannot be created
            if (err.code === 'kfold_too_few_groups') limitK(err.params.groups);  // that many groups is the most folds
            showMsg(err.message || String(err), 'ds-error');
          }
        }
        function changed() {
          drawRatios();
          clearTimeout(timer);
          token += 1;
          allowBox.checked = false;  // a new choice of proportions needs a new "create anyway"
          st.blocked = false;
          const opts = options();
          create.disabled = !!opts.error;
          if (opts.error) { showMsg(opts.error, 'ds-error'); return; }
          showMsg(t('ds.previewing'));
          timer = setTimeout(runPreview, 400);
        }

        let kfoldRadio = null;
        const modeOpt = (mode, label) => {
          const input = el('input', { type: 'radio', name: 'ds-mode', id: `ds-mode-${mode}`, value: mode, class: 'sr-only', checked: st.mode === mode, onchange: () => { st.mode = mode; drawMode(); changed(); } });
          if (mode === 'kfold') kfoldRadio = input;
          return el('label', { for: input.id, class: 'seg-opt' }, input, label);
        };
        const bgRow = cls ? null : el('div', { class: 'field ds-switch' },
          el('label', { for: 'ds-bg', class: 'ds-check' }, bgBox, el('span', null, t('ds.background_switch'))),
          el('p', { class: 'muted ds-help' }, t('ds.background_help')));
        bgBox.addEventListener('change', () => { st.bg = bgBox.checked; changed(); });

        const form = el('form', { class: 'dialog-body form ds-form', method: 'dialog', novalidate: true },
          el('h2', null, t('ds.new_title')),
          el('div', { class: 'ds-form-grid' },
            el('div', { class: 'ds-opts' },
              el('fieldset', { class: 'field' }, el('legend', null, t('ds.tasks')), el('ul', { class: 'ds-tasks' }, ...taskBoxes)),
              el('fieldset', { class: 'field' }, el('legend', null, t('ds.split')),
                el('div', { class: 'seg', role: 'radiogroup', 'aria-label': t('ds.split') }, modeOpt('split', t('ds.mode_split')), modeOpt('kfold', t('ds.mode_kfold'))),
                modeBox),
              bgRow,
              el('details', { class: 'ds-adv field' }, el('summary', null, t('ds.advanced')),
                field(t('ds.seed'), seed), el('p', { class: 'muted ds-help' }, t('ds.seed_help')),
                field(t('ds.videos'), videosSel), el('p', { class: 'muted ds-help' }, t('ds.videos_help')),
                el('div', { class: 'field ds-switch' }, el('label', { for: 'ds-gap-on', class: 'ds-check' }, gapBox, el('span', null, t('ds.gap_switch')))),
                gapField, gapOff),
              el('div', { class: 'field' }, el('label', { for: 'ds-note' }, `${t('ds.note')} `, el('span', { class: 'muted' }, `(${t('step.optional')})`)), note)),
            el('aside', { class: 'ds-side', 'aria-label': t('ds.preview') }, previewBox)),
          el('div', { class: 'form-actions' }, create, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel'))));
        const dlg = el('dialog', { class: 'dialog wide ds-dialog', 'aria-label': t('ds.new_title') }, form);
        [seed, kInput, ktestInput, train, val, test, gapInput].forEach((i) => i.addEventListener('input', changed));
        videosSel.addEventListener('change', changed);
        kInput.addEventListener('change', () => {
          const k = parseInt(kInput.value, 10);
          if (st.maxK >= 2 && k > st.maxK) { kInput.value = String(st.maxK); changed(); }
        });

        form.addEventListener('submit', async (e) => {
          e.preventDefault();
          const opts = options();
          if (opts.error) return;
          create.disabled = true;
          try {
            const body = { ...opts.body, note: note.value.trim() };
            if (st.blocked && allowBox.checked) body.allow_empty = true;
            const v = await api('POST', `/projects/${projectId}/datasets`, { json: body });
            dlg.close();
            toast(t('ds.created', { n: v.version }), 'success', 4000);
            await load();
          } catch (err) {
            create.disabled = false;
            toastError(err);
          }
        });
        dlg.addEventListener('close', () => { clearTimeout(timer); token += 1; dlg.remove(); if (trigger && trigger.isConnected) trigger.focus(); });
        drawMode();
        syncGap();
        document.body.appendChild(dlg);
        openModal(dlg);
        changed();
        const first = dlg.querySelector('input');
        if (first) first.focus();
      }

      const newBtn = el('button', { type: 'button', class: 'btn primary', id: 'ds-new', onclick: once(() => openNew(newBtn)) }, icon('plus', 16), t('ds.new'));
      add(node,
        el('div', { class: 'ds-head' },
          el('div', null, el('h2', { id: 'h-datasets' }, t('step.datasets')), el('p', { class: 'muted' }, t('ds.intro'))),
          newBtn),
        listBox);

      panel.mount = load;
      panel.refresh = load;
      /** Number of versions for the step's status line, before the panel is opened. */
      panel.count = async () => {
        try {
          ctx.datasetCount = ((await api('GET', `/projects/${projectId}/datasets`)).versions || []).length;
          renderStepper();
        } catch (err) { /* the status stays "no versions" */ }
      };
      return panel;
    }

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
    // ---------------------------------------------------------------- step 6: saved models
    function modelsPanel(page) {
      const node = el('section', { class: 'panel', id: 'panel-models', 'aria-labelledby': 'h-models' });
      const panel = { node, mounted: false };
      const box = el('div', { class: 'md-body' });
      const announcer = el('p', { class: 'sr-only', 'aria-live': 'polite' });
      const nf3 = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 3 });
      const labels = {};
      let models = [];
      let timer = null;
      let sig = '';
      const metricName = (m) => tr(`tr.metric.${m}`, String(m).replace(/_/g, ' '));
      const startName = (s) => tr(`tr.start_${s}`, s);
      const archName = (m) => m.label || labels[m.model] || m.model;
      const isPublic = (m) => m.source === 'public';
      const busy = (m) => Object.values(m.exports || {}).some((e) => e && (e.status === 'queued' || e.status === 'running'));
      const scalarsOf = (part) => Object.entries(part || {}).filter(([, v]) => typeof v === 'number');
      const download = (m, kind) => `/api/projects/${projectId}/models/${m.number}/download/${kind}`;

      /* whether the catalogue has public models for this project type (asked once; null = not known yet) */
      let publicAvailable = null;
      async function ensurePublic() {
        if (publicAvailable !== null) return;
        try {
          publicAvailable = ((await api('GET', `/projects/${projectId}/models/public-catalog`)).models || []).length > 0;
        } catch (err) { publicAvailable = true; }  // the button stays and says why when it is used
      }

      async function ensureLabels() {
        if (Object.keys(labels).length) return;
        try {
          const cat = await api('GET', `/training/catalog?type=${encodeURIComponent(ctx.project.task)}`);
          (cat.models || []).forEach((x) => { labels[x.id] = x.label; });
        } catch (err) { /* the catalogue ids are shown instead */ }
      }

      async function change(m, body, message) {
        try {
          await api('PATCH', `/projects/${projectId}/models/${m.number}`, { json: body });
          if (message) { toast(message, 'success', 3000); announcer.textContent = message; }
        } catch (err) { toastError(err); }
        await load(true);
      }

      function editNote(m) {
        const input = el('textarea', { id: 'md-edit-note', rows: '3', maxlength: '500' }, m.note || '');
        const save = el('button', { type: 'submit', class: 'btn primary' }, t('common.save'));
        const dlg = el('dialog', { class: 'dialog', 'aria-labelledby': 'md-note-t' },
          el('form', { class: 'dialog-body form', method: 'dialog' },
            el('h2', { id: 'md-note-t' }, t('md.note_title', { n: m.number })),
            el('div', { class: 'field' }, el('label', { for: 'md-edit-note' }, t('md.note')), input),
            el('div', { class: 'form-actions' }, save, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel')))));
        dlg.querySelector('form').addEventListener('submit', async (e) => {
          e.preventDefault();
          dlg.close();
          await change(m, { note: input.value }, t('md.note_saved'));
        });
        dlg.addEventListener('close', () => dlg.remove());
        document.body.appendChild(dlg);
        openModal(dlg);
        input.focus();
      }

      async function exportOnnx(m) {
        try {
          await api('POST', `/projects/${projectId}/models/${m.number}/export`, { json: { format: 'onnx' } });
          announcer.textContent = t('md.export_queued');
        } catch (err) { toastError(err); }
        await load(true);
      }

      async function removeModel(m) {
        const text = [isPublic(m) ? t('md.delete_text_public') : t('md.delete_text'), m.current ? t('md.delete_text_current') : null].filter(Boolean).join(' ');
        const ok = await confirmDialog({ title: t('md.delete_title', { n: m.number }), text, confirmLabel: t('md.delete') });
        if (!ok) return;
        try {
          await api('DELETE', `/projects/${projectId}/models/${m.number}`, { json: { confirm: true } });
          if (m.run != null) ctx.weightless.add(m.run);
          toast(t('md.deleted', { n: m.number }), 'success', 3000);
        } catch (err) { toastError(err); }
        await load(true);
        ctx.panels.training.count();
        ctx.panels.training.refresh();
      }

      const ONNX_KINDS = ['classification', 'semantic'];   // detection graphs are not reliable in ONNX yet
      function onnxBlock(m) {
        const e = (m.exports || {}).onnx || {};
        if (!ONNX_KINDS.includes(m.task_kind) && e.status !== 'done') return el('span', { class: 'muted md-export' }, t('md.onnx_later'));
        if (e.status === 'queued' || e.status === 'running') {
          const text = e.status === 'queued' ? t('md.export_queued') : e.message && e.message.startsWith('train_env_install:') ? t('md.installing', { pkg: e.message.split(':')[1] }) : t('md.exporting');
          return el('span', { class: 'md-export busy' }, el('span', { class: 'tr-dot', 'aria-hidden': 'true' }), text);
        }
        if (e.status === 'done') {
          return el('a', { class: 'btn small', href: download(m, 'onnx'), download: '', 'data-act': 'onnx' }, icon('download', 14), t('md.download_onnx'), el('span', { class: 'muted' }, ` · ${fmtBytes(e.size || 0)}`));
        }
        return el('span', { class: 'md-export' },
          el('button', { type: 'button', class: 'btn small', 'data-act': 'export', onclick: () => exportOnnx(m) }, e.status === 'failed' ? t('md.retry_onnx') : t('md.export_onnx')),
          e.status === 'failed' ? el('span', { class: 'tr-fail' }, icon('error', 14), el('span', null, e.error === 'interrupted' ? t('md.interrupted') : e.error || '')) : null);
      }

      function card(m) {
        const res = m.results || {};
        const val = (res.val || {})[m.metric];
        const test = (res.test || {})[m.metric];
        const id = `model-${m.number}`;
        return el('li', { class: `tr-run card md-model${m.current ? ' current' : ''}`, id, 'aria-labelledby': `${id}-t` },
          el('div', { class: 'tr-run-head' },
            el('h3', { id: `${id}-t` }, `m${m.number}`),
            el('span', { class: 'tr-model' }, archName(m)),
            m.current
              ? el('span', { class: 'tr-badge done md-current' }, icon('star', 14), t('md.current'))
              : el('button', { type: 'button', class: 'btn small', 'data-act': 'current', onclick: () => change(m, { current: true }, t('md.made_current', { n: m.number })) }, t('md.make_current'))),
          isPublic(m)
            ? el('p', { class: 'muted tr-meta' }, el('span', { class: 'md-public' }, t('md.public_badge')), ` ${t('md.public_meta', { date: fmtDate(m.created) })}`)
            : el('p', { class: 'muted tr-meta' }, [t('tr.dataset_short', { n: m.dataset }), startName(m.start), t('md.from_run', { n: m.run }), fmtDate(m.created)].join(' · ')),
          isPublic(m)
            ? el('details', { class: 'md-mapping' }, el('summary', null, tn('md.classes_n', Object.keys(m.mapping || {}).length)),
              el('p', null, el('span', { class: 'muted' }, `${t('md.mapping')}: `),
                Object.entries(m.mapping || {}).map(([from, to]) => (from === to ? from : `${from} → ${to}`)).join(', ')))
            : el('p', { class: 'tr-facts' },
              val != null ? el('strong', null, `${metricName(m.metric)} ${nf3.format(val)}`) : null,
              test != null ? el('span', null, ` · ${t('md.test_value', { v: nf3.format(test) })}`) : null,
              el('span', { class: 'muted' }, ` · ${fmtBytes(m.size || 0)}`)),
          el('p', { class: 'md-note' },
            m.note ? el('span', null, m.note) : el('span', { class: 'muted' }, t('md.no_note')),
            el('button', { type: 'button', class: 'icon-btn', 'data-act': 'note', 'aria-label': t('md.edit_note_aria', { n: m.number }), title: t('md.edit_note'), onclick: () => editNote(m) }, icon('edit', 16))),
          el('div', { class: 'tr-actions' },
            busy(m) ? null : el('button', { type: 'button', class: 'btn small', 'data-act': 'preannotate', onclick: () => { showTab('annotations'); ctx.panels.annotations.preannotateWith(m.number); } }, icon('star', 14), t('md.preannotate')),
            isPublic(m) ? el('span', { class: 'muted md-export' }, t('md.public_weights')) : null,
            isPublic(m) ? null : el('a', { class: 'btn small', href: download(m, 'pytorch'), download: '', 'data-act': 'pt' }, icon('download', 14), t('md.download_pt')),
            isPublic(m) ? null : onnxBlock(m),
            busy(m) ? null : el('button', { type: 'button', class: 'icon-btn danger tr-del', 'data-act': 'delete', 'aria-label': t('md.delete_aria', { n: m.number }), title: t('md.delete'), onclick: () => removeModel(m) }, icon('trash', 18))));
      }

      function compareBlock() {
        const cols = models.filter((m) => !isPublic(m)).sort((a, b) => a.number - b.number);
        if (cols.length < 2) return null;
        const rows = [];
        for (const part of ['val', 'test']) {
          const keys = [];
          cols.forEach((m) => scalarsOf((m.results || {})[part]).forEach(([k]) => { if (!keys.includes(k) && k !== 'images') keys.push(k); }));
          keys.forEach((k) => rows.push({ label: t('md.row', { part: t(part === 'val' ? 'tr.val' : 'tr.test'), metric: metricName(k) }), values: cols.map((m) => ((m.results || {})[part] || {})[k]), lower: k === 'loss' }));
        }
        const perKey = ['f1', 'ap50_95', 'iou'].find((k) => cols.some((m) => Object.values(((m.results || {}).val || {}).per_class || {}).some((c) => c && k in c)));
        if (perKey) {
          const classes = [];
          cols.forEach((m) => Object.keys(((m.results || {}).val || {}).per_class || {}).forEach((c) => { if (!classes.includes(c)) classes.push(c); }));
          classes.forEach((c) => rows.push({ label: t('md.class_row', { cls: c, metric: metricName(perKey) }), values: cols.map((m) => ((((m.results || {}).val || {}).per_class || {})[c] || {})[perKey]), cls: true }));
        }
        rows.push({ label: t('md.size'), values: cols.map((m) => m.size), size: true, lower: true });
        const cell = (row, v, best) => el('td', { class: `num${best ? ' best' : ''}` }, v == null ? '—' : row.size ? fmtBytes(v) : nf3.format(v));
        return el('section', { class: 'tr-section md-compare', 'aria-labelledby': 'md-compare-t' },
          el('h3', { id: 'md-compare-t', class: 'section-title' }, t('md.compare')),
          el('p', { class: 'muted tr-note' }, t('md.compare_help')),
          el('div', { class: 'tr-table-wrap' }, el('table', { class: 'tr-table md-table' },
            el('caption', { class: 'sr-only' }, t('md.compare')),
            el('thead', null, el('tr', null, el('th', { scope: 'col' }, t('md.metric')),
              ...cols.map((m) => el('th', { scope: 'col' }, `m${m.number}`, m.current ? el('span', { class: 'md-star', title: t('md.current') }, icon('star', 12)) : null, el('span', { class: 'md-arch' }, archName(m)))))),
            el('tbody', null, ...rows.map((row) => {
              const nums = row.values.filter((v) => typeof v === 'number');
              const best = nums.length > 1 && new Set(nums).size > 1 ? (row.lower ? Math.min(...nums) : Math.max(...nums)) : null;   // a tie has no best
              return el('tr', { class: row.cls ? 'sub' : null }, el('th', { scope: 'row' }, row.label), ...row.values.map((v) => cell(row, v, best != null && v === best)));
            })))));
      }

      /* a redraw keeps the keyboard focus on the same button of the same model */
      function refocus(redraw) {
        const a = document.activeElement;
        const act = a && box.contains(a) && a.dataset ? a.dataset.act : null;
        const owner = act ? a.closest('li[id]') : null;
        const ownerId = owner ? owner.id : null;
        redraw();
        if (!act) return;
        const scope = (ownerId && document.getElementById(ownerId)) || box;
        const again = scope.querySelector(`[data-act="${act}"]`) || scope.querySelector('[data-act]');
        if (again) again.focus();
      }

      async function addPublic(trigger) {
        let data;
        try { data = await api('GET', `/projects/${projectId}/models/public-catalog`); } catch (err) { toastError(err); return; }
        if (!(data.models || []).length) { toast(t('md.public_none'), 'info', 5000); return; }
        const st = { id: data.models[0].id, map: {} };
        const modelOf = () => data.models.find((x) => x.id === st.id);
        const choices = el('div', { class: 'md-public-list', role: 'radiogroup', 'aria-label': t('md.public_model') });
        const mapBox = el('div', { class: 'md-map' });
        const note = el('textarea', { id: 'md-public-note', rows: '2', maxlength: '500' });
        const save = el('button', { type: 'submit', class: 'btn primary' }, t('md.public_add'));
        const err = el('p', { class: 'tr-form-error', role: 'alert', hidden: true });
        /* the first public category chosen for two project classes ({category, classes}), or null: a category goes to one class only */
        function clash() {
          const by = {};
          for (const [cls, from] of Object.entries(st.map)) if (from) (by[from] = by[from] || []).push(cls);
          const found = Object.entries(by).find(([, classes]) => classes.length > 1);
          return found ? { category: found[0], classes: found[1] } : null;
        }
        function showClash() {
          const c = clash();
          err.textContent = c ? t('md.mapping_duplicate', { category: c.category, classes: c.classes.join(', ') }) : '';
          err.hidden = !c;
          return !!c;
        }
        function suggested() {
          /* project class -> the public category suggested for it (same name) */
          const out = {};
          for (const [from, to] of Object.entries(modelOf().suggested || {})) if (!(to in out)) out[to] = from;
          return out;
        }
        function drawChoices() {
          fill(choices, ...data.models.map((m) => {
            const id = `md-pub-${m.id}`;
            const input = el('input', { type: 'radio', name: 'md-pub', id, value: m.id, checked: m.id === st.id, onchange: () => { st.id = m.id; st.map = suggested(); drawMap(); showClash(); } });
            return el('label', { class: `md-public-choice${m.id === st.id ? ' selected' : ''}`, for: id }, input,
              el('span', { class: 'md-public-text' }, el('strong', null, m.label),
                el('span', { class: 'muted' }, `${tr(`md.pub.${m.id}`, m.note)} · ${t('md.public_stats', { map: nf3.format(m.box_map), mb: num(m.size_mb) })}`)));
          }));
        }
        let mapOpen = false;
        const linked = () => data.project_classes.filter((cls) => st.map[cls]).length;
        const mapLine = () => t('md.mapping_summary', { n: num(linked()), total: num(data.project_classes.length) });
        function drawMap() {
          drawChoices();
          const m = modelOf();
          const line = el('span', null, mapLine());
          const rows = el('div', { class: 'md-map-rows' }, ...data.project_classes.map((cls) => {
              const id = `md-map-${cls}`;
              const sel = el('select', { id, class: 'tr-select', onchange: (e) => { st.map[cls] = e.target.value; line.textContent = mapLine(); showClash(); } },
                el('option', { value: '' }, t('md.mapping_none')),
                ...m.classes.map((c) => el('option', { value: c, selected: st.map[cls] === c }, c)));
              return el('div', { class: 'md-map-row' }, el('label', { for: id }, cls), sel);
            }));
          const box = el('details', { class: 'md-map', open: mapOpen || !linked() },
            el('summary', null, line, el('span', { class: 'md-map-review' }, t('md.mapping_review'))), rows);
          box.addEventListener('toggle', () => { mapOpen = box.open; });
          fill(mapBox, el('p', { class: 'hint' }, t('md.mapping_help')), box);
        }
        st.map = suggested();
        drawMap();
        const dlg = el('dialog', { class: 'dialog wide', 'aria-labelledby': 'md-pub-t' },
          el('form', { class: 'dialog-body form', method: 'dialog' },
            el('h2', { id: 'md-pub-t' }, t('md.public_title')),
            el('p', { class: 'muted' }, t('md.public_intro')),
            el('fieldset', { class: 'field' }, el('legend', null, t('md.public_model')), choices),
            el('fieldset', { class: 'field' }, el('legend', null, t('md.mapping')), mapBox),
            el('div', { class: 'field' }, el('label', { for: 'md-public-note' }, t('md.note'), el('span', { class: 'optional' }, ` · ${t('step.optional')}`)), note),
            el('p', { class: 'hint' }, t('md.public_license')),
            err,
            el('div', { class: 'form-actions' }, save, el('button', { type: 'button', class: 'btn', onclick: () => dlg.close() }, t('common.cancel')))));
        dlg.querySelector('form').addEventListener('submit', async (e) => {
          e.preventDefault();
          if (showClash()) return;
          const mapping = {};
          for (const [cls, from] of Object.entries(st.map)) if (from) mapping[from] = cls;
          if (!Object.keys(mapping).length) { err.textContent = t('md.mapping_empty'); err.hidden = false; return; }
          save.disabled = true;
          try {
            const m = await api('POST', `/projects/${projectId}/models/public`, { json: { public: st.id, mapping, note: note.value } });
            dlg.close();
            toast(t('md.public_added', { n: m.number }), 'success', 4000);
            await load(true);
          } catch (e2) { save.disabled = false; err.textContent = e2.message; err.hidden = false; }
        });
        dlg.addEventListener('close', () => { dlg.remove(); if (trigger) trigger.focus(); });
        document.body.appendChild(dlg);
        openModal(dlg);
      }

      function draw() {
        const s = JSON.stringify(models) + publicAvailable;
        if (s === sig) return;
        sig = s;
        publicBtn.hidden = publicAvailable === false;  // no public model for this type: nothing to add
        if (!models.length) {
          mdActions.hidden = true;  // the two ways to get a model are offered in the empty state instead
          fill(box, emptyState('star', publicAvailable === false ? t('md.empty_no_public') : t('md.empty'),
            publicAvailable === false ? null : el('button', { type: 'button', class: 'btn primary', onclick: (e) => addPublic(e.currentTarget) }, t('md.public_add_button')),
            el('a', { class: publicAvailable === false ? 'btn primary' : 'btn', href: '#training' }, t('md.go_training'))));
          return;
        }
        mdActions.hidden = false;
        refocus(() => fill(box, ...[
          el('ul', { class: 'tr-cards' }, ...models.map(card)),
          compareBlock(),
          models.some((m) => !isPublic(m)) ? el('p', { class: 'muted md-help' }, icon('info', 14), t('md.export_help')) : null,
        ].filter(Boolean)));
      }

      async function load(quiet = false) {
        clearTimeout(timer);
        try {
          await ensureLabels();
          await ensurePublic();
          const d = await api('GET', `/projects/${projectId}/models`);
          models = d.models || [];
          ctx.modelCount = models.length;
          ctx.currentModel = d.current;
          renderStepper();
          draw();
        } catch (err) { if (!quiet) toastError(err); }
        if (!node.hidden && models.some(busy)) timer = setTimeout(() => load(true), 2000);
      }

      const publicBtn = el('button', { type: 'button', class: 'btn', onclick: (e) => addPublic(e.currentTarget) }, t('md.public_add_button'));
      const mdActions = el('div', { class: 'md-head-actions' },
        publicBtn,
        el('a', { class: 'btn', href: '#training' }, t('md.go_training')));
      add(node,
        el('div', { class: 'ds-head' },
          el('div', null, el('h2', { id: 'h-models' }, t('step.models')), el('p', { class: 'muted' }, t('md.intro'))),
          mdActions),
        announcer,
        box);
      panel.mount = () => load();
      panel.show = () => load(true);
      /** The project changed (its task type may have): model names come from the new catalogue. */
      panel.refresh = () => {
        for (const id of Object.keys(labels)) delete labels[id];
        publicAvailable = null;
        if (panel.mounted) load(true);
      };
      panel.count = async () => {
        try {
          const d = await api('GET', `/projects/${projectId}/models`);
          ctx.modelCount = (d.models || []).length;
          ctx.currentModel = d.current;
          renderStepper();
        } catch (err) { /* the status stays "no models" */ }
      };
      return panel;
    }

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
