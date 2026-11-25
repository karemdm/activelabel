/* Step 2, Annotations: the summary of the project's labels, importing annotation files, and the tabs of
 * pre-annotation and of the external tools. */
import { taskName } from './names.js';
import { preannotateBlock } from './preannotate.js';
import { annotateBlock } from '../integrations/external.js';
import { toolName } from '../integrations/tools.js';
import { api } from '../shell/api.js';
import { confirmDialog } from '../shell/dialogs.js';
import { add, el, fill, icon } from '../shell/dom.js';
import { fmtDateTime, num, t, tn, tr } from '../shell/i18n.js';
import { toast, toastError } from '../shell/toast.js';
import { problemRow } from '../shell/widgets.js';

export function annotationsPanel(page) {
  const { ctx, projectId, refreshProject, tabsView } = page;
  const node = el('section', { class: 'panel', id: 'panel-annotations', hidden: true, 'aria-labelledby': 'h-annotations' });
  const panel = { node, mounted: false };
  const fileInput = el('input', { type: 'file', multiple: true, class: 'sr-only', id: 'ann-input', tabindex: '-1' });
  const reportBox = el('div', { class: 'ann-report', 'aria-live': 'polite' });
  const gtBox = el('section', { class: 'annot-section gt-box', 'aria-label': t('ann.current') });
  const backupsBox = el('section', { class: 'annot-section backups-box', 'aria-label': t('ann.backups_title') });
  let busy = false;
  const annotate = annotateBlock(page);

  // video tasks: files made for a video are matched to the frames of the chosen task
  let videoTasks = [];
  let chosenTask = '';
  const taskBox = el('div', { class: 'field ann-task', hidden: true });
  function drawTask() {
    taskBox.hidden = !videoTasks.length;
    if (!videoTasks.length) return;
    if (!videoTasks.some((x) => String(x.id) === chosenTask)) chosenTask = videoTasks.length === 1 ? String(videoTasks[0].id) : '';
    fill(taskBox,
      el('label', { for: 'ann-task' }, t('ann.for_task')),
      el('select', { id: 'ann-task', onchange: (e) => { chosenTask = e.target.value; } },
        el('option', { value: '', selected: chosenTask === '' }, t('ann.task_images')),
        ...videoTasks.map((x) => el('option', { value: String(x.id), selected: String(x.id) === chosenTask }, x.name))));
  }
  async function loadVideoTasks() {
    try {
      videoTasks = ((await api('GET', `/projects/${projectId}/tasks`)).tasks || []).filter((x) => x.source && x.source.video);
    } catch (err) { videoTasks = []; }
    drawTask();
  }

  const drop = el('div', { class: 'dropzone small', id: 'ann-drop' },
    icon('upload', 28),
    el('p', { class: 'drop-title' }, t('ann.drop_title')),
    el('p', { class: 'muted' }, t('ann.drop_help')),
    el('button', { type: 'button', class: 'btn primary', onclick: () => fileInput.click() }, t('ann.choose')));
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => upload([...e.dataTransfer.files]));
  fileInput.addEventListener('change', () => { upload([...fileInput.files]); fileInput.value = ''; });

  async function upload(files) {
    if (!files.length || busy) return;
    busy = true;
    fill(reportBox, el('p', { class: 'muted' }, t('ann.checking')));
    const form = new FormData();
    files.forEach((f) => form.append('files', f, f.webkitRelativePath || f.name));
    if (chosenTask) form.append('task', chosenTask);
    try {
      const report = await api('POST', `/projects/${projectId}/annotations`, { body: form });
      renderValidation(report);
    } catch (err) {
      fill(reportBox);
      drop.hidden = false;
      toastError(err);
    } finally {
      busy = false;
    }
  }

  function renderValidation(report) {
    drop.hidden = true;  // the report takes its place; "choose other files" is in the report
    const found = report.inferred_task;
    const proj = ctx.project.task;
    const counts = report.counts || {};
    const hasExisting = ((ctx.project.counts || {}).annotated || 0) > 0;
    let mode = 'merge';

    let taskLine;
    if (!found) taskLine = t('ann.task_none');
    else if (found === proj) taskLine = t('ann.task_same', { task: taskName(found) });
    else taskLine = t('ann.task_diff', { found: taskName(found), project: taskName(proj) });

    const classes = report.classes_found || [];
    const fresh = new Set(report.classes_new || []);
    const perClass = counts.labels_per_class || {};

    const optionMeta = {
      import: { label: t('option.import'), help: t('option.import.help'), primary: true },
      change_task_and_import: { label: t('option.change_task_and_import', { task: taskName(found) }), help: t('option.change_task_and_import.help', { task: taskName(found) }) },
      convert_and_import: { label: t('option.convert_and_import', { task: taskName(proj) }), help: t('option.convert_and_import.help', { task: taskName(proj) }) },
      discard: { label: t('option.discard'), help: t('option.discard.help') },
    };
    const options = (report.options || []).filter((o) => optionMeta[o]);
    const needsMode = (o) => o !== 'discard' && hasExisting;
    const nothing = !(counts.labels_total > 0);  // nothing the project could use: the way on is another file
    const other = el('button', { type: 'button', class: `btn${nothing ? ' primary' : ' small'}`, 'data-action': 'other', onclick: () => fileInput.click() }, t('ann.choose_other'));

    const modeBox = el('fieldset', { class: 'mode-box' },
      el('legend', null, t('ann.mode_title')),
      ...['merge', 'replace'].map((m) =>
        el('label', { class: 'radio-row' },
          el('input', { type: 'radio', name: 'ann-mode', value: m, checked: m === mode, onchange: () => { mode = m; } }),
          el('span', null, el('strong', null, t(`ann.mode_${m}`)), el('span', { class: 'muted' }, ` — ${t(`ann.mode_${m}.help`)}`)))));

    const optionList = el('ul', { class: 'options' },
      nothing ? el('li', null, other, el('span', { class: 'muted' }, t('ann.nothing_usable'))) : null,
      ...options.map((o) => {
        const meta = optionMeta[o];
        return el('li', null,
          el('button', { type: 'button', class: `btn ${meta.primary ? 'primary' : o === 'discard' ? 'danger-outline' : ''}`, 'data-action': o, onclick: (e) => apply(report, o, needsMode(o) ? mode : 'replace', e.currentTarget) }, meta.label),
          el('span', { class: 'muted' }, meta.help));
      }));

    fill(reportBox,
      el('div', { class: 'card report-card' },
        el('div', { class: 'row-between' }, el('h3', null, t('ann.report_title')), nothing ? null : other),
        el('p', { class: 'task-line' }, el('span', { class: 'badge' }, tr(`format.${report.format}`, report.format || '?')), ' ', taskLine),
        nothing ? null : el('dl', { class: 'facts' },
          el('div', null, el('dt', null, t('ann.matched')), el('dd', null, num(counts.images_matched || 0))),
          el('div', null, el('dt', null, t('ann.unlabeled')), el('dd', null, num(counts.images_without_labels || 0))),
          el('div', null, el('dt', null, t('ann.labels')), el('dd', null, num(counts.labels_total || 0)))),
        report.unmatched && report.unmatched.count
          ? el('p', { class: 'muted' }, tn('ann.unmatched', report.unmatched.count))
          : null,
        nothing ? null : el('h4', null, t('ann.classes_found')),
        nothing ? null : classes.length
          ? el('ul', { class: 'chips static' }, ...classes.map((c) => el('li', { class: `chip static${fresh.has(c) ? ' is-new' : ''}` },
              el('span', null, c),
              el('span', { class: 'count' }, num(perClass[c] || 0)),
              fresh.has(c) ? el('span', { class: 'new-mark' }, t('ann.new')) : null)))
          : el('p', { class: 'muted' }, t('ann.no_classes')),
        el('h4', null, t('ann.problems')),
        (report.problems || []).length
          ? el('ul', { class: 'plain problems' }, ...report.problems.map(problemRow))
          : el('p', { class: 'report-line ok' }, icon('check'), t('ann.no_problems')),
        el('h4', null, t('ann.what_now')),
        options.some(needsMode) ? modeBox : null,
        optionList));
  }

  async function apply(report, action, mode, button) {
    const all = reportBox.querySelectorAll('button');
    all.forEach((b) => { b.disabled = true; });
    try {
      if (action === 'discard') {
        await api('DELETE', `/projects/${projectId}/annotations/${report.token}`);
        fill(reportBox, el('p', { class: 'report-line ok' }, icon('check'), t('ann.discarded')));
        drop.hidden = false;
        return;
      }
      const summary = await api('POST', `/projects/${projectId}/annotations/${report.token}/apply`, { json: { action, mode } });
      renderSummary(summary || {}, report);
      drop.hidden = false;
      await refreshProject();
      loadGt();
    } catch (err) {
      all.forEach((b) => { b.disabled = false; });
      toastError(err);
    }
    if (button) button.blur();
  }

  function renderSummary(s, report) {
    /* what the person needs to know, in plain words; zero counts of optional facts are left out */
    const always = ['images_imported', 'labels_imported'];
    const optional = ['images_kept', 'converted', 'classes_added', 'problems'];
    const count = (v) => (Array.isArray(v) ? v.length : Number(v) || 0);
    const lines = [...always, ...optional]
      .filter((k) => k in s && (always.includes(k) || count(s[k]) > 0))
      .map((k) => el('div', null, el('dt', null, t(`summary.${k}`)), el('dd', null, num(count(s[k])))));
    fill(reportBox, el('div', { class: 'card report-card' },
      el('p', { class: 'report-line ok' }, icon('check'), el('strong', null, t('ann.applied'))),
      lines.length ? el('dl', { class: 'facts' }, ...lines) : null,
      /* the problems of the check stay in view: the summary counts them and points here */
      (report.problems || []).length ? el('ul', { class: 'plain problems' }, ...report.problems.map(problemRow)) : null));
  }

  /* the safety copies of the annotations: a closed list under the summary, each with Restore */
  async function loadBackups() {
    let items = [];
    try {
      items = ((await api('GET', `/projects/${projectId}/annotations/backups`)) || {}).items || [];
    } catch (err) {
      if (err.status !== 404) toastError(err);
    }
    if (!items.length) { fill(backupsBox); return; }
    const restore = async (copy, button) => {
      const when = fmtDateTime(copy.created);
      const ok = await confirmDialog({ title: t('ann.backup_restore_title'), text: t('ann.backup_restore_text', { when }), confirmLabel: t('ann.backup_restore'), danger: false });
      if (!ok) return;
      button.disabled = true;
      try {
        await api('POST', `/projects/${projectId}/annotations/backups/${encodeURIComponent(copy.name)}/restore`);
        toast(t('ann.backup_restored', { when }), 'success');
        refreshProject();
        loadGt();
      } catch (err) {
        button.disabled = false;
        toastError(err);
      }
    };
    const row = (copy) => el('li', { class: 'backup-row' },
      el('span', { class: 'backup-when', title: copy.created }, fmtDateTime(copy.created)),
      el('span', { class: 'muted' }, copy.reason ? t(`ann.backup_reason_${copy.reason}`) : t('ann.backup_reason_unknown')),
      el('span', { class: 'muted' }, copy.valid ? `${tn('ann.backup_images', copy.images)}, ${tn('ann.backup_labels', copy.labels)}` : t('ann.backup_unreadable')),
      copy.valid ? el('button', { type: 'button', class: 'btn small', onclick: (e) => restore(copy, e.currentTarget) }, t('ann.backup_restore')) : null);
    fill(backupsBox, el('details', { class: 'backups' },
      el('summary', null, tn('ann.backups_summary', items.length)),
      el('p', { class: 'muted' }, t('ann.backups_help')),
      el('ul', { class: 'plain backup-list' }, ...items.map(row))));
  }

  async function loadGt() {
    loadBackups();
    let data;
    try {
      data = await api('GET', `/projects/${projectId}/annotations`);
    } catch (err) {
      fill(gtBox);
      if (err.status !== 404) toastError(err);
      return;
    }
    data = data || {};
    const classes = Object.entries(data.labels_per_class || {}).map(([name, count]) => ({ name, count }));
    const labels = data.labels_total || 0;
    const images = data.images_annotated || 0;
    const emptyImages = data.images_empty || 0;   // reviewed with no objects
    const waitingReview = data.images_preannotated || 0;   // pre-annotations are counted apart until reviewed
    const waitingNote = waitingReview ? el('p', { class: 'report-line' }, icon('info'), tn('ann.to_review', waitingReview)) : null;
    if (!images && !labels && !emptyImages) {
      const way = (tab, key) => el('li', null, el('button', { type: 'button', class: 'btn small', onclick: () => annTabs.select(tab, true) }, t(`ann.way_${key}`)), el('span', { class: 'muted' }, t(`ann.way_${key}_help`)));
      fill(gtBox, el('h3', null, t('ann.current')), el('p', { class: 'muted' }, t('ann.current_none')), waitingNote,
        el('ul', { class: 'ann-ways' }, way('cvat', 'cvat'), way('fiftyone', 'fiftyone'), way('files', 'files'), way('preannotate', 'model')));
      return;
    }
    const tile = (label, value, extra = '') => el('div', { class: `stat-tile${extra}` }, el('dd', null, num(value)), el('dt', null, label));
    fill(gtBox,
      el('h3', null, t('ann.current')),
      el('dl', { class: 'stat-tiles' },
        tile(t('ann.tile_images'), images), tile(t('ann.tile_labels'), labels), tile(t('ann.tile_classes'), classes.length),
        waitingReview ? tile(t('ann.tile_review'), waitingReview, ' is-attention') : null),
      waitingNote,
      classes.length ? classBars(classes) : el('span'));
  }

  /* the classes with their counts: the biggest few, a search and "see all" when there are many */
  const BARS_SHOWN = 8;
  const BARS_SEARCH_FROM = 12;
  function classBars(classes) {
    const max = Math.max(1, ...classes.map((c) => c.count || 0));
    let all = false;
    let query = '';
    const list = el('ul', { class: 'bars' });
    const more = el('button', { type: 'button', class: 'btn small', onclick: () => { all = !all; draw(); } });
    const search = el('input', { type: 'search', class: 'bars-search', placeholder: t('ann.classes_search'), 'aria-label': t('ann.classes_search'), oninput: (e) => { query = e.target.value.trim().toLowerCase(); draw(); } });
    search.hidden = classes.length <= BARS_SEARCH_FROM;
    function draw() {
      const found = query ? classes.filter((c) => c.name.toLowerCase().includes(query)) : classes;
      const shown = query || all ? found : found.slice(0, BARS_SHOWN);
      fill(list, ...shown.map((c) =>
        el('li', null,
          el('span', { class: 'bar-label', title: c.name }, c.name),
          el('span', { class: 'bar-track' }, el('span', { class: 'bar-fill', style: `width:${Math.round(((c.count || 0) / max) * 100)}%` })),
          el('span', { class: 'bar-value' }, num(c.count || 0)))),
        found.length ? null : el('li', { class: 'muted' }, t('ann.no_classes')));
      more.hidden = !!query || classes.length <= BARS_SHOWN;
      more.textContent = all ? t('ann.show_less') : t('ann.show_all', { n: num(classes.length) });
    }
    draw();
    return el('div', { class: 'class-bars' }, el('div', { class: 'row-between' }, el('h4', null, t('ann.classes_title')), search), list, more);
  }

  const pre = preannotateBlock(page, {
    reviewCvat: (taskId) => { annTabs.select('cvat'); annotate.focusCvatTask(taskId); },
    reviewFiftyone: () => { annTabs.select('fiftyone'); annotate.focusScope('preannotated'); },
    changed: () => { loadGt(); refreshProject(); },
  });
  const annTabs = tabsView('annotations', t('step.annotations'), [
    { id: 'summary', label: t('ann.tab_summary'), nodes: [gtBox, backupsBox] },
    { id: 'preannotate', label: t('pre.tab'), nodes: [pre.node], onShow: () => pre.load() },
    { id: 'cvat', label: toolName('cvat'), nodes: [annotate.cvatSection] },
    { id: 'fiftyone', label: toolName('fiftyone'), nodes: [annotate.fiftyoneSection] },
    { id: 'files', label: t('ann.tab_files'), nodes: [el('section', { class: 'annot-section load-files', 'aria-labelledby': 'h-load' },
      el('h3', { id: 'h-load' }, t('ann.load_title')),
      el('p', { class: 'muted' }, t('ann.load_intro')),
      taskBox, drop, fileInput, reportBox)] },
  ], 'summary');
  add(node,
    el('h2', { id: 'h-annotations' }, t('step.annotations')),
    annTabs.node);
  panel.mount = () => { loadGt(); loadVideoTasks(); annotate.mount(); };
  panel.refresh = () => { loadGt(); loadVideoTasks(); annotate.refresh(); };
  panel.show = annotate.show;
  /* a finished activity opens the tab it belongs to, then its card */
  panel.preannotateWith = (number) => { pre.choose(number); annTabs.select('preannotate'); };
  panel.reveal = (sessionId) => { annTabs.select(sessionId === 'sync' ? 'cvat' : 'fiftyone'); annotate.reveal(sessionId); };
  return panel;
}
